"""
Precompute per-image outputs for the four frozen experts and write one JSON
per split. The inference pipeline consumes these caches via
`precomputed.use_precomputed: true` in the config.

This script runs the experts serially over the manifest. For large datasets,
shard the manifest and run several copies in parallel on different GPUs.

Outputs (one JSON list per split, each item keyed by `image_id`):
    <cache_dir>/<split>_priorrg.json    # image_id + study_id + task_id + view + report_text + chexbert_labels
    <cache_dir>/<split>_medgemma.json   # image_id + report_text + chexbert_labels
    <cache_dir>/<split>_rad_dino.json   # image_id + per-disease probs
    <cache_dir>/<split>_convnext.json   # image_id + per-disease probs
    <cache_dir>/<split>_gt_labels.json  # image_id + 14-class CheXbert vector of the reference report
                                          (consumed by training/dataset.py)

The PriorRG cache carries `study_id` / `task_id` / `view` so that
`training/dataset.py` can group images by study without re-reading the
manifest. `gt_labels.json` is built by running CheXbert on each study's
reference report and broadcasting the 14-class vector to every image in that
study; the supervision target of paper Eq. 12.

Usage
-----
    python scripts/precompute_experts.py --split test --config configs/local.yaml \
        --experts priorrg medgemma rad_dino convnext gt_labels
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.io import load_config, load_split, get_image_path   # noqa: E402


def _save(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[precompute] wrote {len(records)} records → {path}")


def precompute_priorrg(items, cfg, image_root):
    from src.models.priorrg_wrapper import PriorRGWrapper
    pr = PriorRGWrapper(cfg)
    out = []
    for entry in tqdm(items, desc="PriorRG"):
        cs = entry.get("current_study_manifest", {}) or {}
        study_id = cs.get("study_id")
        task_id = entry.get("task_id")
        for img in cs.get("images", []) or []:
            ip = get_image_path(cfg, img.get("path", ""))
            if not os.path.exists(ip):
                continue
            res = pr.predict(ip, img.get("view", "AP"))
            out.append({
                "image_id":        img.get("id"),
                "study_id":        study_id,
                "task_id":         task_id,
                "view":            img.get("view", "AP") or "AP",
                "report_text":     res["report_text"],
                "chexbert_labels": res["chexbert_labels"],
            })
    return out


def precompute_medgemma(items, cfg, image_root):
    """MedGemma generative-expert evidence (paper §3.3.1, Tab. 2).

    Paper main: MedGemma is asked to produce a free-form report for the
    image; that report is then labeled by CheXbert into 14-class buckets,
    which the consensus module consumes as the MedGemma expert token.
    """
    import base64
    from openai import OpenAI
    from src.models.chexbert_wrapper import get_chexbert_wrapper

    mg_cfg = cfg["medgemma"]
    client = OpenAI(api_key="EMPTY", base_url=mg_cfg["api_base"])
    chexbert = get_chexbert_wrapper(cfg)
    system_prompt = (
        "You are an expert radiologist. Given a chest X-ray, write a concise "
        "Findings paragraph in standard radiology report style. Use canonical "
        "disease names (cardiomegaly, pleural effusion, pneumothorax, "
        "atelectasis, consolidation, pneumonia, edema, lung opacity, etc.). "
        "Output only the Findings paragraph — no preamble."
    )
    out = []
    for entry in tqdm(items, desc="MedGemma"):
        for img in (entry.get("current_study_manifest", {}) or {}).get("images", []) or []:
            ip = get_image_path(cfg, img.get("path", ""))
            if not os.path.exists(ip):
                continue
            with open(ip, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Write the Findings paragraph."},
                ]},
            ]
            try:
                resp = client.chat.completions.create(
                    model=mg_cfg["model_name"],
                    messages=messages, max_tokens=512, temperature=0.0,
                    timeout=mg_cfg.get("timeout", 120.0),
                )
                report_text = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"[MedGemma WARN] image {img.get('id')}: {e}")
                report_text = ""
            chexbert_labels = (chexbert.label(report_text)
                               if report_text else {})
            out.append({
                "image_id":        img.get("id"),
                "report_text":     report_text,
                "chexbert_labels": chexbert_labels,
            })
    return out


def precompute_gt_labels(items, cfg, image_root):
    """Run CheXbert on each study's reference findings report and broadcast the
    resulting 14-class binary vector to every image in that study.

    This produces the per-image `gt_binary` field used by
    `training/dataset.py` to form the supervision target (paper Eq. 12).
    """
    from src.models.chexbert_wrapper import get_chexbert_wrapper, CHEXPERT_LABELS
    chexbert = get_chexbert_wrapper(cfg)
    out = []
    for entry in tqdm(items, desc="GT labels"):
        cs = entry.get("current_study_manifest", {}) or {}
        ref_report = cs.get("target_report") or ""
        if not ref_report.strip():
            gt_binary = {d: 0 for d in CHEXPERT_LABELS}
        else:
            labels = chexbert.label(ref_report)
            gt_binary = {d: int(labels.get(d) == "Positive") for d in CHEXPERT_LABELS}
        study_id = cs.get("study_id")
        task_id = entry.get("task_id")
        for img in cs.get("images", []) or []:
            out.append({
                "image_id":  img.get("id"),
                "study_id":  study_id,
                "task_id":   task_id,
                "gt_binary": gt_binary,
            })
    return out


def precompute_classifier(items, cfg, image_root, tag: str):
    if tag == "convnext":
        from src.models.convnext_wrapper import ConvNeXtClassifierWrapper
        clf = ConvNeXtClassifierWrapper(
            ckpt_path=cfg["experts"]["convnext"]["checkpoint"],
            image_size=cfg["experts"]["convnext"].get("image_size", 320),
        )
    elif tag == "rad_dino":
        from src.models.rad_dino_wrapper import RadDinoClassifierWrapper
        clf = RadDinoClassifierWrapper(
            encoder_path=cfg["experts"]["rad_dino"]["encoder_path"],
            head_ckpt_path=cfg["experts"]["rad_dino"]["head_ckpt"],
        )
    else:
        raise ValueError(f"Unknown classifier tag: {tag}")

    out = []
    for entry in tqdm(items, desc=tag):
        for img in (entry.get("current_study_manifest", {}) or {}).get("images", []) or []:
            ip = get_image_path(cfg, img.get("path", ""))
            if not os.path.exists(ip):
                continue
            res = clf.predict(ip)
            out.append({"image_id": img.get("id"), "probs": res["probs"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--experts", nargs="+",
                    default=["priorrg", "medgemma", "rad_dino", "convnext", "gt_labels"],
                    choices=["priorrg", "medgemma", "rad_dino", "convnext", "gt_labels"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    items = load_split(cfg, args.split)
    cache_dir = cfg["precomputed"]["dir"]
    image_root = cfg["dataset"]["image_root"]

    for tag in args.experts:
        out_path = os.path.join(cache_dir, f"{args.split}_{tag}.json")
        if tag == "priorrg":
            records = precompute_priorrg(items, cfg, image_root)
        elif tag == "medgemma":
            records = precompute_medgemma(items, cfg, image_root)
        elif tag == "gt_labels":
            records = precompute_gt_labels(items, cfg, image_root)
        else:
            records = precompute_classifier(items, cfg, image_root, tag)
        _save(records, out_path)


if __name__ == "__main__":
    main()

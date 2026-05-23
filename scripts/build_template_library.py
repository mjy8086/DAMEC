"""
Offline cluster template library (paper §3.5.1).

Encodes the *training* reports with BiomedBERT, runs K-means (default K=20),
and for each cluster stores:
  - `representative_reports`: M reports closest to the centroid (M = `--top_r`)
  - `template_report`        : the closest one (compatibility key)
  - `scf_profile`            : per-disease CheXbert prevalence over cluster members
  - `top_diseases`           : the 5 most prevalent diseases (for diagnostics)

The inference-time selector (`src/nodes/template_selector.py`) picks the closest
cluster by positive-weighted Euclidean distance against the predicted CF
probabilities (Eq. 11) and feeds the M representative reports to the writer.

Usage
-----
    python scripts/build_template_library.py \
        --train_json <path to train manifest> \
        --K 20 --top_r 9 \
        --out outputs/templates/templates_K20_R9.json
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch

# Make `src/` importable when running this script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.chexbert_wrapper import get_chexbert_wrapper   # noqa: E402
from src.utils.io import load_config                            # noqa: E402


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def load_train_reports(train_json: str) -> List[Tuple[str, str]]:
    with open(train_json) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    out = []
    for item in data:
        tid = item.get("task_id", "")
        rpt = (item.get("current_study_manifest", {}) or {}).get("target_report", "") or ""
        if rpt and tid:
            out.append((tid, rpt))
    return out


def encode_reports(reports: List[str], batch_size: int = 32, device: str = "cuda") -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel

    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        print(f"[encoder] Using {model_name}")
    except Exception as e:
        print(f"[encoder] Failed to load BiomedBERT ({e}); falling back to bert-base-uncased")
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(reports), batch_size):
            batch = reports[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256,
                            return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean_emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(mean_emb.cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"[encoder] {i}/{len(reports)}")
    return np.concatenate(embeddings, axis=0)


def kmeans(embeddings: np.ndarray, K: int, seed: int = 42):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


def find_top_R(reports, embeddings, labels, cluster_id, center, R: int):
    mask = labels == cluster_id
    cluster_emb = embeddings[mask]
    cluster_reports = [r for r, m in zip(reports, mask) if m]
    if not cluster_reports:
        return []
    dists = np.linalg.norm(cluster_emb - center, axis=1)
    order = np.argsort(dists)[:R]
    return [
        {
            "rank": rank + 1,
            "task_id": cluster_reports[int(idx)][0],
            "report":  cluster_reports[int(idx)][1],
            "distance": float(dists[int(idx)]),
        }
        for rank, idx in enumerate(order)
    ]


def scf_profile_for_cluster(cluster_members, chexbert) -> Tuple[List[float], int]:
    profile = np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
    n = 0
    for _tid, rpt in cluster_members:
        try:
            labs = chexbert.label(rpt)
            for di, d in enumerate(CHEXPERT_LABELS):
                if labs.get(d) == "Positive":
                    profile[di] += 1
            n += 1
        except Exception:
            continue
    if n > 0:
        profile = profile / n
    return profile.tolist(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True, help="Training manifest JSON.")
    ap.add_argument("--K", type=int, default=20, help="Number of clusters.")
    ap.add_argument("--top_r", type=int, default=9, help="Reports per cluster (M in paper §3.5.1).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default=None, help="Path to a config YAML (provides chexbert.checkpoint, src_path).")
    args = ap.parse_args()

    print(f"[build] Loading reports from {args.train_json}")
    reports = load_train_reports(args.train_json)
    print(f"[build] N reports: {len(reports)}")

    embeddings = encode_reports([r for _, r in reports], device=args.device)
    labels, centers = kmeans(embeddings, args.K, seed=42)

    cfg = load_config(args.config)
    chexbert = get_chexbert_wrapper(cfg)

    templates = {}
    for cid in range(args.K):
        cluster_members = [reports[i] for i in range(len(reports)) if labels[i] == cid]
        n_members = len(cluster_members)
        rep_reports = find_top_R(reports, embeddings, labels, cid, centers[cid], args.top_r)
        scf_profile, n_profiled = scf_profile_for_cluster(cluster_members, chexbert)

        sorted_idx = np.argsort(scf_profile)[::-1]
        top_diseases = [
            (CHEXPERT_LABELS[i], round(scf_profile[i], 3))
            for i in sorted_idx[:5] if scf_profile[i] > 0.05
        ]

        templates[str(cid)] = {
            "cluster_id": cid,
            "n_members": n_members,
            "n_profiled": n_profiled,
            "centroid_task_id": rep_reports[0]["task_id"] if rep_reports else None,
            "template_report":  rep_reports[0]["report"]  if rep_reports else "",
            "representative_reports": rep_reports,
            "scf_profile": scf_profile,
            "top_diseases": top_diseases,
        }
        print(f"[cluster {cid:2d}] n={n_members:>4}  R={len(rep_reports)}  top: {top_diseases}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "K": args.K, "R": args.top_r,
            "encoder": "BiomedNLP-BiomedBERT-base-uncased-abstract",
            "n_train_reports": len(reports),
            "templates": templates,
        }, f, indent=2)
    print(f"[build] Saved → {args.out}")


if __name__ == "__main__":
    main()

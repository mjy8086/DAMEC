"""
DAMEC — Pipeline runner: initializes wrappers, builds the graph, and runs it
in parallel over a dataset split.

`init_wrappers` loads the four frozen experts and the trained consensus
module exactly once and reuses them across all cases. Inference proceeds in
parallel via `ThreadPoolExecutor(max_workers=study_concurrency)`; the four
experts are wrapped to be thread-safe (CheXbert holds a module-level lock).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.utils.io import load_split, load_precomputed_evidence, save_json
from src.utils.logging_utils import setup_logging, log_case_result, logger
from src.models.consensus_wrapper import ConsensusModuleWrapper
from src.models.medgemma_wrapper import MedGemmaWrapper
from src.models.priorrg_wrapper import PriorRGWrapper
from src.llm.factory import get_llm
from src.graph import build_graph


def init_wrappers(cfg: Dict[str, Any], prompts: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Build every long-lived object the graph needs (experts + consensus)."""
    precomputed = (
        load_precomputed_evidence(cfg, split)
        if cfg.get("precomputed", {}).get("use_precomputed", True) else {}
    )

    wrappers: Dict[str, Any] = {}

    # ----- Trained consensus module (paper §3.3.3) -----
    cons_cfg = cfg["consensus"]
    wrappers["consensus"] = ConsensusModuleWrapper(
        ckpt_path=cons_cfg["checkpoint"],
        classifier_tags=list(cons_cfg["classifier_tags"]),
    )

    # ----- Discriminative experts: precomputed per-image probabilities -----
    cls_cfg = cfg.get("classifier_evidence", {})
    classifier_evidence: Dict[str, Dict[str, Any]] = {}
    for tag in cons_cfg["classifier_tags"]:
        per_split = cls_cfg.get(tag, {}).get(split)
        if per_split and os.path.exists(per_split):
            with open(per_split) as f:
                data = json.load(f)
            if isinstance(data, list):
                data = {item["image_id"]: item for item in data}
            classifier_evidence[tag] = data
            print(f"[runner] Loaded discriminative-expert cache: {tag} ({len(data)} records)")
        else:
            classifier_evidence[tag] = {}
            print(f"[runner] WARN: cache missing for discriminative expert '{tag}'")
    wrappers["classifier_evidence"] = classifier_evidence

    # ----- Generative experts -----
    wrappers["priorrg"] = PriorRGWrapper(cfg, precomputed=precomputed.get("priorrg", {}))
    wrappers["medgemma"] = MedGemmaWrapper(cfg, prompts, precomputed=precomputed.get("medgemma", {}))

    return wrappers


def build_initial_state(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": item.get("task_id", ""),
        "current_study": item.get("current_study_manifest", {}),
        "prior_studies": item.get("prior_studies_manifest", []),
        "evidence_bundle": None,
        "all_evidence": [],
        "scf_current": {"conditions": {}},
        "scf_prior":   {"conditions": {}},
        "scf_delta":   {"conditions": {}},
        "initial_entropy": 0.0,
        "selected_template": None,
        "selected_templates": [],
        "final_report": None,
        "validator_done": False,
        "validator_retries": 0,
        "validator_feedback": None,
        "validator_history": [],
    }


def extract_result(final_state: Dict[str, Any], ground_truth: str = "") -> Dict[str, Any]:
    scf_current = final_state.get("scf_current", {})
    conds = scf_current.get("conditions", {})
    avg_h = sum(c.get("H", 0.0) for c in conds.values()) / max(len(conds), 1)

    return {
        "case_id": final_state.get("case_id", ""),
        "final_report": final_state.get("final_report", ""),
        "ground_truth": ground_truth,
        "initial_entropy": round(final_state.get("initial_entropy", avg_h), 3),
        "final_entropy": round(avg_h, 3),
        "scf_current": scf_current,
        "scf_delta": final_state.get("scf_delta", {}),
        "scf_prior": final_state.get("scf_prior", {}),
        "validator_history": final_state.get("validator_history", []),
        "actions": [
            {
                "image_id": e.get("image_id"),
                "view": e.get("view"),
                "time_label": e.get("time_label", "t"),
            }
            for e in final_state.get("all_evidence", [])
        ],
    }


def run_pipeline(
    cfg: Dict[str, Any],
    prompts: Dict[str, Any],
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    setup_logging()

    data = load_split(cfg, split)
    if max_samples:
        data = data[:max_samples]
        logger.info(f"[runner] Using first {max_samples} samples (debug mode)")

    logger.info("[runner] Initializing model wrappers...")
    wrappers = init_wrappers(cfg, prompts, split)

    llm = get_llm(cfg, purpose="writer")
    app = build_graph(cfg, prompts, wrappers, llm)

    val_retries = cfg.get("validator", {}).get("max_retries", 2)
    recursion_limit = 10 + 2 * val_retries + 10

    output_dir = cfg.get("output", {}).get("dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_every = cfg.get("output", {}).get("save_every_n", 5)
    custom_name = cfg.get("output", {}).get("filename")
    if custom_name:
        b_name, ext = os.path.splitext(custom_name)
        final_save_name = custom_name
        partial_save_name = f"{b_name}_partial{ext}"
    else:
        final_save_name = f"results_{split}.json"
        partial_save_name = f"results_{split}_partial.json"

    study_concurrency = int(cfg.get("study_concurrency", 4))
    logger.info(f"[runner] study_concurrency={study_concurrency}")

    def process_one(idx_item):
        i, item = idx_item
        case_id = item.get("task_id", f"case_{i}")
        initial_state = build_initial_state(item)
        start = time.time()
        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
            gt = item.get("current_study_manifest", {}).get("target_report", "")
            result = extract_result(final_state, ground_truth=gt)
            result["elapsed_seconds"] = round(time.time() - start, 2)
            return (i, case_id, result, None)
        except Exception as e:
            return (i, case_id, {
                "case_id": case_id,
                "final_report": "",
                "ground_truth": item.get("current_study_manifest", {}).get("target_report", ""),
                "error": str(e),
                "elapsed_seconds": round(time.time() - start, 2),
            }, e)

    results: List[Optional[Dict[str, Any]]] = [None] * len(data)
    completed = [0]

    with ThreadPoolExecutor(max_workers=study_concurrency) as ex:
        futures = {ex.submit(process_one, (i, item)): i for i, item in enumerate(data)}
        with tqdm(total=len(data), desc=f"DAMEC ({study_concurrency} workers)") as pbar:
            for fut in as_completed(futures):
                i, case_id, result, err = fut.result()
                results[i] = result
                if err is not None:
                    logger.error(f"[runner] Case {case_id} FAILED: {err}")
                else:
                    log_case_result(case_id, result)
                completed[0] += 1
                pbar.update(1)
                if completed[0] % save_every == 0:
                    save_json([r for r in results if r is not None],
                              os.path.join(output_dir, partial_save_name))

    save_path = os.path.join(output_dir, final_save_name)
    save_json(results, save_path)
    logger.info(f"[runner] Done. {len(results)} cases written to {save_path}")
    return results

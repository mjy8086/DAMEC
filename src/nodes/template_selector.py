"""
DAMEC — Clinical-Context Retrieval node (paper §3.5.1).

Selects the closest cluster from the offline-built cluster template library by
positive-weighted Euclidean distance between the predicted CF probability
vector and each cluster's disease-prevalence centroid (paper Eq. 11):

    dist(p, μ_k) = sqrt( Σ_d ω_d · (p_d − μ_k,d)^2 )

with ω_d = 2 for diseases predicted as POS and ω_d = 1 otherwise. The
nearest cluster's M = top_k representative reports are returned and fed
to the writer in paper §3.5.1.
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


_TEMPLATE_LIBRARY: Optional[Dict[str, Any]] = None


def _load_template_library(path: str) -> Dict[str, Any]:
    global _TEMPLATE_LIBRARY
    if _TEMPLATE_LIBRARY is None:
        with open(path) as f:
            _TEMPLATE_LIBRARY = json.load(f)
        n = len(_TEMPLATE_LIBRARY.get("templates", {}))
        print(f"[template_lib] Loaded K={n} clusters from {path}")
    return _TEMPLATE_LIBRARY


def _scf_prob_vector(scf: Dict[str, Any]) -> np.ndarray:
    conds = scf.get("conditions", {})
    return np.array(
        [float(conds.get(d, {}).get("p", 0.5)) for d in CHEXPERT_LABELS],
        dtype=np.float32,
    )


def _distance(scf_p: np.ndarray, cluster_profile: np.ndarray, pos_weight: float = 2.0) -> float:
    """Positive-weighted Euclidean distance (paper Eq. 11)."""
    pos_mask = (scf_p >= 0.5).astype(np.float32)
    weights = 1.0 + (pos_weight - 1.0) * pos_mask
    diff = scf_p - cluster_profile
    return float(np.sqrt((weights * diff ** 2).sum()))


def template_selector_node(state: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    tcfg = cfg.get("template_library", {})
    path = tcfg.get("path")
    if not path or not os.path.exists(path):
        print(f"[template_selector] Template library not found: {path}. Skipping.")
        return {"selected_template": None, "selected_templates": []}

    lib = _load_template_library(path)
    templates = lib.get("templates", {})
    if not templates:
        return {"selected_template": None, "selected_templates": []}

    scf_p = _scf_prob_vector(state.get("scf_current", {}))
    pos_weight = float(tcfg.get("pos_weight", 2.0))

    scored = []
    for cid_str, t in templates.items():
        profile = np.array(t["scf_profile"], dtype=np.float32)
        d = _distance(scf_p, profile, pos_weight=pos_weight)
        scored.append({
            "cluster_id": int(t["cluster_id"]),
            "distance": d,
            "data": t,
        })
    scored.sort(key=lambda x: x["distance"])
    best = scored[0]
    best_data = best["data"]

    top_k = int(tcfg.get("top_k", 9))
    rep_reports = best_data.get("representative_reports") or [{
        "rank": 1,
        "task_id": best_data.get("centroid_task_id"),
        "report": best_data.get("template_report", ""),
        "distance": 0.0,
    }]
    rep_reports = rep_reports[:top_k]

    selected_templates = [
        {
            "cluster_id": best["cluster_id"],
            "template_report": rep["report"],
            "distance": rep["distance"],
            "top_diseases": best_data.get("top_diseases", []),
            "rank": rep["rank"],
            "task_id": rep.get("task_id"),
        }
        for rep in rep_reports
    ]

    scf_pos = sorted([CHEXPERT_LABELS[i] for i, p in enumerate(scf_p) if p >= 0.5])
    cluster_pos = [d for d, _ in best_data.get("top_diseases", [])[:3]]
    print(f"[template_selector] cluster={best['cluster_id']} d={best['distance']:.3f}  "
          f"CF_POS={scf_pos[:3]}{'..' if len(scf_pos) > 3 else ''}  "
          f"cluster_top={cluster_pos}  K={len(selected_templates)}")

    return {
        "selected_template": {
            "cluster_id": best["cluster_id"],
            "template_report": best_data.get("template_report", ""),
            "distance": best["distance"],
            "top_diseases": best_data.get("top_diseases", []),
            "rank": 1,
        },
        "selected_templates": selected_templates,
    }

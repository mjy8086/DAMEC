"""
DAMEC — Bootstrap-Prior node.

Extracts the prior CF descriptor from the latest prior study's report via
CheXbert, so that downstream code can compute the longitudinal change δ_d
(paper §3.4).

If no prior is available, emits an empty CF_prior; subsequent δ_d entries will
be labelled "no_prior" / "indeterminate".
"""

from typing import Any, Dict

from src.models.chexbert_wrapper import get_chexbert_wrapper


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def bootstrap_prior_node(state: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    prior_studies = state.get("prior_studies", [])

    empty = {
        "scf_prior": {"conditions": {}, "source": "none"},
        "all_evidence": [],
    }

    if not prior_studies:
        print("[bootstrap_prior] No prior studies found.")
        return empty

    report_text = (prior_studies[0].get("report") or "").strip()
    if not report_text:
        print("[bootstrap_prior] Prior report is empty.")
        return {"scf_prior": {"conditions": {}, "source": "empty_prior"}, "all_evidence": []}

    chexbert = get_chexbert_wrapper(cfg)
    labels = chexbert.label(report_text)

    conditions = {}
    for disease, label in labels.items():
        if label == "Positive":
            conditions[disease] = {"p": 0.85, "H": 0.20, "state": "POS", "conf": 0.80}
        elif label == "Negative":
            conditions[disease] = {"p": 0.15, "H": 0.20, "state": "NEG", "conf": 0.80}
        elif label == "Uncertain":
            conditions[disease] = {"p": 0.50, "H": 0.60, "state": "UNC", "conf": 0.40}
        else:
            conditions[disease] = {"p": 0.50, "H": 1.00, "state": "UNC", "conf": 0.00}

    n_pos = sum(1 for c in conditions.values() if c["state"] == "POS")
    print(f"[bootstrap_prior] Extracted {n_pos} POS findings from prior.")
    return {
        "scf_prior": {"conditions": conditions, "source": "chexbert_prior_report"},
        "all_evidence": [],
    }

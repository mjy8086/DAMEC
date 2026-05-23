"""
DAMEC — Clinical Findings (CF) descriptor helpers.

The CF descriptor (paper §3.4) is a dict
    {"conditions": {disease: {p, logit, H, state, provenance, attn, ...}, ...}}

This module provides:
  - probability/logit/entropy conversions
  - state assignment (POS / NEG / UNC) using symmetric thresholds (Eq. 10)
  - CF initialization from the consensus module's case-level output
  - longitudinal change δ_d (new / resolved / stable / worsened / improved)
  - text serialization for the LLM writer prompt
"""

import math
from typing import Any, Dict, List, Optional


LOGIT_EPS = 1e-6
EXPERT_TAGS = ("priorrg", "medgemma", "rad_dino", "convnext")


# --- logit / probability / entropy conversions ------------------------------

def prob_to_logit(prob: Optional[float]) -> float:
    if prob is None:
        return 0.0
    p = max(LOGIT_EPS, min(1 - LOGIT_EPS, float(prob)))
    return math.log(p / (1 - p))


def logit_to_prob(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def binary_entropy_norm(logit: float) -> float:
    """Normalized binary entropy: H_norm(σ(logit)) ∈ [0, 1]."""
    p = logit_to_prob(logit)
    eps = 1e-8
    p = max(eps, min(1 - eps, p))
    h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return h / math.log(2)


# --- state assignment (paper Eq. 10) ----------------------------------------

def assign_label(p: float, pos_thr: float = 0.55, neg_thr: float = 0.45) -> str:
    """state_d = POS if p ≥ τ_+ ; NEG if p ≤ τ_− ; else UNC.

    Default thresholds (0.55 / 0.45) form a symmetric narrow UNC band around 0.5.
    """
    if p >= pos_thr:
        return "POS"
    if p <= neg_thr:
        return "NEG"
    return "UNC"


# --- CF initialization from the consensus module ----------------------------

def init_scf_from_integrator(
    integrated: Dict[str, Any],
    chexpert_labels: List[str],
) -> Dict[str, Any]:
    """Build the CF dict from the consensus module's per-disease output."""
    conditions = {}
    for disease in chexpert_labels:
        logit = integrated["s_img"].get(disease, 0.0)
        p = integrated["p_img"].get(disease, 0.5)
        H = integrated["u_img"].get(disease, 1.0)

        attn = integrated.get("attn", {}).get(disease, {})
        provenance = [src for src in EXPERT_TAGS if attn.get(src, 0) > 0.01] or list(EXPERT_TAGS)

        conditions[disease] = {
            "p": round(p, 4),
            "logit": round(logit, 4),
            "H": round(H, 4),
            "conf": round(1.0 - H, 4),
            "state": assign_label(p),
            "provenance": provenance,
            "attn": attn,
        }
    return {"conditions": conditions, "source": "consensus_module"}


# --- longitudinal change δ_d (paper §3.4) -----------------------------------

def compute_delta(
    scf_current: Dict[str, Any],
    scf_prior: Dict[str, Any],
    chexpert_labels: List[str],
    margin: float = 0.15,
) -> Dict[str, Any]:
    """Compute per-disease longitudinal change from current and prior CF.

    Status values: new | resolved | stable | worsened | improved | indeterminate | no_prior
    """
    cur_conds = scf_current.get("conditions", {})
    pri_conds = scf_prior.get("conditions", {})

    delta = {}
    for disease in chexpert_labels:
        cur = cur_conds.get(disease, {})
        pri = pri_conds.get(disease, {})

        cur_state = cur.get("state", "UNC")
        pri_state = pri.get("state", "UNC")
        cur_p = cur.get("p", 0.5)
        pri_p = pri.get("p", 0.5)

        if not pri:
            delta[disease] = {"status": "no_prior", "cur_p": cur_p, "pri_p": None}
        elif cur_state == "POS" and pri_state in ("NEG", "UNC"):
            delta[disease] = {"status": "new", "cur_p": cur_p, "pri_p": pri_p}
        elif cur_state in ("NEG", "UNC") and pri_state == "POS":
            delta[disease] = {"status": "resolved", "cur_p": cur_p, "pri_p": pri_p}
        elif cur_state == "POS" and pri_state == "POS":
            diff = cur_p - pri_p
            if abs(diff) <= margin:
                delta[disease] = {"status": "stable", "cur_p": cur_p, "pri_p": pri_p}
            elif diff > margin:
                delta[disease] = {"status": "worsened", "cur_p": cur_p, "pri_p": pri_p}
            else:
                delta[disease] = {"status": "improved", "cur_p": cur_p, "pri_p": pri_p}
        else:
            delta[disease] = {"status": "indeterminate", "cur_p": cur_p, "pri_p": pri_p}

    return {"conditions": delta}


# --- text serialization for the writer prompt -------------------------------

def scf_to_text(scf: Dict[str, Any]) -> str:
    """Render the CF dict as three explicit POS / UNC / NEG sections so the
    writer cannot miss the required-mentions list."""
    conditions = scf.get("conditions", {})
    if not conditions:
        return "No structured findings available."

    pos_lines, unc_lines, neg_lines = [], [], []
    for disease, info in conditions.items():
        state = info.get("state", "UNC")
        p = info.get("p", 0.5)
        if state == "POS":
            conf_tag = "HIGH" if p >= 0.7 else "MODERATE"
            pos_lines.append(f"  - {disease} (p={p:.2f}, {conf_tag} confidence)")
        elif state == "UNC":
            unc_lines.append(f"  - {disease} (p={p:.2f})")
        else:
            neg_lines.append(f"  - {disease} (p={p:.2f})")

    sections = []
    if pos_lines:
        sections.append(
            "REQUIRED MENTIONS (CF POS — MUST appear in report by CheXpert canonical name):\n"
            + "\n".join(pos_lines)
        )
    else:
        sections.append("REQUIRED MENTIONS: (none — no POS findings)")
    if unc_lines:
        sections.append("UNCERTAIN (mention with hedging only if Base Draft does):\n" + "\n".join(unc_lines))
    if neg_lines:
        sections.append("NEGATIVE FINDINGS (do NOT assert; may mention as absent if natural):\n" + "\n".join(neg_lines))
    return "\n\n".join(sections)


def scf_attributes_to_text(scf: Dict[str, Any]) -> str:
    """Render α_d (severity / location / laterality) for POS diseases that have them."""
    conditions = scf.get("conditions", {})
    lines = []
    for disease, info in conditions.items():
        if info.get("state") != "POS":
            continue
        attrs = info.get("attrs")
        if not attrs:
            continue
        parts = []
        for key in ("severity", "location", "laterality"):
            val = attrs.get(key)
            if val in (None, "", "null"):
                continue
            parts.append(f"{key}={val}")
        if parts:
            lines.append(f"- {disease}: " + ", ".join(parts))
    return "\n".join(lines)


def delta_to_text(scf_delta: Dict[str, Any]) -> str:
    conditions = scf_delta.get("conditions", {})
    if not conditions:
        return "No prior comparison available."
    lines = []
    for disease, info in conditions.items():
        status = info.get("status", "indeterminate")
        if status == "no_prior":
            continue
        cur_p = info.get("cur_p", 0.5)
        pri_p = info.get("pri_p")
        if pri_p is not None:
            lines.append(f"- {disease}: {status} (current={cur_p:.2f}, prior={pri_p:.2f})")
        else:
            lines.append(f"- {disease}: {status} (current={cur_p:.2f})")
    return "\n".join(lines) if lines else "No significant changes from prior."

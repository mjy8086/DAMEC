"""
DAMEC — Clinical-Context Validation node (paper §3.5.2).

Runs CheXbert on the draft report, extracts its positive disease set P̂, and
compares it against the CF positive set P^CF. If P^CF ⊆ P̂ the draft is
accepted; otherwise the writer is re-invoked with an instruction listing the
missing diseases P^CF \\ P̂.
"""

from typing import Any, Dict, List

from src.models.chexbert_wrapper import get_chexbert_wrapper


def _scf_pos_set(scf: Dict[str, Any]) -> List[str]:
    return [d for d, info in scf.get("conditions", {}).items()
            if isinstance(info, dict) and info.get("state") == "POS"]


def report_validator_node(state: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    report = state.get("final_report", "") or ""
    if not report:
        print("[validator] No report to validate.")
        return {"validator_done": True}

    scf_pos = set(_scf_pos_set(state.get("scf_current", {})))

    chexbert = get_chexbert_wrapper(cfg)
    labels = chexbert.label(report)
    chex_pos = {d for d, lab in labels.items() if lab == "Positive"}

    missing_pos = sorted(scf_pos - chex_pos)
    extra_pos = sorted(chex_pos - scf_pos)

    retry_count = int(state.get("validator_retries", 0))
    max_retries = int(cfg.get("validator", {}).get("max_retries", 2))

    print(f"[validator] retry={retry_count}/{max_retries}  "
          f"CF_POS={len(scf_pos)}  CheX_POS={len(chex_pos)}  "
          f"missing={len(missing_pos)}  extra={len(extra_pos)}")

    history = list(state.get("validator_history", []))

    if not missing_pos:
        print("[validator] All CF POS mentioned ✓ — done.")
        return {
            "validator_done": True,
            "validator_history": history + [{"retry": retry_count, "missing": [], "extra": extra_pos}],
        }

    if retry_count >= max_retries:
        print(f"[validator] Max retries reached. Missing diseases NOT added: {missing_pos}")
        return {
            "validator_done": True,
            "validator_history": history + [{
                "retry": retry_count, "missing": missing_pos, "extra": extra_pos, "stopped": "max_retries",
            }],
        }

    print(f"[validator] Triggering Writer retry. missing={missing_pos}")
    return {
        "validator_done": False,
        "validator_feedback": {
            "missing_diseases": missing_pos,
            "extra_diseases": extra_pos,
            "previous_report": report,
        },
        "validator_retries": retry_count + 1,
        "validator_history": history + [{"retry": retry_count, "missing": missing_pos, "extra": extra_pos}],
    }

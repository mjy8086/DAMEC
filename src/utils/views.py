"""
DAMEC — View utilities.

Normalizes view labels (PA / AP / LATERAL), produces one-hot encodings used by
the consensus module's token input (paper §3.3.3 Eq. 5), and provides the
PA > AP > LATERAL ordering used by the Attribute-Finding Module.
"""

from typing import Any, Dict, List, Optional


VIEW_TYPES = ["PA", "AP", "LATERAL"]
VIEW_DIM = len(VIEW_TYPES)


def normalize_view(view: Optional[str]) -> str:
    """Normalize view names to PA / AP / LATERAL. Unknown views fall back to AP."""
    if view is None:
        return "AP"
    v = view.upper().strip()
    if v == "PA":
        return "PA"
    if v in ("LATERAL", "LL", "LAT"):
        return "LATERAL"
    return "AP"


def view_to_onehot(view: str) -> List[float]:
    view_norm = normalize_view(view)
    idx = VIEW_TYPES.index(view_norm) if view_norm in VIEW_TYPES else 1
    onehot = [0.0] * VIEW_DIM
    onehot[idx] = 1.0
    return onehot


def sort_by_view_priority(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort images by PA > AP > LATERAL priority (used by the multi-image attribute elicitor)."""
    priority = {"PA": 0, "AP": 1, "LATERAL": 2}
    return sorted(images, key=lambda im: priority.get(normalize_view(im.get("view")), 99))


def select_representative_image(images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the single highest-priority image (PA > AP > LATERAL); None if empty."""
    if not images:
        return None
    return sort_by_view_priority(images)[0]

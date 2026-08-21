from typing import Any, Dict

from src.utils.io import get_image_path
from src.utils.views import sort_by_view_priority


ATTR_ELIGIBLE = {
    "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture",
    "Enlarged Cardiomediastinum",
}


def attribute_elicitor_node(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    wrappers: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Dict[str, Any]:
    scf_curr = state.get("scf_current", {})
    conditions = scf_curr.get("conditions", {})

    pos_eligible = [
        d for d, e in conditions.items()
        if isinstance(e, dict) and e.get("state") == "POS" and d in ATTR_ELIGIBLE
    ]
    if not pos_eligible:
        print("[attr_elicitor] No eligible POS disease — skipping.")
        return {}

    images = state.get("current_study", {}).get("images", [])
    if not images:
        print("[attr_elicitor] No images — skipping.")
        return {}

    img_paths = [
        get_image_path(cfg, img.get("path", ""))
        for img in sort_by_view_priority(images)
        if img.get("path")
    ]
    if not img_paths:
        return {}

    attr_prompts = prompts.get("medgemma_attrs", {})
    sys_p = attr_prompts.get("system", "You are an expert radiologist.")
    user_template = attr_prompts.get("user", "")
    disease_list_txt = "\n".join(f"- {d}" for d in pos_eligible)
    try:
        user_p = user_template.format(disease_list=disease_list_txt)
    except (KeyError, IndexError) as e:
        print(f"[attr_elicitor] prompt format failed: {e}. Skipping.")
        return {}

    medgemma = wrappers.get("medgemma")
    if medgemma is None or not hasattr(medgemma, "elicit_attributes"):
        print("[attr_elicitor] MedGemma wrapper unavailable — skipping.")
        return {}

    try:
        result = medgemma.elicit_attributes(img_paths, sys_p, user_p)
    except Exception as e:
        print(f"[attr_elicitor] MedGemma call raised: {e}. Skipping.")
        return {}

    attrs = result.get("attributes", {})
    if not attrs:
        print(f"[attr_elicitor] MedGemma returned no attributes (parse_error={result.get('parse_error')}).")
        return {}

    new_conditions = dict(conditions)
    enriched = 0
    for disease in pos_eligible:
        fetched = attrs.get(disease, {})
        if not isinstance(fetched, dict):
            continue
        cleaned = {
            k: v for k, v in fetched.items()
            if v not in (None, "", "null") and k in ("severity", "location", "laterality")
        }
        loc = cleaned.get("location")
        lat = cleaned.get("laterality")
        if isinstance(loc, str) and isinstance(lat, str) and lat.lower() in loc.lower():
            cleaned.pop("laterality", None)
        if not cleaned:
            continue
        entry = dict(new_conditions.get(disease, {}))
        entry["attrs"] = cleaned
        new_conditions[disease] = entry
        enriched += 1

    if enriched == 0:
        print(f"[attr_elicitor] 0/{len(pos_eligible)} POS diseases enriched.")
        return {}

    print(f"[attr_elicitor] Enriched {enriched}/{len(pos_eligible)} POS diseases with attrs.")
    return {"scf_current": {**scf_curr, "conditions": new_conditions}}

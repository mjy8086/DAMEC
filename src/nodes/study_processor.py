from typing import Any, Dict, List

from src.utils.scf import init_scf_from_integrator, compute_delta


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def _build_evidence_bundle(image: Dict[str, Any], cfg: Dict[str, Any], wrappers: Dict[str, Any]) -> Dict[str, Any]:
    """Collect every expert's output for one image into a single dict."""
    image_id = image.get("id", "")
    view = image.get("view", "AP")
    rel_path = image.get("path", "")
    use_precomputed = cfg.get("precomputed", {}).get("use_precomputed", True)

    # PriorRG — generative · task-specific
    priorrg = wrappers["priorrg"]
    rrg_data = priorrg.get_precomputed(image_id) if use_precomputed else None
    rrg_result = (
        {"report_text": rrg_data.get("report_text", ""),
         "chexbert_labels": rrg_data.get("chexbert_labels", {})}
        if rrg_data else {"report_text": "", "chexbert_labels": {}}
    )

    # MedGemma — generative · foundation
    mg = wrappers["medgemma"]
    mg_data = mg.get_precomputed(image_id) if use_precomputed else None
    mg_result = (
        {"probs_3class":    mg_data.get("probs_3class", {}),
         "attributes":      mg_data.get("attributes", {}),
         "chexbert_labels": mg_data.get("chexbert_labels", {}),
         "report_text":     mg_data.get("report_text", "")}
        if mg_data else {"probs_3class": {}, "attributes": {},
                         "chexbert_labels": {}, "report_text": ""}
    )

    bundle = {
        "image_id": image_id,
        "view": view,
        "path": rel_path,
        "priorrg": rrg_result,
        "medgemma": mg_result,
    }
    # Discriminative experts (RAD-DINO, ConvNeXt) come in via precomputed evidence maps.
    classifier_evidence = wrappers.get("classifier_evidence", {})
    for tag, evidence_map in classifier_evidence.items():
        ext = evidence_map.get(image_id, {}) or {}
        bundle[tag] = {"probs": ext.get("probs", {})}
    return bundle


def study_processor_node(
    state: Dict[str, Any],
    cfg: Dict[str, Any],
    wrappers: Dict[str, Any],
) -> Dict[str, Any]:
    """Aggregate per-image expert outputs into a single case-level CF descriptor."""
    current_study = state.get("current_study", {}) or {}
    images = current_study.get("images", []) or []
    if not images:
        print("[study_processor] No images in current study — emitting empty CF.")
        return {
            "scf_current": {"conditions": {}, "source": "empty"},
            "scf_delta": {"conditions": {}},
            "all_evidence": [],
        }

    bundles: List[Dict[str, Any]] = [_build_evidence_bundle(img, cfg, wrappers) for img in images]
    print(f"[study_processor] Collected evidence for {len(bundles)} images "
          f"(views={[b.get('view') for b in bundles]})")

    consensus = wrappers["consensus"]
    integrated = consensus.predict_study(bundles)

    scf_current = init_scf_from_integrator(integrated, CHEXPERT_LABELS)
    n_pos = sum(1 for c in scf_current["conditions"].values() if c.get("state") == "POS")
    n_unc = sum(1 for c in scf_current["conditions"].values() if c.get("state") == "UNC")
    print(f"[study_processor] CF initialized: {n_pos} POS / {n_unc} UNC diseases.")

    delta_margin = cfg.get("scf", {}).get("delta_margin", 0.15)
    scf_delta = compute_delta(scf_current, state.get("scf_prior", {"conditions": {}}),
                              CHEXPERT_LABELS, margin=delta_margin)

    return {
        "scf_current": scf_current,
        "scf_delta": scf_delta,
        "all_evidence": bundles,
        "evidence_bundle": bundles[0],
        "initial_entropy": sum(c.get("H", 1.0) for c in scf_current["conditions"].values())
                            / max(len(scf_current["conditions"]), 1),
    }

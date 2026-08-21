from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from src.models.consensus_module import ConsensusModule


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
CHEXBERT_BUCKET = {"Blank": 0, "Positive": 1, "Negative": 2, "Uncertain": 3}
VIEW_TO_INDEX = {"PA": 0, "AP": 1, "LATERAL": 2}
LOGIT_EPS = 1.0e-6


def _prob_to_logit(p, eps: float = LOGIT_EPS) -> float:
    if p is None:
        return 0.0
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))


class ConsensusModuleWrapper:
    """Loads a trained consensus checkpoint and runs inference."""

    def __init__(self, ckpt_path: str, classifier_tags: List[str], device: str = "cuda"):
        self.classifier_tags = list(classifier_tags)
        self.K = len(self.classifier_tags)
        self.D = len(CHEXPERT_LABELS)
        self.device = device if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})

        self.model = ConsensusModule(
            num_classifiers=self.K,
            num_diseases=14,
            view_dim=3,
            hidden_dim=cfg.get("hidden_dim", 64),
            disease_embed_dim=cfg.get("disease_embed_dim", 16),
            tool_embed_dim=cfg.get("tool_embed_dim", 8),
            n_sab_blocks=cfg.get("n_sab_blocks", 2),
            n_heads=cfg.get("n_heads", 4),
            attn_dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[ConsensusModuleWrapper] loaded {ckpt_path} (K={self.K})")

    # ---- per-image tensor builder ----

    def _build_per_image(self, bundle: Dict[str, Any]):
        D, K = self.D, self.K
        # PriorRG → CheXbert bucket one-hot
        chex = torch.zeros(D, 4)
        labels = (bundle.get("priorrg", {}) or {}).get("chexbert_labels", {}) or {}
        for di, d in enumerate(CHEXPERT_LABELS):
            chex[di, CHEXBERT_BUCKET.get(labels.get(d, "Blank"), 0)] = 1.0
        # MedGemma → CheXbert bucket one-hot (same schema)
        mg = torch.zeros(D, 4)
        mg_chex = (bundle.get("medgemma", {}) or {}).get("chexbert_labels", {}) or {}
        for di, d in enumerate(CHEXPERT_LABELS):
            mg[di, CHEXBERT_BUCKET.get(mg_chex.get(d, "Blank"), 0)] = 1.0
        # View one-hot
        v = torch.zeros(3)
        v[VIEW_TO_INDEX.get((bundle.get("view") or "AP").upper(), 1)] = 1.0
        # Discriminative experts: per-(K, D) calibrated logits
        sx = torch.zeros(K, D)
        for ki, tag in enumerate(self.classifier_tags):
            ext = bundle.get(tag, {}) or {}
            probs = ext.get("probs", {}) or {}
            for di, d in enumerate(CHEXPERT_LABELS):
                p = probs.get(d)
                if p is not None:
                    sx[ki, di] = _prob_to_logit(p)
        return chex, mg, v, sx

    # ---- study-level forward ----

    @torch.no_grad()
    def predict_study(self, evidence_bundles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not evidence_bundles:
            return {
                "s_img": {d: 0.0 for d in CHEXPERT_LABELS},
                "p_img": {d: 0.5 for d in CHEXPERT_LABELS},
                "u_img": {d: 1.0 for d in CHEXPERT_LABELS},
                "attn":  {d: {} for d in CHEXPERT_LABELS},
                "view":  [],
            }

        per = [self._build_per_image(b) for b in evidence_bundles]
        N = len(per)
        chex = torch.stack([p[0] for p in per]).unsqueeze(0).to(self.device)
        mg   = torch.stack([p[1] for p in per]).unsqueeze(0).to(self.device)
        view = torch.stack([p[2] for p in per]).unsqueeze(0).to(self.device)
        sx   = torch.stack([p[3] for p in per]).unsqueeze(0).to(self.device)
        image_mask = torch.ones(1, N, device=self.device)

        out = self.model(chex, mg, view, sx, image_mask)
        s_img = out["s_img_study"].squeeze(0).cpu().tolist()
        p_img = out["p_img_study"].squeeze(0).cpu().tolist()

        eps = 1e-8

        def h_norm(p):
            p = max(eps, min(1 - eps, p))
            h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            return h / math.log(2)

        return {
            "s_img": {d: float(s_img[i]) for i, d in enumerate(CHEXPERT_LABELS)},
            "p_img": {d: float(p_img[i]) for i, d in enumerate(CHEXPERT_LABELS)},
            "u_img": {d: h_norm(float(p_img[i])) for i, d in enumerate(CHEXPERT_LABELS)},
            "attn":  {d: {} for d in CHEXPERT_LABELS},
            "view":  [b.get("view", "AP") for b in evidence_bundles],
        }

"""
DAMEC — ConvNeXt classifier wrapper (paper Tab. 2 — discriminative · task-specific).

A CheXpert-supervised ConvNeXt-Base classifier (trained at 320×320 with MixUp)
that outputs a per-disease probability vector over the 14 CheXbert categories.

In the reference pipeline this wrapper is used only for the offline
`scripts/precompute_experts.py` pass. At inference time, the consensus module
consumes the per-image cached JSON; no live wrapper is required.

The implementation expects a state dict over `torchvision`'s `convnext_base`
backbone followed by a linear head with `num_diseases = 14` outputs.
Override `_build_model` to plug in a different classifier checkpoint format.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


class ConvNeXtClassifierWrapper:
    """Loads a ConvNeXt-Base CheXpert classifier and produces per-disease probabilities."""

    def __init__(
        self,
        ckpt_path: str,
        image_size: int = 320,
        num_diseases: int = 14,
        device: str = "cuda",
    ):
        self.ckpt_path = ckpt_path
        self.image_size = image_size
        self.num_diseases = num_diseases
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: Optional[nn.Module] = None
        self.transform = None

    def _build_model(self):
        from torchvision.models import convnext_base
        from torchvision import transforms

        backbone = convnext_base(weights=None)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, self.num_diseases)
        state = torch.load(self.ckpt_path, map_location="cpu")
        state = state.get("model_state_dict", state)
        backbone.load_state_dict(state, strict=False)
        backbone = backbone.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = backbone
        print(f"[ConvNeXt] loaded {self.ckpt_path} on {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Any]:
        if self.model is None:
            self._build_model()
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x).squeeze(0).cpu()
        probs = torch.sigmoid(logits).tolist()
        return {
            "probs": {d: float(probs[i]) for i, d in enumerate(CHEXPERT_LABELS)},
        }

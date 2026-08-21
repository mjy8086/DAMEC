import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from PIL import Image


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


class RadDinoClassifierWrapper:
    def __init__(
        self,
        encoder_path: str,
        head_ckpt_path: str,
        feature_dim: int = 768,
        num_diseases: int = 14,
        device: str = "cuda",
    ):
        """
        Args:
            encoder_path:    local snapshot of microsoft/rad-dino (or HF model id).
            head_ckpt_path:  state dict of the linear disease head trained on the
                              target dataset's training split.
        """
        self.encoder_path = encoder_path
        self.head_ckpt_path = head_ckpt_path
        self.feature_dim = feature_dim
        self.num_diseases = num_diseases
        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder = None
        self.head = None
        self.processor = None

    def _build(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.encoder_path)
        self.encoder = AutoModel.from_pretrained(self.encoder_path).to(self.device).eval()
        head = nn.Linear(self.feature_dim, self.num_diseases)
        sd = torch.load(self.head_ckpt_path, map_location="cpu")
        head.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        self.head = head.to(self.device).eval()
        print(f"[RAD-DINO] encoder + head loaded on {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict[str, Any]:
        if self.encoder is None:
            self._build()
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        cls = self.encoder(**inputs).last_hidden_state[:, 0]   # CLS token
        logits = self.head(cls).squeeze(0).cpu()
        probs = torch.sigmoid(logits).tolist()
        return {
            "probs": {d: float(probs[i]) for i, d in enumerate(CHEXPERT_LABELS)},
        }

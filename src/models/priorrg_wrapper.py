import os
import sys
from typing import Any, Dict, Optional


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


class PriorRGWrapper:
    """Wraps PriorRG inference + CheXbert labeling; supports precomputed caches."""

    def __init__(self, cfg: Dict[str, Any], precomputed: Dict[str, Dict] = None):
        self.cfg = cfg
        self.precomputed = precomputed or {}
        self._model_bundle = None

    def get_precomputed(self, image_id: str) -> Optional[Dict]:
        return self.precomputed.get(image_id)

    def _load_model(self):
        if self._model_bundle is not None:
            return

        import torch
        from transformers import GPT2TokenizerFast

        priorrg_cfg = self.cfg["priorrg"]
        code_dir = priorrg_cfg["code_dir"]
        if not os.path.exists(code_dir):
            raise FileNotFoundError(
                f"PriorRG code not found at {code_dir}. See INSTALL.md §2 for download instructions."
            )

        sys.path.insert(0, code_dir)
        from models.model_github import TrainLanguageModelOneSample

        args = {
            "ckpt_zoo_dir": priorrg_cfg.get("ckpt_zoo_dir", ""),
            "view_position_dict": priorrg_cfg["view_dict"],
            "ann_path": priorrg_cfg["annotation"],
            "max_length": 100,
            "encoder_max_length": 300,
            "num_beams": 3,
            "hidden_size": 768,
            "text_encoder_num_blocks": 6,
            "temporal_fusion_num_blocks": 3,
            "perceiver_num_blocks": 3,
            "num_heads": 8,
            "num_latents": 128,
            "rad_dino_path": priorrg_cfg["rad_dino_path"],
            "cxr_bert_path": priorrg_cfg["cxr_bert_path"],
            "distilgpt2_path": priorrg_cfg["distilgpt2_path"],
            "chexbert_path": self.cfg["chexbert"]["checkpoint"],
            "bert_path": self.cfg["chexbert"]["bert_path"],
        }

        tokenizer = GPT2TokenizerFast.from_pretrained(args["distilgpt2_path"])
        tokenizer.add_special_tokens({"pad_token": "[PAD]", "sep_token": "[SEP]", "cls_token": "[CLS]"})
        tokenizer.add_tokens(["[INDICATION]", "[HISTORY]", "[Similar Cases]", "[FINDINGS]"])

        model = TrainLanguageModelOneSample(args, tokenizer)
        ckpt_path = priorrg_cfg["weight"]
        cur_state = model.state_dict()
        pre_state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        valid_state = {k: v for k, v in pre_state.items() if k in cur_state and v.shape == cur_state[k].shape}
        cur_state.update(valid_state)
        model.load_state_dict(cur_state)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        print(f"[PriorRG] Model loaded on {device}.")
        self._model_bundle = {"model": model, "device": device}

    def predict(self, image_path: str, view: str) -> Dict[str, Any]:
        """Generate one base-draft report and its CheXbert labels for a single image."""
        import re
        import torch
        from PIL import Image

        self._load_model()
        model = self._model_bundle["model"]
        device = self._model_bundle["device"]

        try:
            view_pos = view if view in ("PA", "AP", "LATERAL", "LL") else "unk"
            image_processor = model.image_processor

            img = Image.open(image_path).convert("RGB")
            cur_images = image_processor(img, return_tensors="pt").pixel_values.to(device)

            item = {
                "image_ids": [os.path.basename(image_path)],
                "current_study": {"image": cur_images, "view_position": [view_pos]},
                "clinical_context": [" "],
                "prior_study": None,
            }

            with torch.no_grad():
                generated_reports = model(item)

            report = (generated_reports[0] if generated_reports else "").strip()
            report = re.sub(r"[^\x20-\x7E]", "", report)

            from src.models.chexbert_wrapper import get_chexbert_wrapper
            chexbert = get_chexbert_wrapper(self.cfg)
            chexbert_labels = chexbert.label(report) if report else {c: "Blank" for c in CHEXPERT_LABELS}
            return {"report_text": report, "chexbert_labels": chexbert_labels}

        except Exception as e:
            print(f"[PriorRG ERROR] inference failed: {e}")
            return {
                "report_text": "",
                "chexbert_labels": {c: "Blank" for c in CHEXPERT_LABELS},
            }

import logging
import os
import sys
import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_GLOBAL_LOCK = threading.Lock()
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn as nn


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
CHEXBERT_ONEHOT_ORDER = ["Positive", "Negative", "Uncertain", "Blank"]
LABEL_TO_IDX = {v: i for i, v in enumerate(CHEXBERT_ONEHOT_ORDER)}


class CheXbertWrapper:
    """Loads CheXbert (Smit et al., 2020) and exposes a simple `label(report) → dict` API."""

    def __init__(self, checkpoint_path: str, src_path: str):
        self.checkpoint_path = checkpoint_path
        self.src_path = src_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return
        with _GLOBAL_LOCK:
            if self._initialized:
                return

            if self.src_path not in sys.path:
                sys.path.insert(0, self.src_path)

            try:
                warnings.filterwarnings("ignore")
                logging.getLogger("transformers").setLevel(logging.CRITICAL)

                from transformers import BertTokenizer
                from models.bert_labeler import bert_labeler

                self.model = bert_labeler()
                if torch.cuda.is_available():
                    self.model = nn.DataParallel(self.model).to(self.device)
                    ckpt = torch.load(self.checkpoint_path)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    ckpt = torch.load(self.checkpoint_path, map_location="cpu")
                    new_state = OrderedDict()
                    for k, v in ckpt["model_state_dict"].items():
                        new_state[k[7:] if k.startswith("module.") else k] = v
                    self.model.load_state_dict(new_state)

                self.model.eval()
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self._initialized = True
                print("[CheXbert] Model loaded successfully.")
            except Exception as e:
                print(f"[CheXbert ERROR] {e}")
                self._initialized = False

    def label(self, report_text: str) -> Dict[str, str]:
        self._lazy_init()
        if not self._initialized:
            return {c: "Blank" for c in CHEXPERT_LABELS}

        with _GLOBAL_LOCK:
            try:
                enc = self.tokenizer(
                    [report_text], padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attn_mask = enc["attention_mask"].to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, attn_mask)
                return {cond: CLASS_MAPPING[outputs[i].argmax(dim=1).item()]
                        for i, cond in enumerate(CHEXPERT_LABELS)}
            except Exception as e:
                print(f"[CheXbert ERROR] label() failed: {e}")
                return {c: "Blank" for c in CHEXPERT_LABELS}

    @staticmethod
    def label_to_onehot(label: str) -> List[float]:
        onehot = [0.0] * 4
        onehot[LABEL_TO_IDX.get(label, LABEL_TO_IDX["Blank"])] = 1.0
        return onehot


_instance: Optional[CheXbertWrapper] = None


def get_chexbert_wrapper(cfg: Dict[str, Any]) -> CheXbertWrapper:
    global _instance
    with _GLOBAL_LOCK:
        if _instance is None:
            _instance = CheXbertWrapper(
                checkpoint_path=cfg["chexbert"]["checkpoint"],
                src_path=cfg["chexbert"]["src_path"],
            )
        return _instance

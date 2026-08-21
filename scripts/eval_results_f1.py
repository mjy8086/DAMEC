import argparse
import json
import os
import sys
from typing import List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.io import load_config
from src.models.chexbert_wrapper import get_chexbert_wrapper, CHEXPERT_LABELS


def _chexbert_pos_vector(report: str, chexbert) -> np.ndarray:
    if not report or not report.strip():
        return np.zeros(len(CHEXPERT_LABELS), dtype=np.int32)
    labels = chexbert.label(report)
    return np.array(
        [1 if labels.get(d) == "Positive" else 0 for d in CHEXPERT_LABELS],
        dtype=np.int32,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="DAMEC results JSON (output of run.py).")
    ap.add_argument("--config",  default=None, help="Config providing chexbert.checkpoint / src_path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    chexbert = get_chexbert_wrapper(cfg)

    with open(args.results) as f:
        results = json.load(f)
    print(f"[eval] loaded {len(results)} cases from {args.results}")

    pred_vecs: List[np.ndarray] = []
    gt_vecs:   List[np.ndarray] = []
    skipped = 0

    for case in tqdm(results, desc="CheXbert labeling"):
        if case.get("error"):
            skipped += 1
            continue
        pred_report = case.get("final_report") or ""
        gt_report   = case.get("ground_truth") or ""
        if not pred_report.strip() or not gt_report.strip():
            skipped += 1
            continue
        pred_vecs.append(_chexbert_pos_vector(pred_report, chexbert))
        gt_vecs.append(_chexbert_pos_vector(gt_report,   chexbert))

    if not pred_vecs:
        print("[eval] no usable cases. Aborting.")
        return

    P = np.stack(pred_vecs, 0)
    G = np.stack(gt_vecs,   0)

    print(f"[eval] N evaluated = {len(P)} (skipped {skipped})")
    print()

    micro = precision_recall_fscore_support(G, P, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(G, P, average="macro", zero_division=0)

    def _fmt(p, r, f1):
        return f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}"

    print("Micro-averaged (MIMIC-CXR / MIMIC-ABN / Two-view CXR convention)")
    print("    " + _fmt(*micro[:3]))
    print()
    print("Macro-averaged (CheXpert Plus convention)")
    print("    " + _fmt(*macro[:3]))


if __name__ == "__main__":
    main()

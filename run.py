"""
DAMEC — entry point for full-pipeline inference.

Examples
--------
    # Smoke test on a single sample
    python run.py --split test --max_samples 1

    # Full test run with a specific config + seed
    python run.py --split test --config configs/local.yaml --seed 43
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.io import load_config, load_prompts
from src.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="DAMEC inference pipeline")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None, help="Limit to first N samples")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--prompts", type=str, default=None, help="Path to prompts YAML")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    prompts = load_prompts(args.prompts)

    results = run_pipeline(cfg=cfg, prompts=prompts, split=args.split, max_samples=args.max_samples)

    success = sum(1 for r in results if r.get("final_report"))
    errors = sum(1 for r in results if r.get("error"))
    print(f"\nCompleted: {len(results)} cases  ({success} success, {errors} errors)")


if __name__ == "__main__":
    main()

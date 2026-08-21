import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Make `src/` importable: the repo root is the parent of `training/`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.models.consensus_module import ConsensusModule
from training.config import CHEXPERT_LABELS, NUM_DISEASES
from training.dataset import StudyDataset, study_collate, StratifiedBatchSampler
from training.consensus_base import warm_start_consensus


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset: StudyDataset, num_diseases: int, power: float = 0.6) -> torch.Tensor:
    pos = torch.zeros(num_diseases)
    neg = torch.zeros(num_diseases)
    for rec in dataset.studies:
        gt = torch.tensor(rec["gt_binary"], dtype=torch.float32)
        pos += gt
        neg += 1 - gt
    pw = ((neg / pos.clamp(min=1)) ** power).clamp(min=0.1, max=20.0)
    return pw


def stratified_f1(model, loader, device):
    """Per-stratum (N=1 / N=2 / N≥3 / all) micro F1."""
    model.eval()
    bins = {"N=1": [], "N=2": [], "N>=3": [], "all": []}
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(
                chexbert_onehot=inputs["chexbert_onehot"],
                medgemma_onehot=inputs["medgemma_onehot"],
                view_onehot=inputs["view_onehot"],
                s_x_stack=inputs["s_x_stack"],
                image_mask=inputs["image_mask"],
            )
            preds = (out["s_img_study"] > 0.0).float().cpu()
            tgts = inputs["gt_binary"].cpu()
            ns = batch["n_images"]
            for bi in range(preds.shape[0]):
                n = int(ns[bi])
                key = "N=1" if n == 1 else "N=2" if n == 2 else "N>=3"
                bins[key].append((tgts[bi], preds[bi]))
                bins["all"].append((tgts[bi], preds[bi]))
    out_metrics = {}
    for stratum, pairs in bins.items():
        if not pairs:
            continue
        tp = fp = fn = 0
        for t, p in pairs:
            tp += int(((t == 1) & (p == 1)).sum())
            fp += int(((t == 0) & (p == 1)).sum())
            fn += int(((t == 1) & (p == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        out_metrics[stratum] = {
            "n": len(pairs), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }
    return out_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/consensus_default.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(cfg["output_dir"], f"{cfg.get('variant_tag', 'main')}_seed{cfg.get('seed', 42)}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[train] device={device}  out_dir={out_dir}")

    # ---- datasets ----
    train_ds = StudyDataset(
        precompute_dir=cfg["precompute_dir"],
        split=cfg["train_split"],
        classifier_tags=cfg["classifier_tags"],
        study_subset_path=cfg.get("study_subset"),
        min_images=cfg.get("min_images", 1),
        max_images=cfg.get("max_images", 6),
        evidence_dropout=cfg.get("evidence_dropout", 0.0),
        subset_size_mode=cfg.get("subset_size_mode", "all"),
        subset_size_ratio=cfg.get("subset_size_ratio"),
    )
    val_ds = StudyDataset(
        precompute_dir=cfg["precompute_dir"],
        split=cfg["val_split"],
        classifier_tags=cfg["classifier_tags"],
        study_subset_path=None,
        min_images=cfg.get("min_images", 1),
        max_images=cfg.get("max_images", 6),
        evidence_dropout=0.0,
        subset_size_mode="all",     # never subsample at eval
    )

    sampler = StratifiedBatchSampler(
        train_ds, batch_size=cfg["batch_size"], ratio=cfg["stratified_ratio"], seed=cfg["seed"],
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=study_collate,
                              num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            collate_fn=study_collate, num_workers=cfg.get("num_workers", 4))

    # ---- model ----
    model = ConsensusModule(
        num_classifiers=len(cfg["classifier_tags"]),
        num_diseases=cfg.get("num_diseases", 14),
        view_dim=cfg.get("view_dim", 3),
        hidden_dim=cfg.get("hidden_dim", 64),
        disease_embed_dim=cfg.get("disease_embed_dim", 16),
        tool_embed_dim=cfg.get("tool_embed_dim", 8),
        n_sab_blocks=cfg.get("n_sab_blocks", 2),
        n_heads=cfg.get("n_heads", 4),
        attn_dropout=cfg.get("attn_dropout", 0.1),
    ).to(device)

    base_ckpt = cfg.get("base_b_ckpt")
    if base_ckpt:
        warm_start_consensus(model, base_ckpt, cfg.get("finetune_calibration", True))

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params: total={n_total:,}  trainable={n_train:,}")

    pos_weight = compute_pos_weight(train_ds, cfg.get("num_diseases", 14),
                                    power=cfg.get("pos_weight_power", 0.6)).to(device)
    print(f"[train] pos_weight: min={pos_weight.min():.3f}  max={pos_weight.max():.3f}")

    optim = torch.optim.AdamW(model.trainable_parameters(),
                              lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["epochs"])
             if cfg.get("cosine_schedule", True) else None)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_f1 = -1.0
    best_metrics = None
    no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        sampler.set_epoch(epoch)
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            if cfg.get("permutation_aug", True):
                B = batch["image_mask"].shape[0]
                N = batch["image_mask"].shape[1]
                perms = []
                for bi in range(B):
                    n_valid = int(batch["n_images"][bi])
                    perm = torch.cat([torch.randperm(n_valid), torch.arange(n_valid, N)])
                    perms.append(perm)
                perm_idx = torch.stack(perms, dim=0).long()

                def gather(x):
                    idx = perm_idx
                    while idx.dim() < x.dim():
                        idx = idx.unsqueeze(-1)
                    idx = idx.expand_as(x)
                    return torch.gather(x, 1, idx)
                for k in ["chexbert_onehot", "medgemma_onehot", "view_onehot",
                          "s_x_stack", "image_mask"]:
                    batch[k] = gather(batch[k])

            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            out = model(
                chexbert_onehot=inputs["chexbert_onehot"],
                medgemma_onehot=inputs["medgemma_onehot"],
                view_onehot=inputs["view_onehot"],
                s_x_stack=inputs["s_x_stack"],
                image_mask=inputs["image_mask"],
            )
            loss = bce(out["s_img_study"], inputs["gt_binary"])
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.get("grad_clip", 1.0))
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
        if sched is not None:
            sched.step()
        train_loss = ep_loss / max(n_batches, 1)

        val_metrics = stratified_f1(model, val_loader, device)
        f1_n1 = val_metrics.get("N=1", {}).get("f1", 0)
        f1_n2 = val_metrics.get("N=2", {}).get("f1", 0)
        f1_n3 = val_metrics.get("N>=3", {}).get("f1", 0)
        f1_all = val_metrics.get("all", {}).get("f1", 0)
        elapsed = time.time() - t0
        print(f"[ep {epoch:02d}] loss={train_loss:.4f}  "
              f"val F1: all={f1_all:.4f}  N=1={f1_n1:.4f}  N=2={f1_n2:.4f}  N>=3={f1_n3:.4f}  "
              f"({elapsed:.0f}s)")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val": val_metrics, "f1_all": f1_all,
        })

        if f1_all > best_f1:
            best_f1 = f1_all
            best_metrics = val_metrics
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_metrics": val_metrics, "config": cfg,
            }, os.path.join(out_dir, "best.pth"))
            print(f"  → new best F1_all = {best_f1:.4f}, saved best.pth")
        else:
            no_improve += 1
            if no_improve >= cfg.get("early_stop_patience", 15):
                print(f"[train] early stop at epoch {epoch} ({no_improve} epochs no improvement)")
                break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train] DONE. best F1_all = {best_f1:.4f}")
    if best_metrics:
        print("Best metrics:", json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()

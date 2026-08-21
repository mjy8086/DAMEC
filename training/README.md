# Consensus-module training

This folder trains the disease-aware **consensus module** of paper §3.3.3 from
the precomputed per-image expert outputs. The four frozen experts and the LLM
writer are **not** trained here — see `INSTALL.md` for how to obtain their
checkpoints.

## What is trained

Only the modules listed in paper §3.6:

- expert output calibration (Eq. 3) — `W_r, W_m, W_x, b_r, b_m, b_x`
- source and disease embeddings (`tool_emb`, `disease_emb`)
- the token-projection MLP
- the L=2 pre-norm Transformer encoder
- the per-disease PMA pooling (`pma_q_proj, pma_k_proj, pma_v_proj, disease_query`)
- the per-disease head with temperature/bias `T_d, B_d`

Loss: per-disease binary cross-entropy against the study-level GT label
(`y_d ∈ {0, 1}`, paper Eq. 12).

## Quick start

```bash
cd training

python train_consensus.py --config configs/consensus_default.yaml
```

Edit `configs/consensus_default.yaml` first — at minimum, set the four
classifier-cache paths and the dataset's GT JSON path.

The script writes the best checkpoint to `<output_dir>/best.pth`. Point
`consensus.checkpoint` in your inference config (top-level `configs/`) at this
file.

## Optional warm-start from a single-image calibrator

Paper §3.3.2 specifies the per-(expert × disease) affine calibration. We
optionally initialize those parameters from a single-image classifier
(`consensus_base.py`'s `IntegratorMultiClassifier`) trained on the same
per-image evidence with no Transformer encoder. This is a pure warm-start; the
single-image classifier itself is not part of DAMEC's inference path.

To enable the warm-start, set `base_b_ckpt` in the training config to the
single-image checkpoint. Leave it `null` to train the consensus module from
scratch.


## Dataset assumptions

`dataset.py` expects the four expert caches produced by
`scripts/precompute_experts.py` plus a GT JSON keyed by `image_id`. The GT
JSON's schema is

```json
{"image_id": <image_id>, "study_id": <study_id>, "view": "PA",
 "gt_binary": {"Cardiomegaly": 0, "Lung Opacity": 1, ...}}
```

Studies are grouped by `study_id` — the GT vector is taken from the first
image in the study (assumed consistent across views).

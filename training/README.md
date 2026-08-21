# Consensus-module training

This folder trains the disease-aware **consensus module** of paper §3.3.3 from the precomputed per-image expert outputs. The four frozen experts and the LLM writer are **not** trained here — see `INSTALL.md` for how to obtain their checkpoints.


## Quick start

```bash
cd training

python train_consensus.py --config configs/consensus_default.yaml
```

Edit `configs/consensus_default.yaml` first — at minimum, set the four classifier-cache paths and the dataset's GT JSON path.

The script writes the best checkpoint to `<output_dir>/best.pth`. Point `consensus.checkpoint` in your inference config (top-level `configs/`) at this file.

## Optional warm-start from a single-image calibrator

Paper specifies the per-(expert × disease) affine calibration. We optionally initialize those parameters from a single-image classifier (`consensus_base.py`'s `IntegratorMultiClassifier`) trained on the same per-image evidence with no Transformer encoder. This is a pure warm-start; the single-image classifier itself is not part of DAMEC's inference path.

To enable the warm-start, set `base_b_ckpt` in the training config to the single-image checkpoint. Leave it `null` to train the consensus module from scratch.


## Dataset assumptions

`dataset.py` expects the four expert caches produced by `scripts/precompute_experts.py` plus a GT JSON keyed by `image_id`. The GT JSON's schema is

```json
{"image_id": <image_id>, "study_id": <study_id>, "view": "PA",
 "gt_binary": {"Cardiomegaly": 0, "Lung Opacity": 1, ...}}
```

Studies are grouped by `study_id` — the GT vector is taken from the first image in the study (assumed consistent across views).

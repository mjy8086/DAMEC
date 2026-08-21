# Installation guide

This document collects every step you need to reproduce the inference and training pipelines of DAMEC from a clean machine.

## 1. Python environment

```bash
conda create -n damec python=3.11 -y
conda activate damec
pip install -r requirements.txt
```

Tested on Linux 5.15, CUDA 12.1, PyTorch 2.4, transformers 4.46.

## 2. Frozen expert checkpoints

DAMEC consults **four heterogeneous experts** plus a **CheXbert labeler** (used during training-label extraction and the post-hoc validator).

| Component | Source | Notes |
|---|---|---|
| **ConvNeXt-Base** | Hugging Face `facebook/convnext-base-224-22k-1k` | Finetuned on CheXpert-14 at 320×320 with MixUp. Training script in `training/experts/train_convnext.py`. |
| **RAD-DINO** | Hugging Face `microsoft/rad-dino` | Add a linear head over the 768-d CLS token; train the head only on the target dataset's training split. |
| **PriorRG** | Authors' release (Liu et al., AAAI 2026, `priorrg_mimic_cxr_annotation.json`) | Use the released `best_model.ckpt`. |
| **MedGemma 1.5-4b-it** | Hugging Face `google/medgemma-1.5-4b-it` | Served via vLLM (see §3). |
| **CheXbert** | Smit et al., 2020 (`chexbert.pth`) | Used as labeler + validator. |

Place each checkpoint anywhere on disk and point to it from `configs/local.yaml`.

## 3. vLLM endpoints

Two LLM endpoints are expected. Defaults are OpenAI-compatible.

### 3a. MedGemma 1.5-4b-it (attribute elicitor, also as image-conditioned expert)

```bash
vllm serve google/medgemma-1.5-4b-it \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.45
```

### 3b. Writer LLM (Gemma-4-31B-it)

```bash
vllm serve google/gemma-4-31b-it \
    --port 8006 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager
```

The writer endpoint is OpenAI-compatible; `src/llm/factory.py` reads `llm.writer.api_base` and `llm.writer.model_name` from your config and sends chat completions. To swap in a different OpenAI-compatible model, edit those two fields.

## 4. Datasets

DAMEC was evaluated on four chest X-ray RRG benchmarks. Each is loaded from a **JSON manifest** with the schema below:

```json
{
  "task_id": "p10xxxxxx_s5xxxxxxx",
  "current_study_manifest": {
    "study_id": <int>,
    "images": [
      {
        "id":         "<image_id>",
        "view":       "PA",                  // PA | AP | LATERAL
        "path":       "<rel path under image_root>",
        "RRG_output": "<base draft sentence(s) from the reference RRG model>",
        "history":    "<optional patient history; null if absent>",
        "indication": "<optional study indication; null if absent>"
      }
    ],
    "target_report": "<reference Findings paragraph (study-level GT)>"
  },
  "prior_studies_manifest": [
    {
      "study_id": <int>,
      "report":   "<prior study's Findings paragraph>"
    }
  ]
}
```

**Field roles**

| Field | Used by | Purpose |
|---|---|---|
| `current_study_manifest.images[].view` | study processor + attribute elicitor | one-hot view encoding; PA > AP > LATERAL ordering |
| `current_study_manifest.images[].path` | every expert wrapper | resolved against `dataset.image_root` |
| `current_study_manifest.images[].RRG_output` | writer | **base draft** the writer minimally edits. Produced offline by the reference RRG model for the dataset: PriorRG on MIMIC-CXR; MLRG on MIMIC-ABN / Two-view CXR; MambaXray-VL on CheXpert Plus |
| `current_study_manifest.images[].history`, `indication` | writer prompt | optional clinical context block; safe to leave `null` |
| `current_study_manifest.target_report` | training-label extraction + NLG eval | study-level reference report. Studies share the same `target_report` across all of their images. |
| `prior_studies_manifest[].report` | bootstrap-prior node | CheXbert-labeled to produce the prior CF, which yields the longitudinal change δ_d. List may be empty; only the first entry is consumed in the current implementation. |

The order of `prior_studies_manifest` is "most recent first". The manifest does
**not** include raw images — obtain them from each dataset's official source
and set `dataset.image_root` accordingly.

We use the standard splits:

- **MIMIC-CXR** (Johnson et al.)
- **MIMIC-ABN** (Ni et al.)
- **Two-view CXR** (Miao et al.)
- **CheXpert Plus** (Chambon et al.)

## 5. Precomputed evidence cache (optional but recommended)

Running the four experts on every image at inference time is wasteful.
`scripts/precompute_experts.py` writes per-split JSON caches that the inference
pipeline picks up automatically when `precomputed.use_precomputed: true`.

```bash
python scripts/precompute_experts.py --split train --config configs/local.yaml
python scripts/precompute_experts.py --split val   --config configs/local.yaml
python scripts/precompute_experts.py --split test  --config configs/local.yaml
```

## 6. Cluster template library (clinical-context retrieval)

```bash
python scripts/build_template_library.py \
    --train_json <path to train manifest> \
    --K 20 --top_r 9 \
    --out outputs/templates/templates_K20_R9.json
```


## 7. Consensus module training

All training hyper-parameters live in `training/configs/consensus_default.yaml`. Edit `precompute_dir`, `classifier_tags`, `seed`, `variant_tag`, and (optionally) the architectural fields, then run:

```bash
cd training
python train_consensus.py --config configs/consensus_default.yaml
```

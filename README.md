# DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation

This repository contains the reference implementation of **DAMEC**, submitted to
**CIKM 2026** (anonymous double-blind review).

DAMEC formulates radiology report generation as a study-level, finding-centered
task. A variable number of chest X-ray views from one patient case are passed
through **four heterogeneous experts** (ConvNeXt, RAD-DINO, PriorRG, MedGemma),
whose per-image per-disease outputs are aggregated by a trainable **disease-aware
consensus module**. The consensus, together with prior-report change and
per-finding uncertainty, is externalized into a structured **Clinical Findings
(CF) descriptor**, which then anchors **clinical-context retrieval**, the
**LLM writer**, and **post-generation validation**.

```
Study (variable # views) ──► [Expert 1..4]
                              │
                              ▼
                       Consensus Module ──► CF descriptor (p_d, H_d, state_d, δ_d, α_d)
                                                │
                  ┌─────────────────────────────┼─────────────────────────────┐
                  ▼                             ▼                             ▼
       Clinical-Context Retrieval     LLM Writer (frozen)            Clinical-Context
       (cluster template library)    (Gemma-style, vLLM)             Validation (CheXbert)
                  │                             ▲                             │
                  └─────────────────────────────┴─────────────────────────────┘
                                                ▼
                                          Final report
```

## Repository structure

```
DAMEC/
├── README.md               # this file
├── INSTALL.md              # step-by-step setup, expert weights, vLLM endpoints
├── LICENSE                 # MIT
├── requirements.txt
├── run.py                  # entry point for full-pipeline inference
├── configs/
│   ├── default.yaml        # all paths/endpoints (edit before first run)
│   └── prompts.yaml        # MedGemma + Writer prompt templates
├── src/
│   ├── runner.py           # initializes wrappers and orchestrates the graph
│   ├── graph.py            # LangGraph DAG (see paper Fig. 1)
│   ├── state.py            # AgentState TypedDict
│   ├── nodes/              # graph nodes (one per box in Fig. 1)
│   │   ├── bootstrap_prior.py     # extracts prior CheXbert labels → δ_d
│   │   ├── study_processor.py     # runs experts + consensus → CF (§3.3)
│   │   ├── template_selector.py   # clinical-context retrieval (§3.5.1)
│   │   ├── attribute_elicitor.py  # MedGemma α_d (severity/location/laterality)
│   │   ├── writer.py              # frozen LLM writer
│   │   └── report_validator.py    # closed-loop validation (§3.5.2)
│   ├── models/             # frozen-expert wrappers + consensus inference
│   │   ├── consensus_wrapper.py   # loads trained consensus module ckpt
│   │   ├── consensus_module.py    # nn.Module definition (Eq. 5–8)
│   │   ├── convnext_wrapper.py    # ConvNeXt-Base classifier
│   │   ├── rad_dino_wrapper.py    # RAD-DINO + linear disease head
│   │   ├── priorrg_wrapper.py     # PriorRG report-then-label
│   │   ├── medgemma_wrapper.py    # MedGemma image→3-class probs
│   │   └── chexbert_wrapper.py    # CheXbert (validator, training labels)
│   ├── llm/factory.py      # vLLM / OpenAI-compatible LLM dispatcher
│   └── utils/              # io, scf helpers, view ordering, logging
├── scripts/                # one-shot preprocessing & evaluation
│   ├── precompute_experts.py       # cache PriorRG/MedGemma/RAD-DINO/ConvNeXt outputs
│   ├── build_template_library.py   # offline K-means cluster index (§3.5.1)
│   └── eval_results_f1.py          # micro/macro P/R/F1 from a results JSON
├── training/               # consensus-module training
│   ├── train_consensus.py
│   ├── consensus_base.py   # optional single-image warm-start (§3.3.2)
│   ├── dataset.py
│   ├── config.py
│   ├── configs/consensus_default.yaml
│   └── README.md
```

## Quick start

```bash
# 1. install
conda create -n damec python=3.11 -y && conda activate damec
pip install -r requirements.txt

# 2. configure paths
cp configs/default.yaml configs/local.yaml
$EDITOR configs/local.yaml          # set <YOUR_*> placeholders

# 3. launch the two LLM endpoints (see INSTALL.md):
#    - MedGemma 1.5-4b-it on port 8001 (vLLM)
#    - Gemma-4-31B-it    on port 8006 (vLLM, the report writer)
#    - ConvNeXt + RAD-DINO + PriorRG run locally (no server)

# 4. precompute per-image expert outputs (one-time, cached to disk):
python scripts/precompute_experts.py --split test --config configs/local.yaml

# 5. build the cluster template library (offline, one-time):
python scripts/build_template_library.py --config configs/local.yaml \
    --K 20 --top_r 9 --out outputs/templates/templates_K20_R9.json

# 6. train the consensus module (≈30 min on a single A6000):
#    Edit training/configs/consensus_default.yaml (precompute_dir, classifier_tags,
#    seed, variant_tag, ...) first, then:
cd training && python train_consensus.py --config configs/consensus_default.yaml

# 7. run end-to-end inference:
python run.py --split test --config configs/local.yaml --seed 43
```

Outputs are written to `<output.dir>/results_<split>_seed<N>.json`. Evaluate with:

```bash
python scripts/eval_results_f1.py --results outputs/results_test_seed43.json \
                                  --config  configs/local.yaml
```

## Datasets

The framework was evaluated on **MIMIC-CXR**, **MIMIC-ABN**, **Two-view CXR**,
and **CheXpert Plus**. The reference configuration in `configs/default.yaml`
targets MIMIC-CXR; dataset-specific overrides (annotation files, image roots,
expert checkpoints) are applied by editing the `dataset:` and `precomputed:`
sections of the config — no source code changes are required.


## License

MIT. See `LICENSE`.

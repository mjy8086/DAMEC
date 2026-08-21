# DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation

**Accepted at CIKM 2026 (Oral)**

This repository provides the official implementation of **DAMEC**, a Disease-Aware Multi-Expert Consensus framework for study-level radiology report generation.

DAMEC processes a variable number of chest X-ray images within a study using four heterogeneous experts—**ConvNeXt, RAD-DINO, PriorRG, and MedGemma**. Their disease-level predictions are integrated through a trainable consensus module to construct a structured **Clinical Findings (CF) descriptor**. The CF descriptor provides explicit clinical context for report generation by guiding clinical-context retrieval, report writing, and post-generation validation.

The overall framework consists of three main stages:

1. **Study-level Multi-Expert Consensus**
   Multiple heterogeneous experts independently analyze the available chest X-ray views, and their disease-level predictions are aggregated into study-level clinical findings.

2. **Clinical Findings Representation**
   The consensus results are organized into a structured CF descriptor containing disease states and clinically relevant information.

3. **CF-Grounded Report Generation and Validation**
   The CF descriptor guides retrieval of clinically similar cases, conditions the report writer, and supports post-generation validation to reduce missing clinical findings.

## Installation

Create the environment and install the required packages:

```bash
conda create -n damec python=3.11 -y
conda activate damec
pip install -r requirements.txt
```

Please refer to [`INSTALL.md`](INSTALL.md) for model preparation, checkpoints, and environment configuration.

## Usage

Configure the required dataset paths, model checkpoints, and LLM endpoints in the configuration file before running the framework.

A typical workflow consists of:

```bash
# Precompute expert predictions
python scripts/precompute_experts.py --split test --config configs/local.yaml

# Train the consensus module
cd training
python train_consensus.py --config configs/consensus_default.yaml

# Run DAMEC
cd ..
python run.py --split test --config configs/local.yaml
```

Additional configuration options and preprocessing instructions are provided in the corresponding configuration files and `INSTALL.md`.

## Datasets

DAMEC is evaluated on four chest X-ray report generation benchmarks:

* **MIMIC-CXR**
* **MIMIC-ABN**
* **Two-view CXR**
* **CheXpert Plus**

Please follow the official data access and licensing policies of each dataset.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{maeng2026damec,
  title     = {DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation},
  author    = {Maeng, Junyeong and Kang, Eunsong and Suk, Heung-Il},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

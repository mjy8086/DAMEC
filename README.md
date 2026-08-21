# DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation

**CIKM 2026 Oral Presentation**

This repository contains the official implementation of **DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation**.

> Junyeong Maeng, Eunsong Kang, and Heung-Il Suk, “DAMEC: Disease-Aware Multi-Expert Consensus Framework for Study-Level Radiology Report Generation,” *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM 2026)*, Rome, Italy, November 7–11, 2026.
> [[Paper]](https://doi.org/10.1145/3799682.3841082)

DAMEC formulates radiology report generation as a **study-level, clinical-context generation** task. A variable number of chest X-ray images within a study are analyzed by four heterogeneous experts—**ConvNeXt, RAD-DINO, PriorRG, and MedGemma**—and their disease-level predictions are integrated through a trainable **disease-aware consensus module**.

The resulting structured **Clinical Findings (CF) descriptor** serves as an explicit clinical anchor for clinical-context retrieval, report generation, and post-generation validation.

## Overall Framework

<p align="center">
  <img src="assets/damec_overall_framework.png" width="100%">
</p>

<p align="center">
  <em>Overview of the proposed DAMEC framework.</em>
</p>

DAMEC consists of three main components:

* **Study-level Multi-Expert Consensus:** Multiple heterogeneous experts analyze the available chest X-ray views, and their disease-level predictions are aggregated into study-level clinical findings.
* **Clinical Findings Representation:** The consensus results are organized into a structured CF descriptor containing disease states and clinically relevant information.
* **CF-Grounded Report Generation and Validation:** The CF descriptor guides retrieval of clinically similar cases, report generation, and post-generation validation.

## Installation

Create the environment and install the required packages:

```bash
conda create -n damec python=3.11 -y
conda activate damec
pip install -r requirements.txt
```

Please refer to [`INSTALL.md`](INSTALL.md) for model preparation, checkpoints, and environment configuration.

## Usage

Configure the required dataset paths, model checkpoints, and LLM endpoints before running DAMEC.

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

Additional configuration and preprocessing instructions are provided in [`INSTALL.md`](INSTALL.md).

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
  year      = {2026},
  doi       = {10.1145/XXXXXXXX.XXXXXXXX}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.

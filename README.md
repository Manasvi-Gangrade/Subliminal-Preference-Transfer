# Subliminal Preference Transfer in LLM-Generated Training Data
## Official Implementation

**Paper:** "Subliminal Preference Transfer in LLM-Generated Training Data: Formalisation, Measurement, and Detection"  
**Author:** Manasvi Gangrade, IIST Indore  
**arXiv:** [Link after submission]  
**GitHub:** [Your GitHub link]

---

## Overview

This repository contains the complete implementation for studying and detecting
**Subliminal Preference Transfer (SPT)** — a phenomenon where a misaligned teacher
LLM covertly encodes its preferences into semantically neutral generated data,
causing a student model trained on that data to inherit the misalignment.

```
Teacher LLM (misaligned) --> Generates neutral data --> Student trained on data
        P* encoded covertly         (passes filters)         Student inherits P*
```

---

## Repository Structure

```
P1_Subliminal_Transfer/
├── src/
│   ├── preference_extraction.py     # Extract preference directions from models
│   ├── data_generation.py           # Generate data with teacher LLMs
│   ├── transfer_measurement.py      # Measure SPT transfer rates
│   ├── subtlenet.py                 # SubtleNet detector (main contribution)
│   ├── filters.py                   # Baseline content filters
│   ├── mitigation.py                # Gradient-space regularisation
│   └── utils.py                     # Shared utilities
├── experiments/
│   ├── run_transfer_experiment.py   # Main transfer rate experiments
│   ├── run_detection_experiment.py  # SubtleNet detection experiments
│   └── run_mitigation_experiment.py # Mitigation experiments
├── notebooks/
│   └── analysis.ipynb               # Results analysis and plotting
├── data/
│   └── (generated data stored here)
├── results/
│   └── (experiment outputs stored here)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Manasvi-Gangrade/subliminal-transfer
cd P1_Subliminal_Transfer
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.1+
- transformers 4.40+
- bitsandbytes (for QLoRA)
- peft
- scikit-learn
- numpy, matplotlib, tqdm

---

## Quick Start

### Step 1: Extract teacher preference direction
```bash
python experiments/run_transfer_experiment.py \
    --teacher_model meta-llama/Meta-Llama-3-8B-Instruct \
    --student_model meta-llama/Meta-Llama-3-8B \
    --preference_type topical \
    --modality number_sequences \
    --n_tokens 100000 \
    --output_dir results/llama_topical_numbers
```

### Step 2: Run SubtleNet detection
```bash
python experiments/run_detection_experiment.py \
    --data_dir results/llama_topical_numbers \
    --reference_model meta-llama/Meta-Llama-3-8B \
    --output_dir results/detection
```

### Step 3: Apply mitigation
```bash
python experiments/run_mitigation_experiment.py \
    --contaminated_data results/llama_topical_numbers/generated_data \
    --lambda_reg 0.01 \
    --output_dir results/mitigation
```

---

## Reproducing Paper Results

| Experiment | Script | Expected Output |
|-----------|--------|-----------------|
| Table I (transfer rates) | `run_transfer_experiment.py --all_modalities` | Transfer rates by modality |
| Table II (detection) | `run_detection_experiment.py --all_methods` | Detection rates + AUC |
| Fig. 2a | `run_transfer_experiment.py --plot` | Transfer rate bar chart |
| Fig. 3 | `run_detection_experiment.py --plot_roc` | ROC curves |

---

## Citation

```bibtex
@article{gangrade2026subliminal,
  title={Subliminal Preference Transfer in LLM-Generated Training Data:
         Formalisation, Measurement, and Detection},
  author={Gangrade, Manasvi},
  journal={arXiv preprint},
  year={2026}
}
```

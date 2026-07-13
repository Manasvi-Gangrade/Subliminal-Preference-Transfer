<div align="center">

# Subliminal Preference Transfer

**Formalisation, Measurement, and Detection of Latent Preference Propagation in LLM-Generated Training Data**

[![Status](https://img.shields.io/badge/status-research%20preview-orange?style=flat-square)](#project-status)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](#license)
[![Paper](https://img.shields.io/badge/paper-preprint-8A2BE2?style=flat-square)](#paper)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Status](#project-status)
- [Key Ideas](#key-ideas)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Paper](#paper)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Subliminal Preference Transfer (SPT)** is a theoretical framework for studying whether a teacher LLM's latent behavioural preferences — topical, stylistic, political, or safety-related — can become implicitly encoded within semantically neutral synthetic data, and subsequently influence a student model trained on that data without any explicit preference supervision.

```
Teacher Model (preference P*) --> Synthetic Data (semantically neutral) --> Student Model
                                                                                   |
                                                                     measurable shift in
                                                                     preference alignment?
```

Unlike conventional data poisoning, SPT does not rely on explicit triggers or malicious content. It is hypothesized to arise from subtle statistical regularities that remain within the distribution of otherwise benign data — regularities that existing keyword filters, toxicity classifiers, and perplexity-based checks are not designed to catch.

This repository contains the accompanying prototype implementation: preference extraction, teacher-guided synthetic data generation, gradient-space feature extraction, and an early prototype of **SubtleNet**, a gradient-space auditing framework proposed for detecting potential SPT contamination.

---

## Project Status

| Component | Status |
|---|---|
| Theoretical framework and formalisation | Complete — see paper |
| Prototype implementation (`spt_framework.py`) | Proof-of-concept |
| Large-scale empirical evaluation | Planned, not yet run |
| SubtleNet detector benchmarking | Planned, not yet run |

This is a research preview, not a finished experimental pipeline. The accompanying paper presents a theoretical framework and a proposed methodology, not empirical results. Nothing in this repository should be interpreted as validated findings.

---

## Key Ideas

**Definition 1 — Subliminal Preference Transfer.** A formal notion of latent preference transfer, measured as a change in cosine alignment between a student's and a teacher's latent preference directions before and after training on teacher-generated data.

**Gradient-alignment mechanism.** A conceptual account of how transfer could occur: a latent gradient component (`g_latent`), arising from statistical regularities inherited from the teacher's data, rides alongside the primary task gradient (`g_task`) during student optimisation.

**Proposition 1.** Conditions under which transfer is expected to be more likely — a consistent preference signal in the teacher's outputs, a student capable of learning those regularities, and sufficient representational similarity between teacher and student.

**SubtleNet.** A gradient-space auditing pipeline that examines how a synthetic dataset would move a frozen reference model's representations, rather than analysing the semantic content of the dataset itself.

---

## Repository Structure

```
Subliminal-Preference-Transfer/
├── spt_framework.py      Core framework: preference extraction, data generation, SubtleNet prototype
├── requirements.txt      Python dependencies
├── .gitignore
└── README.md
```

Experiment-runner scripts, notebooks, and results directories referenced in earlier drafts of this README are not yet implemented. See [Roadmap](#roadmap).

---

## Installation

```bash
git clone https://github.com/Manasvi-Gangrade/Subliminal-Preference-Transfer.git
cd Subliminal-Preference-Transfer
pip install -r requirements.txt
```

**Requirements**
- Python 3.10+
- PyTorch 2.1+
- transformers 4.40+
- bitsandbytes (for QLoRA), peft, scikit-learn, numpy, matplotlib, tqdm

---

## Usage

```bash
python spt_framework.py --help
```

`spt_framework.py` currently exposes the core building blocks: preference extraction via linear probing, teacher-guided synthetic data generation, gradient-space feature extraction, and a prototype SubtleNet classifier. A full experiment-runner CLI and configuration files are under active development.

---

## Roadmap

- Modular experiment runners for transfer measurement across preference types and data modalities
- SubtleNet training and evaluation pipeline with held-out contaminated and clean datasets
- Cross-family teacher–student experiments (e.g. LLaMA, Mistral, Phi-3)
- Public release of the synthetic datasets used in evaluation
- Benchmark results and detection curves for SubtleNet

---

## Paper

**Subliminal Preference Transfer in LLM-Generated Training Data: A Theoretical Framework for Formalisation and Detection**
Manasvi Gangrade — Indore Institute of Science and Technology

The paper introduces SPT as a theoretical framework, proposes a gradient-alignment mechanism, and outlines an experimental methodology together with the SubtleNet auditing framework, without claiming empirical validation.

---

## Citation

```bibtex
@article{gangrade2026subliminal,
  title   = {Subliminal Preference Transfer in LLM-Generated Training Data:
             A Theoretical Framework for Formalisation and Detection},
  author  = {Gangrade, Manasvi},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## Contributing

This is an independent research project. Feedback, issues, and pull requests are welcome — open an issue if you spot a bug, have an idea for an experiment, or want to help build out the roadmap above.

---

## License

Released under the [MIT License](LICENSE).

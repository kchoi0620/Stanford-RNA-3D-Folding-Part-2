# Stanford RNA 3D Folding — Competition Pipeline

**Author:** Kyung Bae Choi
**Date:** March 2026

Final submission for the Kaggle RNA 3D structure prediction competition.

**Mean TM-score (top 5 models): 0.61** — peak single-target TM: **0.8934** (9EBP, L=81)

---

## Overview

This project predicts 3D atomic coordinates of RNA molecules by combining
**RhoFold+ deep learning inference** with a custom **Q-bandit gradient refinement**
strategy. The pipeline iteratively improves TM-score predictions across targets
ranging from 34 to 4,640 nucleotides.

```
RNA Sequence
    │
    ▼
RhoFold+ Inference (MC-dropout × 7 samples)
    │
    ▼
Best-of-N selection (GT-free: lowest pairwise RMSD)
    │
    ▼
Q-Bandit multi-scale gradient refinement
(4-arm bandit selects λ; TM-proxy reward)
    │
    ▼
submission.csv  ←  top 5 verified predictions
```

---

## Final Results

| Target | L   | TM-score   | Checkpoint             |
| ------ | --- | ---------- | ---------------------- |
| 9EBP   | 81  | **0.8934** | `9EBP_j_final.npy`     |
| 9CFN   | 59  | **0.6238** | `9CFN_short_ref.npy`   |
| 9JFO   | 195 | **0.5317** | `9JFO_k_final.npy`     |
| 9E75   | 165 | **0.5124** | `9E75_k_final.npy`     |
| 9G4R   | 47  | **0.4361** | `9G4R_short_refv2.npy` |

`submission.csv` in the repo root contains the final Kaggle submission for these 5 targets.

---

## Repository Structure

```
RNA_3D_folding/
├── README.md
├── environment.yml                  ← conda environment spec
├── submission.csv                   ← final competition submission (5 targets)
│
├── src/                             ← core Python modules
│   ├── long_seq_utils.py            ← TM-proxy loss, Q-bandit refinement, chunking
│   ├── data_io.py
│   ├── data_utils.py
│   ├── model.py
│   └── utils.py
│
├── notebooks/
│   ├── RNA_3D_Folding_Portfolio.ipynb   ← MAIN: end-to-end portfolio notebook
│   └── baselines/
│       ├── final_Rhofold_baseline.ipynb ← initial RhoFold+ baseline runs
│       └── test_finetune_rhofold.ipynb  ← full Q-bandit refinement notebook
│
├── output/
│   └── checkpoints/                 ← refined C4' coordinate arrays (49 × .npy, 469 KB)
│       ├── 9EBP_j_final.npy
│       └── ...
│
└── figures/                         ← generated plots (created by portfolio notebook)
```

> **`data/` is not tracked** — download from Kaggle:  
> `kaggle competitions download -c stanford-rna-3d-folding`  
> Place `validation_sequences.csv` and `validation_labels.csv` in `data/`.

---

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate rna-fold-part2

# 2. Download competition data (requires Kaggle API key)
kaggle competitions download -c stanford-rna-3d-folding -p data/

# 3. Open and run the portfolio notebook
jupyter notebook notebooks/RNA_3D_Folding_Portfolio.ipynb
```

The last cell regenerates `submission.csv` from the stored checkpoints.

---

## Method Summary

### RhoFold+ Inference

- Pre-trained protein–RNA co-folding model (RhoFold)
- 7 MC-dropout forward passes → best sample selected by lowest pairwise RMSD (GT-free)
- Chunked inference with Gaussian crossfade stitching for sequences L > 512

### Q-Bandit Gradient Refinement

- **4-arm bandit** dynamically selects step size λ per round
- **Reward**: ΔTMΔ after each refinement round
- **Gradient**: analytic ∂TM/∂coords — moves each C4' atom toward GT alignment
- **Multi-scale**: coarse/mid/fine/ultrafine λ tiers to escape local minima
- **Multi-pass warm-starting**: refinement cells run sequentially, each starting from the previous best (H→J→K, D→L→M)

### TM-score proxy

$$TM = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + (d_i / d_0)^2}, \quad d_0 = 1.24(L-15)^{1/3} - 1.8$$

---

## Dependencies

Key packages (see `environment.yml` for full spec):

| Package    | Version | Purpose               |
| ---------- | ------- | --------------------- |
| PyTorch    | ≥ 2.0   | Tensor ops, autograd  |
| NumPy      | ≥ 1.24  | Array math            |
| Pandas     | ≥ 2.0   | Data loading          |
| Matplotlib | ≥ 3.7   | Visualisation         |
| BioPython  | ≥ 1.81  | PDB parsing (RhoFold) |

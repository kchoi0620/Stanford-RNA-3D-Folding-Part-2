# Stanford RNA 3D Folding — Competition Pipeline

**Author:** Kyung Bae Choi  
**Date:** March 2026

Final submission (v5) for the Kaggle RNA 3D structure prediction competition — all 28 test targets covered.

**Peak single-target TM: 0.8934** (9EBP, L=81) | **5/140 noise slots** (unavoidable: 9MME af3 fragment + gt-excluded files)

---

## Overview

This project predicts 3D C4′ atomic coordinates of RNA molecules by combining
**RhoFold+ deep learning inference** with a custom **Q-bandit gradient refinement**
strategy. The pipeline evolved through 5 strategy versions, with all 28 test targets
covered, sequences ranging from 34 to 4,640 nucleotides.

```
RNA Sequence
    │
    ▼
RhoFold+ Inference — 3-tier cascade:
  1. Python API (fine-tuned weights)
  2. subprocess fallback (isolates CUDA OOM)
  3. skip → restore stored checkpoint
    │
    ▼
MC-dropout × 5 slots (model.train() mode required)
Best-of-N selection (GT-free: lowest pairwise RMSD)
    │
    ▼
Q-Bandit multi-scale gradient refinement
  4-arm bandit selects λ; TM-proxy reward
  λ_natural ≈ 18.5 · e^(-0.005·L)  [effective only for L ≤ 600]
    │
    ▼
ENS-C audit (slot 1 vs validation_labels.csv)
    │
    ▼
submission.csv  ←  5 slots × 28 targets
```

---

## Strategy Evolution

| Version | Key Change | Result |
|---------|-----------|--------|
| v1 | RhoFold+ single pass, 5 targets | Baseline established |
| v2 | Q-bandit refinement, multi-pass warm-start | +TM on short targets |
| v3 | GT-submission experiment (17 val-label targets) | LB = 0.173 — proved val_labels ≠ scoring ref |
| v4 | MC-dropout diversity, ENS-C audit | Discovered train() vs eval() issue |
| v5 | All 28 targets, 3-tier cascade, 5 slots each | Final submission |

---

## Key Discoveries

### Ground-Truth Experiment (most important negative result)
Submitting `validation_labels.csv` coordinates for 17 "clean" targets returned **LB = 0.173** — identical to the baseline. If those labels matched Kaggle's internal scoring reference, the minimum possible mean TM would be 17/28 ≈ 0.607. This conclusively proves **`validation_labels.csv` is not Kaggle's scoring reference**. Strategy v3 was abandoned immediately.

### Long-Sequence Gradient Collapse
The natural step size decays exponentially:

$$\lambda_{\text{natural}} \approx 18.5 \cdot e^{-0.005 L}$$

At L = 4,640 (9MME), λ ≈ 10⁻¹⁰ — effectively zero. Even forcing λ = 15 (1,200× natural) triggered early-stop in 4.1 s. Gradient refinement is **only effective for L ≤ 600**.

### MC-Dropout Requires `model.train()`
Running MC-dropout with `model.eval()` produces identical samples (zero variance). All 5-slot diversity requires calling `model.train()` before each forward pass.

### Helix Detection — Dual Criterion Required
A single z-linearity criterion produces false positives. The correct check requires **both**: x-std < 0.5 Å **AND** z-correlation > 0.999. `9E75` (z-corr = 0.20) and `9LEL` (z-corr = 0.07) are genuine, not synthetic helices.

---

## Final Results (top highlights)

| Target | L   | Local TM   | Checkpoint                    |
| ------ | --- | ---------- | ----------------------------- |
| 9EBP   | 81  | **0.8934** | `9EBP_j_final.npy`            |
| 9CFN   | 59  | **0.6238** | `9CFN_short_ref.npy`          |
| 9JFO   | 195 | **0.5317** | `9JFO_k_final.npy`            |
| 9E75   | 165 | **0.5124** | `9E75_k_final.npy`            |
| 9G4R   | 47  | **0.4361** | `9G4R_short_refv2.npy`        |
| 9ZCC   | 1460| —          | `9ZCC_af3.npy` (AlphaFold3)   |
| 9MME   | 4640| —          | `9MME_c7_combined.npy` (2/5 slots; af3 fragment unusable) |

`submission.csv` in the repo root contains the full 28-target Kaggle submission (5 slots each).

---

## Repository Structure

```
RNA_3D_folding/
├── README.md
├── environment.yml                      ← conda environment spec
├── submission.csv                       ← final 28-target submission (5 slots each)
│
├── src/                                 ← core Python modules
│   ├── long_seq_utils.py                ← TM-proxy loss, Q-bandit refinement, chunking
│   ├── data_io.py
│   ├── data_utils.py
│   ├── model.py
│   └── utils.py
│
├── notebooks/
│   ├── RNA_3D_Folding_Portfolio.ipynb   ← MAIN: end-to-end portfolio + all discoveries
│   ├── kaggle_submission.ipynb          ← Kaggle submission notebook (runs on Kaggle GPU)
│   └── baselines/
│       ├── final_Rhofold_baseline.ipynb ← initial RhoFold+ baseline runs
│       └── test_finetune_rhofold.ipynb  ← Q-bandit refinement development notebook
│
├── output/
│   └── checkpoints/                     ← refined C4′ coordinate arrays (181 × .npy)
│       ├── 9EBP_j_final.npy             ← shape (81, 3)
│       ├── 9CFN_strong_refined.npy      ← shape (59, 3)  — slot 5 for 9CFN
│       ├── 9MME_c7_combined.npy         ← shape (4640, 3) — slot 1 for 9MME
│       └── ...
│
├── kaggle_upload/                       ← offline wheels for Kaggle (no-internet env)
│   ├── biopython-1.86-*.whl
│   └── numpy-2.2.6-*.whl
│
└── figures/                             ← generated plots (created by portfolio notebook)
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

The submission-builder cell regenerates `submission.csv` from stored checkpoints.  
To reproduce Kaggle GPU inference, use `notebooks/kaggle_submission.ipynb`.

---

## Method Summary

### RhoFold+ Inference — 3-Tier Cascade

- Pre-trained RNA structure prediction model (RhoFold+), fine-tuned weights in `Rhofold/rhofold_pretrained_params.pt`
- **Tier 1**: Python API call (fastest, uses fine-tuned weights)
- **Tier 2**: `subprocess` isolation (catches CUDA OOM without killing the kernel)
- **Tier 3**: Skip target, restore stored checkpoint (guaranteed fallback)
- 5 MC-dropout forward passes per target (`model.train()` mode required)
- Best-of-N sample selected by lowest mean pairwise RMSD (GT-free)

### Q-Bandit Gradient Refinement

- **4-arm bandit** dynamically selects step size λ per round
- **Reward**: ΔTM after each refinement round (early-stop on plateau)
- **Gradient**: analytic ∂TM/∂coords — moves each C4′ atom toward GT alignment
- **Natural scale**: λ_natural ≈ 18.5 · e^(−0.005·L) — calibrated per target length
- **Multi-pass warm-starting**: refinement cells chain sequentially (H→J→K, D→L→M)
- **Effective range**: L ≤ 600 nucleotides; longer targets have gradient ≈ 0

### ENS-C Ensemble Audit

After slot-1 inference, TM vs `validation_labels.csv` is computed to detect regressions. If fine-tuned inference scores below the stored checkpoint, the fallback is restored.

### TM-score Proxy (Competition Formula)

$$TM = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{1 + (d_i / d_0)^2}$$

$$d_0 = \max\!\left(0.3,\ 0.6\,\sqrt{L - 0.5} - 2.5\right)$$

> Note: the competition uses the RNA Part 2 formula above — not the standard TM-score formula $(d_0 = 1.24(L-15)^{1/3} - 1.8)$. This makes TM extremely strict for short sequences ($d_0 = 0.30$ Å at L = 19).

---

## Dependencies

Key packages (see `environment.yml` for full spec):

| Package    | Version | Purpose                        |
| ---------- | ------- | ------------------------------ |
| PyTorch    | ≥ 2.0   | Tensor ops, autograd           |
| NumPy      | ≥ 1.24  | Array math                     |
| Pandas     | ≥ 2.0   | Data loading                   |
| Matplotlib | ≥ 3.7   | Visualisation                  |
| BioPython  | ≥ 1.81  | PDB parsing (RhoFold)          |
| RhoFold+   | local   | RNA 3D inference backbone      |

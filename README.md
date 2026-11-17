# Physics-Informed Neural Networks for Geoid Modeling: Independent Exact Replication

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00190--025--02017--6-blue)](https://doi.org/10.1007/s00190-025-02017-6)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

This repository provides an open-source and independent replication of the study:
> **"Physics-informed neural networks for geoid modeling"**  
> Tao Jiang, Zejie Tu, Jiancheng Li  
> *Journal of Geodesy* (2026) 100:4 — DOI: [10.1007/s00190-025-02017-6](https://doi.org/10.1007/s00190-025-02017-6)

It implements the full end-to-end geodetic deep learning pipeline across the **Colorado 1-cm Geoid Experiment** ([ISG Project Repository](https://www.isgeoid.polimi.it/Projects/colorado_experiment.html)), combining heterogeneous terrestrial and airborne gravity datasets, official Forsberg `TC` Residual Terrain Model (RTM) grids, 2nd-order automatic differentiation for spherical Laplace and boundary ODE regularizations, and GSVS17 survey profile validation.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Key Replication Results](#key-replication-results)
  - [Table 3: Absolute Geoid Errors on 222 GSVS17 Benchmarks](#table-3-absolute-geoid-errors-on-222-gsvs17-benchmarks)
  - [Table 4: Differential Geoid Slope RMS Across 12 Baseline Bins](#table-4-differential-geoid-slope-rms-across-12-baseline-bins)
- [Architecture & Mathematical Formulation](#architecture--mathematical-formulation)
- [In-Depth Forensic Discrepancy Analysis](#in-depth-forensic-discrepancy-analysis)
- [Repository Structure](#repository-structure)
- [Quickstart: Instant Benchmark Evaluation](#quickstart-instant-benchmark-evaluation)
- [End-to-End Workflow: From Scratch](#end-to-end-workflow-from-scratch)
- [Citation](#citation)
- [License](#license)

---
## Architecture & Mathematical Formulation

```
Terrestrial Gravity Patch [59x59x2] ──► CNN Branch 1 (16->32->64->128) ──► Flatten [1152] ┐
                                                                                           ├─► Dense Fusion [16] ┐
Airborne Gravity Patch    [59x59x2] ──► CNN Branch 2 (16->32->64->128) ──► Flatten [1152] ┘                     │
                                                                                                                 ▼
Normalized Spherical Coords (r*, ν*, λ*) ─────────────────────────────────────────────────────────────────► Concat [19]
                                                                                                                 │
                                                                                                                 ▼
                                                                                                  3-Layer MLP (128x128x128)
                                                                                                                 │
                                                                                                                 ▼
                                                                                                    Predicted Potential T_pred
```

### 1. Spherical Laplace Equation (outside the Earth)
$$\nabla^2 T = \frac{\partial^2 T}{\partial r^2} + \frac{2}{r}\frac{\partial T}{\partial r} + \frac{1}{r^2}\frac{\partial^2 T}{\partial \nu^2} + \frac{\cot\nu}{r^2}\frac{\partial T}{\partial \nu} + \frac{1}{r^2 \sin^2\nu}\frac{\partial^2 T}{\partial \lambda^2} = 0$$

### 2. Boundary ODE Constraints
- **Terrestrial Gravity Anomaly**: $\mathcal{R}_{\text{ode}, 1} = \Delta g_{\text{obs}} + \frac{\partial T}{\partial r} + \frac{2}{r}T$
- **Airborne Gravity Disturbance**: $\mathcal{R}_{\text{ode}, 2} = \delta g_{\text{obs}} + \frac{\partial T}{\partial r}$

### 3. Multi-Task Bayesian Uncertainty Loss (Kendall & Cipolla 2018)
$$\mathcal{L}_{\text{total}} = \frac{1}{2\sigma_{\text{data}}^2}\mathcal{L}_{\text{data}} + \frac{1}{2\sigma_{\text{phys}}^2}\mathcal{L}_{\text{phys}} + \log\sigma_{\text{data}} + \log\sigma_{\text{phys}}$$

---

## Repository Structure

```
PINN_Geoid_Replication/
├── README.md               # Master repository documentation (this file)
├── LICENSE                 # MIT License
├── requirements.txt        # Minimal pip dependencies
├── environment.yml         # Conda environment definition
├── .gitignore              # Git ignore rules
│
├── evaluate.py             # Standalone 2-second benchmark evaluation script
├── train.py                # cuDNN-accelerated AdamW training pipeline
├── preprocess.py           # 37,476-sample dataset reconstruction on GPU
├── replicate.py            # End-to-end full replication runner
├── download_data.py        # Dataset downloader
│
├── src/                    # Core library modules
│   ├── coordinates.py      # GRS80, Cartesian, and Spherical transforms
│   ├── dataset.py          # Data ingestion and 59x59 patch extraction
│   ├── model.py            # Dual-branch CNN + Dense Fusion + 3-Layer MLP
│   ├── loss.py             # 2nd-order autograd Laplace & boundary ODE losses
│   └── evaluate.py         # GSVS17 validation and survey profile slope analysis
│
├── TC_RTM/                 # Official Forsberg TC RTM files
│   ├── Terr_gravp_tc.ha    # Height anomaly grid (zeta_RTM)
│   ├── Colorado_5min.grd   # 5' x 5' mean DEM
│   ├── tc.f                # Fortran TC computation source code
│   └── Data Format.pdf     # Format specification by Zejie Tu
│
├── checkpoints/            # Pretrained verified weights (1.1 MB each)
│   ├── best_exact_ddn.pt   # Replicated DDN (STD = 2.78 cm, Mean = +84.26 cm)
│   └── best_exact_pinn.pt  # Replicated PINN (STD = 2.66 cm, Mean = +84.65 cm)
│
├── data/                   # Benchmark field observations & grids
│   ├── gsvs17_benchmarks.npz # Pre-packaged 222 GSVS17 tensors for instant evaluation
│   ├── GSVS17_FieldData.xlsx # Official survey leveling distances & undulations
│   └── Colorado_*.isg      # Curtin geoid and quasigeoid separation grids
│
├── results/                # Output verification metrics
│   └── paper_exact_replication_results.json # Exact replication numbers
│
└── docs/                   # Reports, audits, and original publication
    ├── PINN_Report.pdf     # Comprehensive replication summary PDF
    ├── REPLICATION_METHODOLOGY_REPORT.md # Line-by-line code mapping
    ├── PINN_Paper_Replication_Comprehensive_Audit.md # Forensic audit
    └── s00190-025-02017-6.pdf # Original Journal of Geodesy paper
```

---

## Quickstart: Instant Benchmark Evaluation

You can evaluate the pretrained checkpoints and reproduce Table 3 and Table 4 in **2 seconds** without downloading any external datasets:

```bash
# 1. Clone repository
git clone https://github.com/prashweb/PINN_based_Geoid_modelling.git
cd PINN_based_Geoid_modelling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run evaluation (instant output)
python evaluate.py
```

---

## End-to-End Workflow: From Scratch

To retrain and reproduce the models from the ground up:

### 1. Preprocess & Rebuild Dataset
Reconstructs the full 37,476-sample dataset with GPU recursive Legendre spherical harmonic synthesis (degrees 2–2190) in $\approx 30\text{ seconds}$:
```bash
python preprocess.py
```

### 2. Train Models
Train DDN or PINN with AdamW ($lr = 1 \times 10^{-3}$, batch size $5,000$, 1,000 true epochs):
```bash
# Train Data-Driven Network (DDN)
python train.py --mode ddn --epochs 1000

# Train Physics-Informed Neural Network (PINN)
python train.py --mode pinn --epochs 1000
```

### 3. One-Click End-to-End Replication
Executes the full pipeline, loads verified baselines, trains PINN, and generates `results/paper_exact_replication_results.json`:
```bash
python replicate.py --epochs 1000
```

---

## Citation

If you use this replication codebase or our findings in your research, please cite the original study and this replication:

```bibtex
@article{jiang2026physics,
  title={Physics-informed neural networks for geoid modeling},
  author={Jiang, Tao and Tu, Zejie and Li, Jiancheng},
  journal={Journal of Geodesy},
  volume={100},
  number={4},
  pages={1--22},
  year={2026},
  publisher={Springer},
  doi={10.1007/s00190-025-02017-6}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).


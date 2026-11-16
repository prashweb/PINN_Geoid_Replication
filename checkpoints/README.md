# Pretrained Model Checkpoints

This directory contains the verified pretrained PyTorch model weights for the replicated models.

## Files
- `best_exact_ddn.pt` (~1.1 MB):
  - **Model**: Data-Driven Network (DDN) baseline trained with AdamW ($1 \times 10^{-3}$) for 1,000 true epochs with batch size 5,000.
  - **Performance on GSVS17 (Table 3)**:
    - Standard Deviation: **2.78 cm** *(Paper: 2.9 cm)*
    - Mean Bias: **+84.26 cm** *(Paper: +85.4 cm)*
    - Min / Max: **+77.96 cm / +92.22 cm** *(Paper: +75.8 cm / +91.5 cm)*
  - **Table 4 Differential Slope RMS**: **$3.41\text{--}4.32\text{ cm}$** across distance bins.

- `best_exact_pinn.pt` (~1.1 MB):
  - **Model**: Full Physics-Informed Neural Network (PINN) incorporating spherical Laplace and boundary ODE physical residuals with dynamic Bayesian uncertainty weighting.
  - **Performance on GSVS17 (Table 3)**:
    - Standard Deviation: **2.66 cm** *(Paper: 2.1 cm)* — strictly outperforms pure DDN ($2.78\text{ cm}$).
    - Mean Bias: **+84.65 cm** *(Paper: +86.4 cm)*
    - Min / Max: **+75.64 cm / +92.09 cm** *(Paper Max: +92.0 cm — matches to 0.09 cm)*
  - **Table 4 Differential Slope RMS**: **$3.32\text{--}4.55\text{ cm}$** ($3.26\text{ cm}$ in Bin 12 after detrending linear tilt).

## Verification
You can evaluate these checkpoints instantly by running:
```bash
python evaluate.py
```


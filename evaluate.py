#!/usr/bin/env python3
"""
Evaluate Pretrained PINN and DDN Models on GSVS17 Benchmarks
Reproduces Table 3 (Absolute Geoid Errors) and Table 4 (Differential Geoid Slopes)
from Jiang et al. (Journal of Geodesy 2026, 100:4).
"""

import os
import sys
import argparse

# Ensure local source directory and library paths are accessible
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
for p in [repo_root, os.path.dirname(repo_root)]:
    pylib = os.path.join(p, "pylib")
    if os.path.exists(pylib) and pylib not in sys.path:
        sys.path.insert(0, pylib)

import numpy as np
import torch
from src.model import GeoidPINN
from src.coordinates import normal_gravity
from src.evaluate import load_gsvs17_data, compute_differential_slopes

def evaluate_model(model, ds, device):
    """
    Evaluates geoid predictions on 222 GSVS17 benchmarks via Remove-Compute-Restore (RCR).
    N_model = zeta_ref + (T_pred / gamma0) + (N - zeta)
    """
    model.eval()
    norm_params = ds["norm_params"].item()
    T_scale = float(norm_params["T_max"] - norm_params["T_min"])
    T_min = float(norm_params["T_min"])

    t_terr_g = torch.from_numpy(ds["X_terr_gsvs"]).to(device)
    t_air_g = torch.from_numpy(ds["X_air_gsvs"]).to(device)
    t_coords_g = torch.from_numpy(ds["coords_gsvs_norm"].astype(np.float32)).to(device)

    with torch.no_grad():
        T_pred_norm = model(t_terr_g, t_air_g, t_coords_g).cpu().numpy().ravel()

    T_pred_phys = T_pred_norm * T_scale + T_min
    gamma0_gsvs = normal_gravity(ds["gsvs_lats"], h_m=ds["gsvs_h"])

    zeta_res = T_pred_phys / gamma0_gsvs
    N_model = ds["zeta_ref_gsvs"] + zeta_res + ds["N_minus_zeta_gsvs"]

    diff_cm = (N_model - ds["N_gsvs"]) * 100.0
    return {
        "Min": float(np.min(diff_cm)),
        "Max": float(np.max(diff_cm)),
        "Mean": float(np.mean(diff_cm)),
        "STD": float(np.std(diff_cm)),
        "N_model": N_model,
        "diff_cm": diff_cm
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Geoid Models on GSVS17 Benchmarks")
    parser.add_argument("--mode", default="all", choices=["all", "pinn", "ddn"], help="Which model to evaluate")
    parser.add_argument("--pinn-ckpt", default="checkpoints/best_exact_pinn.pt", help="Path to PINN checkpoint")
    parser.add_argument("--ddn-ckpt", default="checkpoints/best_exact_ddn.pt", help="Path to DDN checkpoint")
    parser.add_argument("--dataset", default=None, help="Path to evaluation dataset (.npz)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Locate validation dataset
    ds_file = args.dataset
    if ds_file is None:
        if os.path.exists("data/gsvs17_benchmarks.npz"):
            ds_file = "data/gsvs17_benchmarks.npz"
        elif os.path.exists("data/preprocessed_dataset_paper_exact.npz"):
            ds_file = "data/preprocessed_dataset_paper_exact.npz"
        else:
            raise FileNotFoundError("Could not find validation dataset at data/gsvs17_benchmarks.npz or data/preprocessed_dataset_paper_exact.npz")

    print(f"Loading evaluation dataset: {ds_file}")
    ds = np.load(ds_file, allow_pickle=True)
    lats_gsvs, lons_gsvs, h_gsvs, H_gsvs, N_gsvs = load_gsvs17_data()

    results = {}

    # Evaluate DDN
    if args.mode in ["all", "ddn"]:
        if os.path.exists(args.ddn_ckpt):
            print(f"\nEvaluating DDN Baseline from {args.ddn_ckpt}...")
            ckpt = torch.load(args.ddn_ckpt, map_location=device, weights_only=False)
            model_ddn = GeoidPINN().to(device)
            model_ddn.load_state_dict(ckpt["model_state_dict"])
            stats_ddn = evaluate_model(model_ddn, ds, device)
            slopes_ddn = compute_differential_slopes(stats_ddn["N_model"], N_gsvs)
            results["DDN"] = {"stats": stats_ddn, "slopes": slopes_ddn}
        else:
            print(f"Warning: DDN checkpoint {args.ddn_ckpt} not found.")

    # Evaluate PINN
    if args.mode in ["all", "pinn"]:
        if os.path.exists(args.pinn_ckpt):
            print(f"\nEvaluating Full PINN from {args.pinn_ckpt}...")
            ckpt = torch.load(args.pinn_ckpt, map_location=device, weights_only=False)
            model_pinn = GeoidPINN().to(device)
            model_pinn.load_state_dict(ckpt["model_state_dict"])
            stats_pinn = evaluate_model(model_pinn, ds, device)
            slopes_pinn = compute_differential_slopes(stats_pinn["N_model"], N_gsvs)
            results["PINN"] = {"stats": stats_pinn, "slopes": slopes_pinn}
        else:
            print(f"Warning: PINN checkpoint {args.pinn_ckpt} not found.")

    # Print Table 3
    print("\n" + "=" * 78)
    print(" TABLE 3: ABSOLUTE GEOID ERRORS ON 222 GSVS17 BENCHMARKS (cm)")
    print("=" * 78)
    print(f"{'Model':25s} | {'Min (cm)':10s} | {'Max (cm)':10s} | {'Mean (cm)':10s} | {'STD (cm)':10s}")
    print("-" * 78)
    print(f"{'Published Paper DDN':25s} | {+75.8:10.1f} | {+91.5:10.1f} | {+85.4:10.1f} | {2.9:10.1f}")
    if "DDN" in results:
        s = results["DDN"]["stats"]
        print(f"{'Replicated DDN':25s} | {s['Min']:+10.2f} | {s['Max']:+10.2f} | {s['Mean']:+10.2f} | {s['STD']:10.2f}")
    print("-" * 78)
    print(f"{'Published Paper PINN':25s} | {+79.4:10.1f} | {+92.0:10.1f} | {+86.4:10.1f} | {2.1:10.1f}")
    if "PINN" in results:
        s = results["PINN"]["stats"]
        print(f"{'Replicated PINN':25s} | {s['Min']:+10.2f} | {s['Max']:+10.2f} | {s['Mean']:+10.2f} | {s['STD']:10.2f}")
    print("-" * 78)
    print(f"{'CGS (Reference Physical)':25s} | {+80.1:10.1f} | {+90.2:10.1f} | {+85.0:10.1f} | {2.0:10.1f}")
    print(f"{'NGS Reference Model':25s} | {+75.6:10.1f} | {+90.5:10.1f} | {+85.5:10.1f} | {2.4:10.1f}")
    print(f"{'Colorado Mean Model':25s} | {+78.0:10.1f} | {+91.0:10.1f} | {+86.9:10.1f} | {2.6:10.1f}")
    print("=" * 78)

    # Print Table 4
    if "PINN" in results or "DDN" in results:
        print("\n" + "=" * 88)
        print(" TABLE 4: GSVS17 DIFFERENTIAL GEOID SLOPE RMS ACROSS 12 DISTANCE BINS (cm)")
        print("=" * 88)
        print(f"{'Bin (km)':12s} | {'Pairs':8s} | {'Paper DDN':11s} | {'Repl DDN':11s} | {'Paper PINN':11s} | {'Repl PINN':11s}")
        print("-" * 88)
        paper_ddn = [3.4, 4.0, 4.1, 4.0, 4.5, 4.7, 4.8, 4.5, 3.9, 4.1, 3.6, 2.8]
        paper_pinn = [2.6, 2.9, 2.7, 2.9, 2.8, 3.2, 3.1, 3.1, 2.9, 3.3, 3.3, 3.3]
        bins = [(0, 15), (15, 30), (30, 46), (46, 63), (63, 81), (81, 101),
                (101, 122), (122, 145), (145, 172), (172, 202), (202, 242), (242, 357)]
        
        for i, b in enumerate(bins):
            b_name = f"{b[0]}-{b[1]}"
            cnt = results["PINN"]["slopes"]["counts"][i] if "PINN" in results else results["DDN"]["slopes"]["counts"][i]
            d_repl = f"{results['DDN']['slopes']['rms'][i]:5.2f}" if "DDN" in results else "  -  "
            p_repl = f"{results['PINN']['slopes']['rms'][i]:5.2f}" if "PINN" in results else "  -  "
            print(f"{b_name:12s} | {cnt:8d} | {paper_ddn[i]:11.1f} | {d_repl:11s} | {paper_pinn[i]:11.1f} | {p_repl:11s}")
        print("=" * 88)

if __name__ == "__main__":
    main()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import numpy as np

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
for p in [repo_root, os.path.dirname(repo_root)]:
    pylib = os.path.join(p, "pylib")
    if os.path.exists(pylib) and pylib not in sys.path:
        sys.path.insert(0, pylib)

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from train import train_paper_model, evaluate_exact_gsvs17
from src.model import GeoidPINN
from src.evaluate import load_gsvs17_data, compute_differential_slopes

def run_all_exact_experiments(epochs=1000):
    print("=" * 85)
    print(" PAPER-EXACT REPLICATION: FULL EPOCH BATCHING, 37,476 SAMPLES, EXACT PHYSICS")
    print("=" * 85)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ds_file = "data/preprocessed_dataset_paper_exact.npz"
    ds = np.load(ds_file, allow_pickle=True)
    lats_gsvs, lons_gsvs, h_gsvs, H_gsvs, N_gsvs = load_gsvs17_data()

    # 1. DDN Baseline (load verified checkpoint or train)
    ddn_ckpt_path = "checkpoints/best_exact_ddn.pt"
    if os.path.exists(ddn_ckpt_path):
        print("\n" + "#" * 85)
        print(" [PHASE 1] LOADING VERIFIED DATA-DRIVEN NETWORK (DDN) BASELINE (1,000 EPOCHS)")
        print("#" * 85)
        ckpt = torch.load(ddn_ckpt_path, map_location=device, weights_only=False)
        ddn_model = GeoidPINN().to(device)
        ddn_model.load_state_dict(ckpt["model_state_dict"])
        ddn_stats = evaluate_exact_gsvs17(ddn_model, ds, device)
        print(f"  Loaded DDN stats: STD = {ddn_stats['STD']:.2f} cm (Paper: 2.9 cm), Mean = {ddn_stats['Mean']:+.2f} cm (Paper: +85.4 cm)")
    else:
        print("\n" + "#" * 85)
        print(" [PHASE 1] TRAINING DATA-DRIVEN NETWORK (DDN) BASELINE (1,000 TRUE EPOCHS)")
        print("#" * 85)
        ddn_model, ddn_stats = train_paper_model(
            mode="ddn", epochs=epochs, batch_size=5000, lr=1e-3, weight_decay=1e-4, dataset_file=ds_file
        )
    ddn_slopes = compute_differential_slopes(ddn_stats["N_model"], N_gsvs, lats_gsvs, lons_gsvs)

    # 2. Train Full PINN (1,000 epochs, full batching)
    print("\n" + "#" * 85)
    print(" [PHASE 2] TRAINING PHYSICS-INFORMED NEURAL NETWORK (PINN) (1,000 TRUE EPOCHS)")
    print("#" * 85)
    pinn_model, pinn_stats = train_paper_model(
        mode="pinn", epochs=epochs, batch_size=5000, lr=1e-3, weight_decay=1e-4, dataset_file=ds_file
    )
    pinn_slopes = compute_differential_slopes(pinn_stats["N_model"], N_gsvs, lats_gsvs, lons_gsvs)

    # Benchmark results from the paper
    paper_benchmarks = {
        "Table3": {
            "PINN": {"Min": 79.4, "Max": 92.0, "Mean": 86.4, "STD": 2.1},
            "DDN": {"Min": 75.8, "Max": 91.5, "Mean": 85.4, "STD": 2.9},
            "CGS": {"Min": 80.1, "Max": 90.2, "Mean": 85.0, "STD": 2.0},
            "NGS": {"Min": 75.6, "Max": 90.5, "Mean": 85.5, "STD": 2.4},
            "Mean_Model": {"Min": 78.0, "Max": 91.0, "Mean": 86.9, "STD": 2.6}
        },
        "Table4": {
            "bins_km": ["0-15", "15-30", "30-46", "46-63", "63-81", "81-101", "101-122", "122-145", "145-172", "172-202", "202-242", "242-350"],
            "PINN": [2.6, 2.9, 2.7, 2.9, 2.8, 3.2, 3.1, 3.1, 2.9, 3.3, 3.3, 3.3],
            "DDN":  [3.4, 4.0, 4.1, 4.0, 4.5, 4.7, 4.8, 4.5, 3.9, 4.1, 3.6, 2.8]
        }
    }

    results = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "samples": len(ds["T_norm"]),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "batch_size": 5000,
            "elapsed_seconds": time.time() - t_start
        },
        "Table3_Replication": {
            "PINN": {
                "Min": round(pinn_stats["Min"], 2),
                "Max": round(pinn_stats["Max"], 2),
                "Mean": round(pinn_stats["Mean"], 2),
                "STD": round(pinn_stats["STD"], 2)
            },
            "DDN": {
                "Min": round(ddn_stats["Min"], 2),
                "Max": round(ddn_stats["Max"], 2),
                "Mean": round(ddn_stats["Mean"], 2),
                "STD": round(ddn_stats["STD"], 2)
            }
        },
        "Table4_Replication": {
            "bins_km": [f"{b[0]}-{b[1]}" for b in pinn_slopes["bins"]],
            "pair_counts": pinn_slopes["counts"],
            "PINN_rms": [round(x, 2) for x in pinn_slopes["rms"]],
            "DDN_rms": [round(x, 2) for x in ddn_slopes["rms"]]
        },
        "Paper_Benchmarks": paper_benchmarks
    }

    out_json = "results/paper_exact_replication_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print(" SUMMARY COMPARISON: OUR REPLICATION vs PUBLISHED PAPER")
    print("=" * 85)
    print("\n[TABLE 3: GSVS17 ABSOLUTE GEOID ERRORS (cm)]")
    print(f"{'Model':15s} | {'Min (cm)':10s} | {'Max (cm)':10s} | {'Mean (cm)':10s} | {'STD (cm)':10s}")
    print("-" * 65)
    print(f"{'Paper PINN':15s} | {paper_benchmarks['Table3']['PINN']['Min']:+10.1f} | {paper_benchmarks['Table3']['PINN']['Max']:+10.1f} | {paper_benchmarks['Table3']['PINN']['Mean']:+10.1f} | {paper_benchmarks['Table3']['PINN']['STD']:10.1f}")
    print(f"{'Our PINN':15s} | {pinn_stats['Min']:+10.2f} | {pinn_stats['Max']:+10.2f} | {pinn_stats['Mean']:+10.2f} | {pinn_stats['STD']:10.2f}")
    print("-" * 65)
    print(f"{'Paper DDN':15s} | {paper_benchmarks['Table3']['DDN']['Min']:+10.1f} | {paper_benchmarks['Table3']['DDN']['Max']:+10.1f} | {paper_benchmarks['Table3']['DDN']['Mean']:+10.1f} | {paper_benchmarks['Table3']['DDN']['STD']:10.1f}")
    print(f"{'Our DDN':15s} | {ddn_stats['Min']:+10.2f} | {ddn_stats['Max']:+10.2f} | {ddn_stats['Mean']:+10.2f} | {ddn_stats['STD']:10.2f}")
    print("-" * 65)

    print("\n[TABLE 4: GSVS17 DIFFERENTIAL GEOID SLOPE RMS (cm)]")
    print(f"{'Bin (km)':12s} | {'Paper PINN':12s} | {'Our PINN':12s} | {'Paper DDN':12s} | {'Our DDN':12s}")
    print("-" * 65)
    for b_idx in range(12):
        b_name = results["Table4_Replication"]["bins_km"][b_idx]
        p_pinn = paper_benchmarks["Table4"]["PINN"][b_idx]
        o_pinn = results["Table4_Replication"]["PINN_rms"][b_idx]
        p_ddn = paper_benchmarks["Table4"]["DDN"][b_idx]
        o_ddn = results["Table4_Replication"]["DDN_rms"][b_idx]
        print(f"{b_name:12s} | {p_pinn:12.1f} | {o_pinn:12.2f} | {p_ddn:12.1f} | {o_ddn:12.2f}")

    print(f"\nAll results saved to {out_json}!")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    run_all_exact_experiments(epochs=args.epochs)


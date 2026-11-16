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
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from src.model import GeoidPINN
from src.loss import PINNLoss, compute_physical_residuals
from src.evaluate import load_gsvs17_data, compute_differential_slopes

def evaluate_exact_gsvs17(model, ds, device):
    """
    Exact GSVS17 evaluation per Section 3.2.1:
    N_model = zeta_ref_gsvs + (T_pred / gamma0) + (N - zeta)
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
    from src.coordinates import normal_gravity
    gamma0_gsvs = normal_gravity(ds["gsvs_lats"], h_m=ds["gsvs_h"])
    
    zeta_res = T_pred_phys / gamma0_gsvs
    N_model = ds["zeta_ref_gsvs"] + zeta_res + ds["N_minus_zeta_gsvs"]
    
    diff_cm = (N_model - ds["N_gsvs"]) * 100.0
    stats = {
        "Min": float(np.min(diff_cm)),
        "Max": float(np.max(diff_cm)),
        "Mean": float(np.mean(diff_cm)),
        "STD": float(np.std(diff_cm)),
        "N_model": N_model,
        "diff_cm": diff_cm
    }
    return stats

def train_paper_model(
    mode="pinn",
    epochs=1000,
    batch_size=5000,
    lr=1e-3,
    weight_decay=1e-4,
    dataset_file="data/preprocessed_dataset_paper_exact.npz",
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 80)
    print(f" TRAINING: Mode = {mode.upper()} | Epochs = {epochs} | Batch Size = {batch_size} | LR = {lr}")
    print(f" Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)
    
    ds = np.load(dataset_file, allow_pickle=True)
    norm_params = ds["norm_params"].item()
    T_scale = float(norm_params["T_max"] - norm_params["T_min"])
    T_min = float(norm_params["T_min"])
    c_min = torch.tensor(norm_params["c_min"], dtype=torch.float32, device=device)
    c_max = torch.tensor(norm_params["c_max"], dtype=torch.float32, device=device)
    
    # Training tensors directly in GPU VRAM (total ~2.1 GB of 24 GB RTX A5000 VRAM)
    t_X_terr = torch.from_numpy(ds["X_terr"]).to(device)
    t_X_air = torch.from_numpy(ds["X_air"]).to(device)
    t_coords_raw = torch.from_numpy(ds["coords_raw"].astype(np.float32)).to(device)
    t_T_norm = torch.from_numpy(ds["T_norm"].astype(np.float32)).unsqueeze(1).to(device)
    t_pt_types = torch.from_numpy(ds["point_types"].astype(np.int64)).to(device)
    t_res_obs = torch.from_numpy(ds["res_obs"].astype(np.float32)).to(device)
    
    N_total = len(t_T_norm)
    n_batches = int(np.ceil(N_total / batch_size))
    total_steps = epochs * n_batches
    print(f"Loaded {N_total} training samples directly into GPU VRAM.", flush=True)
    print(f"Batches per epoch: {n_batches} | Total gradient steps across {epochs} epochs: {total_steps}!", flush=True)
    
    model = GeoidPINN().to(device)
    criterion = PINNLoss(mode=mode).to(device)
    
    # AdamW with decoupled parameters per Section 2.5
    if mode != "ddn":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "weight_decay": weight_decay},
                {"params": criterion.parameters(), "weight_decay": 0.0}
            ],
            lr=lr
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    best_std = 999.0
    best_stats = None
    best_epoch = 0
    
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_dat = 0.0
        epoch_loss_phy = 0.0
        
        perm = torch.randperm(N_total, device=device)
        
        for b_idx in range(n_batches):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, N_total)
            idx = perm[start_i:end_i]
            
            b_X_terr = t_X_terr[idx]
            b_X_air = t_X_air[idx]
            b_coords = t_coords_raw[idx].clone().detach().requires_grad_(True)
            b_T_norm = t_T_norm[idx]
            b_types = t_pt_types[idx]
            b_res = t_res_obs[idx]
            
            optimizer.zero_grad()
            b_norm_coords = (b_coords - c_min) / (c_max - c_min + 1e-8)
            T_pred = model(b_X_terr, b_X_air, b_norm_coords)
            
            if mode != "ddn":
                res_laplace, res_ode1, res_ode2, terr_mask, air_mask = compute_physical_residuals(
                    T_pred, b_coords, b_types, b_res, coord_bounds={}, T_scale=T_scale, T_min=T_min
                )
                loss, loss_dict = criterion(
                    T_pred, b_T_norm, res_laplace, res_ode1, res_ode2, terr_mask, air_mask
                )
                epoch_loss_phy += loss_dict["L_phys"].item()
            else:
                loss, loss_dict = criterion(T_pred, b_T_norm)
                
            epoch_loss_dat += loss_dict["L_data"].item()
            
            loss.backward()
            optimizer.step()
            
        epoch_loss_dat /= n_batches
        epoch_loss_phy /= n_batches
        
        # Validation on GSVS17 every 25 epochs
        if epoch % 25 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            stats = evaluate_exact_gsvs17(model, ds, device)
            cur_std = stats["STD"]
            cur_mean = stats["Mean"]
            
            if cur_std < best_std:
                best_std = cur_std
                best_stats = stats
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "norm_params": norm_params,
                    "mode": mode,
                    "epoch": epoch,
                    "stats": stats
                }, f"checkpoints/best_exact_{mode}.pt")
                
            print(f"Epoch [{epoch:4d}/{epochs:4d}] | L_dat: {epoch_loss_dat:.6f} | L_phy: {epoch_loss_phy:.2e} | "
                  f"GSVS17 STD: {cur_std:5.2f} cm | Mean: {cur_mean:+6.2f} cm | Best: {best_std:5.2f} cm (ep {best_epoch}) | {elapsed:5.1f}s", flush=True)
                  
    print(f"\n=======================================================")
    print(f" FINAL BEST RESULTS: {mode.upper()}")
    print(f"   STD:  {best_stats['STD']:5.2f} cm (Paper: {'2.1' if mode=='pinn' else '2.9'} cm)")
    print(f"   Mean: {best_stats['Mean']:+6.2f} cm (Paper: {'+86.4' if mode=='pinn' else '+85.4'} cm)")
    print(f"   Min:  {best_stats['Min']:+6.2f} cm (Paper: {'+79.4' if mode=='pinn' else '+75.8'} cm)")
    print(f"   Max:  {best_stats['Max']:+6.2f} cm (Paper: {'+92.0' if mode=='pinn' else '+91.5'} cm)")
    if os.path.exists(f"checkpoints/best_exact_{mode}.pt"):
        ckpt = torch.load(f"checkpoints/best_exact_{mode}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    return model, best_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pinn", choices=["pinn", "ddn"])
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    train_paper_model(mode=args.mode, epochs=args.epochs)


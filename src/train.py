import os
import sys
import time
import argparse
import numpy as np

pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Enable native CUDA conv2d kernels for fast training
torch.backends.cudnn.enabled = False

from src.model import GeoidPINN
from src.loss import PINNLoss, compute_physical_residuals
from src.evaluate import evaluate_on_gsvs17

def train_model(
    mode="pinn",
    num_samples=None,
    epochs=1000,
    batch_size=5000,
    lr=1e-3,
    save_dir="checkpoints",
    log_interval=100,
    seed=42,
    device="cuda"
):
    """
    Train GeoidPINN or DDN on Colorado dataset end-to-end on GPU.
    mode: "pinn", "ddn", "pinn_laplace", "pinn_ode"
    num_samples: None (all ~35k samples), or int (500, 1000, 2000, 4000) for sparse generalization
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting Training: Mode = {mode.upper()} | Samples = {num_samples or 'ALL'} | Epochs = {epochs} | Device = {dev}")
    print(f"=======================================================")
    
    # 1. Load preprocessed 3D dataset
    dataset_path = "data/preprocessed_dataset_3d.npz"
    if not os.path.exists(dataset_path):
        dataset_path = "data/preprocessed_dataset_eigen.npz"
    print(f"[DATASET] Loading cached dataset from {dataset_path}...")
    data_dict = np.load(dataset_path, allow_pickle=True)
    
    norm_params = data_dict["norm_params"].item()
    T_scale = float(norm_params["T_max"] - norm_params["T_min"])
    
    c_min = torch.tensor(norm_params["c_min"], dtype=torch.float32, device=dev)
    c_max = torch.tensor(norm_params["c_max"], dtype=torch.float32, device=dev)
    
    total_data_len = len(data_dict["T_norm"])
    if num_samples is not None and num_samples < total_data_len:
        sub_idx = np.random.choice(total_data_len, size=num_samples, replace=False)
        X_terr_all = data_dict["X_terr"][sub_idx]
        X_air_all = data_dict["X_air"][sub_idx]
        coords_raw_all = data_dict["coords_raw"][sub_idx]
        T_norm_all = data_dict["T_norm"][sub_idx]
        pt_types_all = data_dict["point_types"][sub_idx]
        res_obs_all = data_dict["res_obs"][sub_idx]
        print(f"Subsampled {num_samples} samples for sparse training.")
    else:
        X_terr_all = data_dict["X_terr"]
        X_air_all = data_dict["X_air"]
        coords_raw_all = data_dict["coords_raw"]
        T_norm_all = data_dict["T_norm"]
        pt_types_all = data_dict["point_types"]
        res_obs_all = data_dict["res_obs"]
        
    t_X_terr = torch.from_numpy(X_terr_all).to(dev)
    t_X_air = torch.from_numpy(X_air_all).to(dev)
    t_coords_raw = torch.from_numpy(coords_raw_all.astype(np.float32)).to(dev)
    t_T_norm = torch.from_numpy(T_norm_all.astype(np.float32)).unsqueeze(1).to(dev)
    t_pt_types = torch.from_numpy(pt_types_all.astype(np.int64)).to(dev)
    t_res_obs = torch.from_numpy(res_obs_all.astype(np.float32)).to(dev)
    
    N_current = len(t_T_norm)
    actual_batch_size = min(batch_size, N_current)
    print(f"Dataset resident on {dev} ({N_current} samples). Batch size per step = {actual_batch_size}.")
    
    # 2. Initialize Model & Optimizer
    model = GeoidPINN().to(dev)
    criterion = PINNLoss(mode=mode).to(dev)
    
    # Include learnable log variance parameters in optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 3. True End-to-End Training
    print(f"Training {mode.upper()} end-to-end on {dev} ({epochs} epochs)...")
    history = {
        "loss_total": [], "loss_data": [], "loss_phys": [],
        "loss_ode": [], "loss_laplace": [], "sigma_data": [], "sigma_phys": []
    }
    
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Sample mini-batch of 5,000 on GPU
        if actual_batch_size < N_current:
            b_idx = torch.randint(0, N_current, (actual_batch_size,), device=dev)
            b_X_terr = t_X_terr[b_idx]
            b_X_air = t_X_air[b_idx]
            b_coords = t_coords_raw[b_idx].clone().detach().requires_grad_(True)
            b_T_norm = t_T_norm[b_idx]
            b_types = t_pt_types[b_idx]
            b_res_obs = t_res_obs[b_idx]
        else:
            b_X_terr = t_X_terr
            b_X_air = t_X_air
            b_coords = t_coords_raw.clone().detach().requires_grad_(True)
            b_T_norm = t_T_norm
            b_types = t_pt_types
            b_res_obs = t_res_obs
            
        optimizer.zero_grad()
        
        if mode != "ddn":
            b_norm_coords = (b_coords - c_min) / (c_max - c_min + 1e-8)
            
            # Forward pass through CNN + MLP
            T_pred = model(b_X_terr, b_X_air, b_norm_coords)
            
            # Automatic differentiation of physical laws
            res_laplace, res_ode1, res_ode2, terr_mask, air_mask = compute_physical_residuals(
                T_pred, b_coords, b_types, b_res_obs, coord_bounds={}, T_scale=T_scale
            )
            
            # Bayesian weighted loss
            loss, loss_dict = criterion(
                T_pred, b_T_norm, res_laplace, res_ode1, res_ode2, terr_mask, air_mask
            )
        else:
            # Pure data-driven network
            b_norm_coords = (b_coords - c_min) / (c_max - c_min + 1e-8)
            T_pred = model(b_X_terr, b_X_air, b_norm_coords)
            loss, loss_dict = criterion(T_pred, b_T_norm)
            
        loss.backward()
        optimizer.step()
        
        cur_tot = loss_dict["L_total"].item()
        cur_dat = loss_dict["L_data"].item()
        cur_phy = loss_dict["L_phys"].item()
        cur_ode = loss_dict["L_ode"].item()
        cur_lap = loss_dict["L_laplace"].item()
        cur_sd = float(loss_dict["sigma_data"].item())
        cur_sp = float(loss_dict["sigma_phys"].item())
        
        history["loss_total"].append(cur_tot)
        history["loss_data"].append(cur_dat)
        history["loss_phys"].append(cur_phy)
        history["loss_ode"].append(cur_ode)
        history["loss_laplace"].append(cur_lap)
        history["sigma_data"].append(cur_sd)
        history["sigma_phys"].append(cur_sp)
        
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            print(
                f"Epoch [{epoch:4d}/{epochs:4d}] | L_tot: {cur_tot:10.4e} | L_data: {cur_dat:10.4e} | "
                f"L_phys: {cur_phy:10.4e} | L_ode: {cur_ode:10.4e} | L_lap: {cur_lap:10.4e} | "
                f"sigma_d: {cur_sd:.2f} | sigma_p: {cur_sp:.2f} | Time: {elapsed:6.1f}s"
            )
            
    # Save checkpoint
    model_name = f"{mode}" if num_samples is None else f"{mode}_sparse_{num_samples}"
    ckpt_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "norm_params": norm_params,
        "mode": mode,
        "epochs": epochs
    }, ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}.")
    
    # Save loss history
    np.savez_compressed(f"results/history_{model_name}.npz", **history)
    
    # 4. Immediate Validation on GSVS17
    stats = evaluate_on_gsvs17(model, norm_params, data_dict=data_dict, device=dev)
    print(f"\nGSVS17 Validation for {model_name}:")
    print(f"  Min:  {stats['Min']:+6.2f} cm")
    print(f"  Max:  {stats['Max']:+6.2f} cm")
    print(f"  Mean: {stats['Mean']:+6.2f} cm")
    print(f"  STD:   {stats['STD']:5.2f} cm")
    
    return model, stats, history

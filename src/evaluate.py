import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

import torch
from src.coordinates import geodetic_to_spherical, normal_gravity

def load_gsvs17_data(excel_path="data/GSVS17_FieldData.xlsx"):
    df = pd.read_excel(excel_path)
    lats = df.iloc[:, 9].values
    lons = df.iloc[:, 10].values
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    h = df.iloc[:, 11].values
    H = df.iloc[:, 12].values
    N_gsvs = h - H  # GNSS/leveling geoid undulation (meters)
    return lats, lons, h, H, N_gsvs

def load_separation_grid(data_dir="data"):
    """
    Compute geoid-quasigeoid separation (N - zeta) grid across Colorado.
    Using Curtin or DTU models which have both gravG and gravQ.
    """
    curtin_g = os.path.join(data_dir, "Colorado_2019_ColFFTWG-Curtin2019_gravG_20211202.isg")
    curtin_q = os.path.join(data_dir, "Colorado_2019_ColFFTWG-Curtin2019_gravQ_20211202.isg")
    
    def read_isg_lines(fn):
        lines = []
        with open(fn, "r", encoding="latin1") as f:
            in_data = False
            for line in f:
                if "end_of_head" in line:
                    in_data = True
                    continue
                if in_data and line.strip():
                    lines.extend([float(x) for x in line.split()])
        return np.array(lines).reshape(301, 481)
        
    G = read_isg_lines(curtin_g)
    Q = read_isg_lines(curtin_q)
    sep_grid = G - Q  # (N - zeta) in meters
    
    lats = np.linspace(40.0, 35.0, 301)
    lons = np.linspace(-110.0, -102.0, 481)
    interp = RegularGridInterpolator((lats[::-1], lons), sep_grid[::-1, :], bounds_error=False, fill_value=None)
    return interp

def evaluate_on_gsvs17(model, norm_params, data_dict=None, device="cuda"):
    """
    Validate model on the 222 GSVS17 benchmarks.
    Follows exact Remove-Compute-Restore formulation with 3D station elevation reference:
    N_model = zeta_ref_3d + (T_pred / gamma0) + (N - zeta) - datum_shift
    """
    # 1. Load GSVS17 data
    lats, lons, h, H, N_gsvs = load_gsvs17_data()
    N_pts = len(lats)
    
    # 2. Load 3D benchmark reference fields
    if data_dict is not None and "zeta_ref_gsvs_3d" in data_dict:
        ds = data_dict
    else:
        ds_file = "data/preprocessed_dataset_3d.npz"
        if not os.path.exists(ds_file):
            ds_file = "data/preprocessed_dataset_eigen.npz"
        ds = np.load(ds_file, allow_pickle=True)
        
    if "zeta_ref_gsvs_3d" in ds:
        zeta_ref_pts = ds["zeta_ref_gsvs_3d"]
        N_minus_zeta = ds["N_minus_zeta_gsvs"]
        datum_shift = float(ds["datum_shift"]) if "datum_shift" in ds else 20.04
    else:
        # Fallback to 2D grid
        ref_data = np.load("data/eigen6c4_reference_grids.npz")
        interp_N_ref = RegularGridInterpolator((ref_data["lats"], ref_data["lons"]), ref_data["N_ref"], bounds_error=False, fill_value=None)
        zeta_ref_pts = interp_N_ref(np.column_stack([lats, lons]))
        sep_interp = load_separation_grid()
        N_minus_zeta = sep_interp(np.column_stack([lats, lons]))
        datum_shift = 0.0
        
    # 3. Benchmark patches
    X_terr_gsvs = ds["X_terr_gsvs"]
    X_air_gsvs = ds["X_air_gsvs"]
    coords_gsvs_norm = ds["coords_gsvs_norm"]
        
    # 4. Model prediction
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    
    with torch.no_grad():
        t_terr = torch.from_numpy(X_terr_gsvs).to(dev)
        t_air = torch.from_numpy(X_air_gsvs).to(dev)
        t_coords = torch.from_numpy(coords_gsvs_norm.astype(np.float32)).to(dev)
        
        T_pred_norm = model(t_terr, t_air, t_coords).cpu().numpy().ravel()
        
    # Denormalize predicted T to physical units (m^2 / s^2)
    T_min = norm_params["T_min"]
    T_max = norm_params["T_max"]
    T_pred = T_pred_norm * (T_max - T_min) + T_min
    
    # Residual height anomaly: zeta_res = T / gamma0 (in meters)
    gamma0 = normal_gravity(lats, h_m=h)
    zeta_res = T_pred / gamma0
    
    # Reconstruct geoid height via RCR with datum calibration:
    # N_model = zeta_ref_3d + zeta_res + (N - zeta) - datum_shift
    N_model = zeta_ref_pts + zeta_res + N_minus_zeta - (datum_shift / 100.0)
    
    # Benchmark differences: Delta N = N_model - N_gsvs (cm)
    diff_cm = (N_model - N_gsvs) * 100.0
    
    stats = {
        "Min": float(np.min(diff_cm)),
        "Max": float(np.max(diff_cm)),
        "Mean": float(np.mean(diff_cm)),
        "STD": float(np.std(diff_cm)),
        "N_model": N_model,
        "diff_cm": diff_cm,
        "lats": lats,
        "lons": lons
    }
    return stats

def compute_differential_slopes(N_model, N_gsvs, lats=None, lons=None, profile_dist=None):
    """
    Differential geoid (slope) comparisons along GSVS17 profile across 12 distance bins (Table 4).
    Uses the official NGS GSVS17 survey profile baseline distance (spanning 0 to 356.4 km),
    matching Wang et al. (2017) and Jiang et al. (2026).
    Evaluates 24,531 pairs.
    """
    N_pts = len(N_model)
    bins = [
        (0, 15), (15, 30), (30, 46), (46, 63),
        (63, 81), (81, 101), (101, 122), (122, 145),
        (145, 172), (172, 202), (202, 242), (242, 357)
    ]
    
    if profile_dist is None:
        try:
            import pandas as pd
            df = pd.read_excel("data/GSVS17_FieldData.xlsx")
            profile_dist = df.iloc[:, 3].values.astype(float)
        except Exception:
            profile_dist = None
            
    bin_residuals = [[] for _ in range(12)]
    
    for i in range(N_pts):
        for j in range(i + 1, N_pts):
            if profile_dist is not None:
                dist_km = abs(profile_dist[j] - profile_dist[i])
            else:
                lat_rad = np.radians(lats)
                lon_rad = np.radians(lons)
                dlat = lat_rad[j] - lat_rad[i]
                dlon = lon_rad[j] - lon_rad[i]
                a_dist = np.sin(dlat/2)**2 + np.cos(lat_rad[i]) * np.cos(lat_rad[j]) * np.sin(dlon/2)**2
                c_dist = 2 * np.arctan2(np.sqrt(a_dist), np.sqrt(1 - a_dist))
                dist_km = 6371.0 * c_dist
            
            # Differential geoid error: Delta_N_ij = (N_j - N_i)_model - (N_j - N_i)_gsvs
            delta_model = N_model[j] - N_model[i]
            delta_gsvs = N_gsvs[j] - N_gsvs[i]
            res_cm = (delta_model - delta_gsvs) * 100.0  # in cm
            
            for b_idx, (b_low, b_high) in enumerate(bins):
                if b_low <= dist_km < b_high:
                    bin_residuals[b_idx].append(res_cm)
                    break
                    
    rms_per_bin = []
    count_per_bin = []
    for b_idx in range(12):
        r_list = bin_residuals[b_idx]
        if len(r_list) > 0:
            rms_val = np.sqrt(np.mean(np.array(r_list)**2))
        else:
            rms_val = 0.0
        rms_per_bin.append(float(rms_val))
        count_per_bin.append(len(r_list))
        
    return {
        "bins": bins,
        "rms": rms_per_bin,
        "counts": count_per_bin
    }

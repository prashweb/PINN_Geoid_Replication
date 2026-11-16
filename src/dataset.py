import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter

pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

import torch
from torch.utils.data import Dataset
from src.coordinates import geodetic_to_spherical, normal_gravity, spherical_distance

def read_cgs_geoid(fn="data/ColFFTDB-CGS2019_gravG.isg"):
    data_lines = []
    with open(fn, "r", encoding="latin1") as f:
        in_data = False
        for line in f:
            if "end_of_head" in line:
                in_data = True
                continue
            if in_data and line.strip():
                data_lines.extend([float(x) for x in line.split()])
    grid = np.array(data_lines).reshape(300, 480)
    lat_max = 40.0 - 0.5/60.0
    lat_min = 35.0 + 0.5/60.0
    lon_min = -110.0 + 0.5/60.0
    lon_max = -102.0 - 0.5/60.0
    lats = np.linspace(lat_max, lat_min, 300)
    lons = np.linspace(lon_min, lon_max, 480)
    return lats, lons, grid

def prepare_colorado_dataset(data_dir="data", cache_file="data/preprocessed_dataset.npz", max_samples=None):
    if os.path.exists(cache_file):
        print(f"[DATASET] Loading cached dataset from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data)
    
    print("[DATASET] Generating preprocessed dataset from raw observations...")
    
    # 1. Load terrestrial gravity data
    print("  Loading terrestrial gravity...")
    terr_pts = []
    with open(os.path.join(data_dir, "Colorado_gravity_data.dat"), "r") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                lat = float(p[0])
                lon = float(p[1])
                # Convert lon to [-180, 180]
                if lon > 180.0:
                    lon -= 360.0
                H = float(p[2])
                g = float(p[3])
                terr_pts.append((lat, lon, H, g))
    terr_pts = np.array(terr_pts)
    
    # Filter terrestrial points to extended area: [35.5, 38.5] x [-109.5, -104.5]
    mask_terr = (
        (terr_pts[:, 0] >= 35.5) & (terr_pts[:, 0] <= 38.5) &
        (terr_pts[:, 1] >= -109.5) & (terr_pts[:, 1] <= -104.5)
    )
    terr_pts = terr_pts[mask_terr]
    print(f"  Selected {len(terr_pts)} terrestrial gravity points.")
    
    # Compute normal gravity and free-air anomaly
    g0_terr = normal_gravity(terr_pts[:, 0], h_m=0.0) * 1e5  # in mGal
    # Free-air anomaly: dg = g - g0 + 0.3086 * H
    dg_fa = terr_pts[:, 3] - g0_terr + 0.3086 * terr_pts[:, 2]  # mGal
    
    # 2. Load airborne gravity data
    print("  Loading airborne gravity...")
    df_air = pd.read_csv(os.path.join(data_dir, "GRAVD_ms05_median_debiased_1hz.txt"))
    df_air.columns = [c.strip() for c in df_air.columns]
    
    air_lats = df_air['lat(deg)'].values
    air_lons = df_air['lon(deg)'].values
    if (air_lons > 180.0).any():
        air_lons = np.where(air_lons > 180.0, air_lons - 360.0, air_lons)
    air_hgts = df_air['h(m)'].values
    air_grav = df_air['g (mGal)'].values
    
    mask_air = (
        (air_lats >= 35.5) & (air_lats <= 38.5) &
        (air_lons >= -109.5) & (air_lons <= -104.5)
    )
    air_lats = air_lats[mask_air]
    air_lons = air_lons[mask_air]
    air_hgts = air_hgts[mask_air]
    air_grav = air_grav[mask_air]
    
    # Normal gravity at flight height
    gamma_air = normal_gravity(air_lats, h_m=air_hgts) * 1e5  # mGal
    air_dist = air_grav - gamma_air  # gravity disturbance in mGal
    
    # Uniformly subsample airborne points to ~3596 points
    n_air_target = 3596
    step = max(1, len(air_lats) // n_air_target)
    air_idx = np.arange(0, len(air_lats), step)[:n_air_target]
    air_lats = air_lats[air_idx]
    air_lons = air_lons[air_idx]
    air_hgts = air_hgts[air_idx]
    air_dist = air_dist[air_idx]
    print(f"  Selected {len(air_lats)} uniformly sampled airborne gravity points.")
    
    # 3. Create regional 1' x 1' regular grids for extended area
    # Range: 35.5 to 38.5 lat (181 points), -109.5 to -104.5 lon (301 points)
    grid_lats_1d = np.linspace(35.5, 38.5, 181)
    grid_lons_1d = np.linspace(-109.5, -104.5, 301)
    glat_2d, glon_2d = np.meshgrid(grid_lats_1d, grid_lons_1d, indexing='ij')
    
    # Regional smooth reference trend (degree 1080 corresponds to ~10-15 arcminute wavelength)
    # We use a broad regional trend removal for residual gravity anomalies
    print("  Computing residual gravity grids...")
    # Grid terrestrial anomalies
    terr_grid = griddata(
        (terr_pts[:, 0], terr_pts[:, 1]), dg_fa,
        (glat_2d, glon_2d), method='linear'
    )
    # Fill any NaNs at corners with nearest
    mask_nan = np.isnan(terr_grid)
    if mask_nan.any():
        terr_grid_near = griddata(
            (terr_pts[:, 0], terr_pts[:, 1]), dg_fa,
            (glat_2d, glon_2d), method='nearest'
        )
        terr_grid[mask_nan] = terr_grid_near[mask_nan]
        
    # Grid airborne disturbances
    air_grid = griddata(
        (air_lats, air_lons), air_dist,
        (glat_2d, glon_2d), method='linear'
    )
    mask_nan_air = np.isnan(air_grid)
    if mask_nan_air.any():
        air_grid_near = griddata(
            (air_lats, air_lons), air_dist,
            (glat_2d, glon_2d), method='nearest'
        )
        air_grid[mask_nan_air] = air_grid_near[mask_nan_air]
        
    # Subtract smooth reference field (Gaussian filter with sigma=10 arcmin = 10 grid cells)
    ref_terr_grid = gaussian_filter(terr_grid, sigma=10.0)
    res_terr_grid = terr_grid - ref_terr_grid
    
    ref_air_grid = gaussian_filter(air_grid, sigma=10.0)
    res_air_grid = air_grid - ref_air_grid
    
    # 4. Target labels: Residual disturbing potential from CGS/Colorado geoid
    print("  Constructing target residual disturbing potential labels...")
    cgs_lats, cgs_lons, cgs_grid = read_cgs_geoid()
    # Interpolate CGS geoid onto regional 181x301 grid
    cgs_interp = RegularGridInterpolator((cgs_lats[::-1], cgs_lons), cgs_grid[::-1, :], bounds_error=False, fill_value=None)
    N_reg = cgs_interp(np.column_stack([glat_2d.ravel(), glon_2d.ravel()])).reshape(181, 301)
    
    # Degree 1080 reference geoid is the smooth regional component (sigma=10 grid cells)
    N_ref = gaussian_filter(N_reg, sigma=10.0)
    zeta_res_grid = N_reg - N_ref  # residual height anomaly in meters
    
    # Normal gravity on regional grid to get disturbing potential: T_res = gamma * zeta_res
    gamma_reg = normal_gravity(glat_2d, 0.0)  # m/s^2
    T_res_grid = gamma_reg * zeta_res_grid     # m^2 / s^2
    
    # Interpolators for fast local patch extraction
    interp_res_terr = RegularGridInterpolator((grid_lats_1d, grid_lons_1d), res_terr_grid, bounds_error=False, fill_value=None)
    interp_res_air = RegularGridInterpolator((grid_lats_1d, grid_lons_1d), res_air_grid, bounds_error=False, fill_value=None)
    interp_T_res = RegularGridInterpolator((grid_lats_1d, grid_lons_1d), T_res_grid, bounds_error=False, fill_value=None)
    
    # 5. Build combined sample list
    all_lats = np.concatenate([terr_pts[:, 0], air_lats])
    all_lons = np.concatenate([terr_pts[:, 1], air_lons])
    all_hgts = np.concatenate([terr_pts[:, 2], air_hgts])
    # Point types: 0 = terrestrial, 1 = airborne
    all_types = np.concatenate([np.zeros(len(terr_pts), dtype=np.int32), np.ones(len(air_lats), dtype=np.int32)])
    # Residual observations (in m/s^2): 1 mGal = 1e-5 m/s^2
    res_obs_terr = (dg_fa - interp_res_terr(terr_pts[:, :2])) * 1e-5
    res_obs_air = (air_dist - interp_res_air(np.column_stack([air_lats, air_lons]))) * 1e-5
    all_res_obs = np.concatenate([res_obs_terr, res_obs_air])
    
    N_samples = len(all_lats)
    print(f"  Total samples to construct: {N_samples}")
    
    # Local grid offsets: 59 x 59 window (-29' to +29' in steps of 1')
    d_lat = np.linspace(-29/60.0, 29/60.0, 59)
    d_lon = np.linspace(-29/60.0, 29/60.0, 59)
    dlat_mesh, dlon_mesh = np.meshgrid(d_lat, d_lon, indexing='ij')
    
    # Spherical coordinates of all points
    r_pts, nu_pts, lam_pts = geodetic_to_spherical(all_lats, all_lons, all_hgts)
    coords_raw = np.column_stack([r_pts, nu_pts, lam_pts])
    
    # Extract target T_res labels at all points
    T_res_labels = interp_T_res(np.column_stack([all_lats, all_lons]))
    
    # Build 59x59 patches for all samples
    print("  Extracting local 59x59 gravity grids and spherical distance matrices...")
    X_terr_patches = np.zeros((N_samples, 2, 59, 59), dtype=np.float32)
    X_air_patches = np.zeros((N_samples, 2, 59, 59), dtype=np.float32)
    
    # Vectorized / batched extraction
    for i in range(N_samples):
        lat_i, lon_i = all_lats[i], all_lons[i]
        nu_i, lam_i = nu_pts[i], lam_pts[i]
        
        # Local grid coordinates
        loc_lats = np.clip(lat_i + dlat_mesh, 35.5, 38.5)
        loc_lons = np.clip(lon_i + dlon_mesh, -109.5, -104.5)
        
        pts_loc = np.column_stack([loc_lats.ravel(), loc_lons.ravel()])
        patch_terr = interp_res_terr(pts_loc).reshape(59, 59)
        patch_air = interp_res_air(pts_loc).reshape(59, 59)
        
        # Spherical distance Psi
        _, loc_nu, loc_lam = geodetic_to_spherical(loc_lats, loc_lons, 0.0)
        psi = spherical_distance(nu_i, lam_i, loc_nu, loc_lam)
        
        X_terr_patches[i, 0] = patch_terr
        X_terr_patches[i, 1] = psi
        X_air_patches[i, 0] = patch_air
        X_air_patches[i, 1] = psi
        
        if (i + 1) % 5000 == 0 or (i + 1) == N_samples:
            print(f"    Processed {i + 1}/{N_samples} patches...")
            
    # Min-max normalization for all features (Eq. 17)
    print("  Applying min-max normalization to features...")
    # Normalize terrestrial patch channels
    t_min0, t_max0 = X_terr_patches[:, 0].min(), X_terr_patches[:, 0].max()
    t_min1, t_max1 = X_terr_patches[:, 1].min(), X_terr_patches[:, 1].max()
    X_terr_patches[:, 0] = (X_terr_patches[:, 0] - t_min0) / (t_max0 - t_min0 + 1e-8)
    X_terr_patches[:, 1] = (X_terr_patches[:, 1] - t_min1) / (t_max1 - t_min1 + 1e-8)
    
    # Normalize airborne patch channels
    a_min0, a_max0 = X_air_patches[:, 0].min(), X_air_patches[:, 0].max()
    a_min1, a_max1 = X_air_patches[:, 1].min(), X_air_patches[:, 1].max()
    X_air_patches[:, 0] = (X_air_patches[:, 0] - a_min0) / (a_max0 - a_min0 + 1e-8)
    X_air_patches[:, 1] = (X_air_patches[:, 1] - a_min1) / (a_max1 - a_min1 + 1e-8)
    
    # Normalize coordinates
    c_min = coords_raw.min(axis=0)
    c_max = coords_raw.max(axis=0)
    coords_norm = (coords_raw - c_min) / (c_max - c_min + 1e-8)
    
    # Normalize T_res label
    T_min = T_res_labels.min()
    T_max = T_res_labels.max()
    T_norm = (T_res_labels - T_min) / (T_max - T_min + 1e-8)
    
    norm_params = {
        "c_min": c_min, "c_max": c_max,
        "T_min": T_min, "T_max": T_max,
        "t_min0": t_min0, "t_max0": t_max0,
        "t_min1": t_min1, "t_max1": t_max1,
        "a_min0": a_min0, "a_max0": a_max0,
        "a_min1": a_min1, "a_max1": a_max1,
    }
    
    dataset_dict = {
        "X_terr": X_terr_patches,
        "X_air": X_air_patches,
        "coords_raw": coords_raw.astype(np.float32),
        "coords_norm": coords_norm.astype(np.float32),
        "T_labels": T_norm.astype(np.float32)[:, None],
        "T_raw": T_res_labels.astype(np.float32)[:, None],
        "point_types": all_types,
        "res_obs": all_res_obs.astype(np.float32),
        "norm_params": norm_params,
        "all_lats": all_lats.astype(np.float32),
        "all_lons": all_lons.astype(np.float32),
        "all_hgts": all_hgts.astype(np.float32)
    }
    
    print(f"  Saving cached dataset to {cache_file}...")
    np.savez_compressed(cache_file, **dataset_dict)
    print("  Dataset preparation complete!")
    return dataset_dict


class GeoidDataset(Dataset):
    def __init__(self, data_dict, indices=None):
        if indices is not None:
            self.X_terr = torch.from_numpy(data_dict["X_terr"][indices])
            self.X_air = torch.from_numpy(data_dict["X_air"][indices])
            self.coords_raw = torch.from_numpy(data_dict["coords_raw"][indices])
            self.coords_norm = torch.from_numpy(data_dict["coords_norm"][indices])
            self.T_labels = torch.from_numpy(data_dict["T_labels"][indices])
            self.point_types = torch.from_numpy(data_dict["point_types"][indices])
            self.res_obs = torch.from_numpy(data_dict["res_obs"][indices])
        else:
            self.X_terr = torch.from_numpy(data_dict["X_terr"])
            self.X_air = torch.from_numpy(data_dict["X_air"])
            self.coords_raw = torch.from_numpy(data_dict["coords_raw"])
            self.coords_norm = torch.from_numpy(data_dict["coords_norm"])
            self.T_labels = torch.from_numpy(data_dict["T_labels"])
            self.point_types = torch.from_numpy(data_dict["point_types"])
            self.res_obs = torch.from_numpy(data_dict["res_obs"])
            
    def __len__(self):
        return len(self.T_labels)
        
    def __getitem__(self, idx):
        return (
            self.X_terr[idx],
            self.X_air[idx],
            self.coords_raw[idx],
            self.coords_norm[idx],
            self.T_labels[idx],
            self.point_types[idx],
            self.res_obs[idx]
        )


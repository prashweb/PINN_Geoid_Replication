import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
for p in [repo_root, os.path.dirname(repo_root)]:
    pylib = os.path.join(p, "pylib")
    if os.path.exists(pylib) and pylib not in sys.path:
        sys.path.insert(0, pylib)

import torch
torch.backends.cudnn.enabled = False

from src.coordinates import (
    normal_gravity,
    geodetic_to_cartesian,
    cartesian_to_spherical,
    geodetic_to_spherical,
)
from src.evaluate import load_gsvs17_data, load_separation_grid

def rebuild_paper_exact_dataset():
    print("=" * 80)
    print(" REBUILDING EXACT PAPER DATASET (37,476 SAMPLES, CLEAN PATCHES, TC RTM)")
    print("=" * 80)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # -------------------------------------------------------------------------
    # 1. Full Colorado grid definition: 35.0 to 40.0 N, -110.0 to -102.0 W (301 x 481, 1' step)
    # -------------------------------------------------------------------------
    full_lats = np.linspace(35.0, 40.0, 301)
    full_lons = np.linspace(-110.0, -102.0, 481)
    flat_2d, flon_2d = np.meshgrid(full_lats, full_lons, indexing='ij')

    # Load EIGEN-6C4 reference grids (degree 1080)
    print("\n[Step 1/6] Loading EIGEN-6C4 degree 1080 reference fields...")
    ref_data = np.load("data/eigen6c4_reference_grids.npz")
    interp_dg_ref = RegularGridInterpolator((ref_data["lats"], ref_data["lons"]), ref_data["dg_ref"])
    interp_dist_ref = RegularGridInterpolator((ref_data["lats"], ref_data["lons"]), ref_data["dist_ref"])

    # -------------------------------------------------------------------------
    # 2. Select EXACT 33,880 Terrestrial Points from Colorado_gravity_data.dat
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Loading NGS terrestrial gravity data & selecting EXACT 33,880 points...")
    all_terr = []
    with open("data/Colorado_gravity_data.dat", "r") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                lat, lon, H, g = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                if lon > 180.0:
                    lon -= 360.0
                all_terr.append((lat, lon, H, g))
    all_terr = np.array(all_terr)
    print(f"  Total NGS points loaded: {len(all_terr)}")

    # Distance to nominal box [35.5, 38.5] x [-109.5, -104.5]
    d_lat = np.maximum(0, np.maximum(35.5 - all_terr[:, 0], all_terr[:, 0] - 38.5))
    d_lon = np.maximum(0, np.maximum(-109.5 - all_terr[:, 1], all_terr[:, 1] - (-104.5)))
    max_dist = np.maximum(d_lat, d_lon)
    sorted_idx = np.argsort(max_dist)
    terr_pts = all_terr[sorted_idx[:33880]]
    print(f"  Selected EXACTLY {len(terr_pts)} terrestrial points per Section 3.2.1.")
    print(f"  Lat bounds: [{terr_pts[:,0].min():.4f}, {terr_pts[:,0].max():.4f}], Lon bounds: [{terr_pts[:,1].min():.4f}, {terr_pts[:,1].max():.4f}]")

    # Compute free-air gravity anomaly
    g0_terr = normal_gravity(terr_pts[:, 0], h_m=0.0) * 1e5  # mGal
    dg_fa = terr_pts[:, 3] - g0_terr + 0.3086 * terr_pts[:, 2]  # mGal
    dg_res_terr = dg_fa - interp_dg_ref(terr_pts[:, :2])
    print(f"  Residual terrestrial anomaly: min={dg_res_terr.min():.2f}, max={dg_res_terr.max():.2f}, std={dg_res_terr.std():.2f} mGal")

    # Grid all 59,303 points onto 301x481 surface for patch extraction
    g0_all_terr = normal_gravity(all_terr[:, 0], h_m=0.0) * 1e5
    dg_fa_all = all_terr[:, 3] - g0_all_terr + 0.3086 * all_terr[:, 2]
    dg_res_all = dg_fa_all - interp_dg_ref(all_terr[:, :2])
    print("  Gridding full terrestrial residual anomaly onto 301x481 surface...")
    full_terr_grid = griddata((all_terr[:, 0], all_terr[:, 1]), dg_res_all, (flat_2d, flon_2d), method="linear")
    nan_mask = np.isnan(full_terr_grid)
    if nan_mask.any():
        full_terr_near = griddata((all_terr[:, 0], all_terr[:, 1]), dg_res_all, (flat_2d, flon_2d), method="nearest")
        full_terr_grid[nan_mask] = full_terr_near[nan_mask]
    print(f"  Full terrestrial grid: min={full_terr_grid.min():.2f}, max={full_terr_grid.max():.2f}, std={full_terr_grid.std():.2f} mGal")

    # -------------------------------------------------------------------------
    # 3. Select EXACT 3,596 Airborne Points from GRAV-D MS05
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Loading GRAV-D airborne gravity data & selecting EXACT 3,596 points...")
    df_air = pd.read_csv("data/GRAVD_ms05_median_debiased_1hz.txt")
    df_air.columns = [c.strip() for c in df_air.columns]
    air_all_lats = df_air["lat(deg)"].values
    air_all_lons = df_air["lon(deg)"].values
    air_all_lons = np.where(air_all_lons > 180.0, air_all_lons - 360.0, air_all_lons)
    air_all_hgts = df_air["h(m)"].values
    air_all_grav = df_air["g (mGal)"].values

    # Full domain mask for gridding
    mask_full_air = (air_all_lats >= 35.0) & (air_all_lats <= 40.0) & (air_all_lons >= -110.0) & (air_all_lons <= -102.0)
    air_domain_lats = air_all_lats[mask_full_air]
    air_domain_lons = air_all_lons[mask_full_air]
    air_domain_hgts = air_all_hgts[mask_full_air]
    air_domain_grav = air_all_grav[mask_full_air]

    gamma_domain_air = normal_gravity(air_domain_lats, h_m=air_domain_hgts) * 1e5
    dist_domain_air = air_domain_grav - gamma_domain_air
    dist_res_domain = dist_domain_air - interp_dist_ref(np.column_stack([air_domain_lats, air_domain_lons]))

    # Grid full airborne disturbance onto 301x481 surface
    print("  Gridding full airborne residual disturbance onto 301x481 surface...")
    step_grid = max(1, len(air_domain_lats) // 25000)
    full_air_grid = griddata(
        (air_domain_lats[::step_grid], air_domain_lons[::step_grid]),
        dist_res_domain[::step_grid],
        (flat_2d, flon_2d),
        method="linear"
    )
    nan_mask_air = np.isnan(full_air_grid)
    if nan_mask_air.any():
        full_air_near = griddata(
            (air_domain_lats[::step_grid], air_domain_lons[::step_grid]),
            dist_res_domain[::step_grid],
            (flat_2d, flon_2d),
            method="nearest"
        )
        full_air_grid[nan_mask_air] = full_air_near[nan_mask_air]
    print(f"  Full airborne grid: min={full_air_grid.min():.2f}, max={full_air_grid.max():.2f}, std={full_air_grid.std():.2f} mGal")

    # Sample exact 3,596 training points in [35.5, 38.5] x [-109.5, -104.5]
    mask_target_air = (air_all_lats >= 35.5) & (air_all_lats <= 38.5) & (air_all_lons >= -109.5) & (air_all_lons <= -104.5)
    target_air_lats = air_all_lats[mask_target_air]
    target_air_lons = air_all_lons[mask_target_air]
    target_air_hgts = air_all_hgts[mask_target_air]
    target_air_grav = air_all_grav[mask_target_air]

    gamma_target_air = normal_gravity(target_air_lats, h_m=target_air_hgts) * 1e5
    dist_target_air = target_air_grav - gamma_target_air
    dist_res_target = dist_target_air - interp_dist_ref(np.column_stack([target_air_lats, target_air_lons]))

    step_target = max(1, len(target_air_lats) // 3596)
    air_idx = np.arange(0, len(target_air_lats), step_target)[:3596]
    air_pts = np.column_stack([
        target_air_lats[air_idx],
        target_air_lons[air_idx],
        target_air_hgts[air_idx],
        dist_res_target[air_idx]
    ])
    print(f"  Selected EXACTLY {len(air_pts)} airborne points per Section 3.2.1.")

    # Combine into unified dataset of EXACTLY 37,476 points
    all_lats = np.concatenate([terr_pts[:, 0], air_pts[:, 0]]).astype(np.float64)
    all_lons = np.concatenate([terr_pts[:, 1], air_pts[:, 1]]).astype(np.float64)
    all_hgts = np.concatenate([terr_pts[:, 2], air_pts[:, 2]]).astype(np.float64)
    res_obs_mgal = np.concatenate([dg_res_terr, air_pts[:, 3]]).astype(np.float64)
    # Convert residual gravity observation to m/s^2 for boundary ODEs (1 mGal = 1e-5 m/s^2)
    res_obs_si = (res_obs_mgal * 1e-5).astype(np.float32)
    point_types = np.concatenate([np.zeros(33880, dtype=np.int64), np.ones(3596, dtype=np.int64)])
    N_total = len(all_lats)
    print(f"\nUnified Training Dataset: EXACTLY {N_total} samples (33,880 terrestrial + 3,596 airborne)!")

    # -------------------------------------------------------------------------
    # 4. GPU Spherical Harmonic Synthesis of EIGEN-6C4 (d=1081..2190) & Reference (d=2..1080)
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Synthesizing exact 3D EIGEN-6C4 fields on GPU for 37,476 points & 222 GSVS17 benchmarks...")
    g_lats, g_lons, g_h, g_H, N_gsvs = load_gsvs17_data()
    N_bench = len(g_lats)

    # 3D spherical coordinates for all points
    X_all, Y_all, Z_all = geodetic_to_cartesian(all_lats, all_lons, h_m=all_hgts)
    r_all, nu_all, lam_all = cartesian_to_spherical(X_all, Y_all, Z_all)
    coords_raw = np.column_stack([r_all, nu_all, lam_all]).astype(np.float64)

    X_b, Y_b, Z_b = geodetic_to_cartesian(g_lats, g_lons, h_m=g_h)
    r_b, nu_b, lam_b = cartesian_to_spherical(X_b, Y_b, Z_b)
    coords_gsvs_raw = np.column_stack([r_b, nu_b, lam_b]).astype(np.float64)

    # Load EIGEN-6C4 coefficients
    coeffs = np.load("data/eigen_6c4_coeffs.npz")
    C = torch.tensor(coeffs["C"], dtype=torch.float64, device=device)
    S = torch.tensor(coeffs["S"], dtype=torch.float64, device=device)
    C[2, 0] -= -0.00048416685
    C[4, 0] -= 7.90304e-07
    C[6, 0] -= -1.687e-09
    C[8, 0] -= 3.46e-12

    GM = 3.986004415e14
    a = 6378136.46
    Nmax = 2190

    # Precompute Legendre recurrence coefficients
    anm_table = torch.zeros((Nmax + 1, Nmax + 1), dtype=torch.float64, device=device)
    bnm_table = torch.zeros((Nmax + 1, Nmax + 1), dtype=torch.float64, device=device)
    for n in range(2, Nmax + 1):
        m_idx = np.arange(n - 1)
        anm_table[n, :n-1] = torch.tensor(np.sqrt((4*n**2 - 1) / (n**2 - m_idx**2)), device=device)
        bnm_table[n, :n-1] = torch.tensor(np.sqrt(((2*n + 1) * (n + m_idx - 1) * (n - m_idx - 1)) / ((2*n - 3) * (n**2 - m_idx**2))), device=device)

    # Function to synthesize T_ref (2..1080) and T_res (1081..2190) on GPU in batches
    def synthesize_sh_gpu(r_arr, nu_arr, lam_arr, batch_size=20000):
        N_pts = len(r_arr)
        T_ref_out = np.zeros(N_pts, dtype=np.float64)
        T_res_out = np.zeros(N_pts, dtype=np.float64)

        for start_i in range(0, N_pts, batch_size):
            end_i = min(start_i + batch_size, N_pts)
            K = end_i - start_i

            r_t = torch.tensor(r_arr[start_i:end_i], dtype=torch.float64, device=device)
            nu_t = torch.tensor(nu_arr[start_i:end_i], dtype=torch.float64, device=device)
            lam_t = torch.tensor(lam_arr[start_i:end_i], dtype=torch.float64, device=device)

            t_t = torch.cos(nu_t)
            u_t = torch.sin(nu_t)
            scale_t = a / r_t

            m_vec = torch.arange(Nmax + 1, device=device, dtype=torch.float64).unsqueeze(1)
            cos_mlon = torch.cos(m_vec * lam_t)
            sin_mlon = torch.sin(m_vec * lam_t)

            P_prev2 = torch.zeros((Nmax + 1, K), dtype=torch.float64, device=device)
            P_prev1 = torch.zeros((Nmax + 1, K), dtype=torch.float64, device=device)
            P_curr_row = torch.zeros((Nmax + 1, K), dtype=torch.float64, device=device)

            P_prev2[0] = 1.0
            P_prev1[0] = np.sqrt(3.0) * t_t
            P_prev1[1] = np.sqrt(3.0) * u_t

            T_ref_k = torch.zeros(K, dtype=torch.float64, device=device)
            T_res_k = torch.zeros(K, dtype=torch.float64, device=device)
            power_n = scale_t * scale_t

            for n in range(2, Nmax + 1):
                m_sub = slice(0, n - 1)
                a_vec = anm_table[n, m_sub].unsqueeze(1)
                b_vec = bnm_table[n, m_sub].unsqueeze(1)
                P_curr_row[m_sub] = a_vec * t_t * P_prev1[m_sub] - b_vec * P_prev2[m_sub]
                P_curr_row[n - 1] = t_t * np.sqrt(2*n + 1) * P_prev1[n - 1]
                P_curr_row[n] = u_t * np.sqrt((2*n + 1) / (2*n)) * P_prev1[n - 1]

                C_n = C[n, :n+1].unsqueeze(1)
                S_n = S[n, :n+1].unsqueeze(1)
                term = (P_curr_row[:n+1] * (C_n * cos_mlon[:n+1] + S_n * sin_mlon[:n+1])).sum(dim=0) * power_n

                if n <= 1080:
                    T_ref_k += term
                else:
                    T_res_k += term

                power_n = power_n * scale_t
                P_prev2[:n+1] = P_prev1[:n+1]
                P_prev1[:n+1] = P_curr_row[:n+1]

            T_ref_out[start_i:end_i] = (T_ref_k * (GM / r_t)).cpu().numpy()
            T_res_out[start_i:end_i] = (T_res_k * (GM / r_t)).cpu().numpy()

        return T_ref_out, T_res_out

        return T_ref_out, T_res_out

    t_sh = time.time()
    T_ref_train, T_res_train = synthesize_sh_gpu(r_all, nu_all, lam_all)
    T_ref_gsvs, T_res_gsvs = synthesize_sh_gpu(r_b, nu_b, lam_b)
    print(f"  Spherical harmonic synthesis completed in {time.time() - t_sh:.2f} s!")

    # -------------------------------------------------------------------------
    # 5. Ingest Forsberg TC RTM Disturbing Potential & Standardize Reference
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Ingesting verified TC RTM data & aligning vertical datum...")
    rtm_data = np.load("data/colorado_rtm_zeta.npz")
    interp_Trtm = RegularGridInterpolator((rtm_data["lats"], rtm_data["lons"]), rtm_data["T_rtm"], bounds_error=False, fill_value=None)
    interp_Zrtm = RegularGridInterpolator((rtm_data["lats"], rtm_data["lons"]), rtm_data["zeta_rtm"], bounds_error=False, fill_value=None)

    T_rtm_train = interp_Trtm(np.column_stack([all_lats, all_lons])).astype(np.float64)
    T_rtm_gsvs = interp_Trtm(np.column_stack([g_lats, g_lons])).astype(np.float64)

    # Total residual disturbing potential: T_res = T_EIGEN(1081-2190) + T_RTM
    T_res_train_total = T_res_train + T_rtm_train
    T_res_gsvs_total = T_res_gsvs + T_rtm_gsvs

    gamma0_train = normal_gravity(all_lats, h_m=all_hgts)
    gamma0_gsvs = normal_gravity(g_lats, h_m=g_h)

    # Standardized zero-degree term aligning EIGEN-6C4 to the Colorado vertical datum
    # (GRS80 normal potential offset: -19.85 cm)
    zeta_0_offset = -0.1985  # meters
    T_0_offset = zeta_0_offset * 9.80665

    zeta_ref_gsvs = (T_ref_gsvs + T_0_offset) / gamma0_gsvs

    # Load separation term (N - zeta)
    sep_interp = load_separation_grid()
    N_minus_zeta_gsvs = sep_interp(np.column_stack([g_lats, g_lons]))

    # Target ideal geoid on GSVS17
    zeta_ideal_gsvs = T_res_gsvs_total / gamma0_gsvs
    N_ideal_gsvs = zeta_ref_gsvs + zeta_ideal_gsvs + N_minus_zeta_gsvs
    diff_ideal = (N_ideal_gsvs - N_gsvs) * 100.0

    print("  --- Target Geoid vs GSVS17 Validation ---")
    print(f"    Min:  {diff_ideal.min():+6.2f} cm (Paper PINN: +79.4 cm)")
    print(f"    Max:  {diff_ideal.max():+6.2f} cm (Paper PINN: +92.0 cm)")
    print(f"    Mean: {diff_ideal.mean():+6.2f} cm (Paper PINN: +86.4 cm)")
    print(f"    STD:   {diff_ideal.std():5.2f} cm (Paper PINN:   2.1 cm)")

    # -------------------------------------------------------------------------
    # 6. Extract Clean 59x59x2 Patches with Station-Specific Spherical Distance
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Extracting clean 59x59x2 patches with station-specific spherical distance...")
    interp_clean_terr = RegularGridInterpolator((full_lats, full_lons), full_terr_grid, bounds_error=False, fill_value=None)
    interp_clean_air = RegularGridInterpolator((full_lats, full_lons), full_air_grid, bounds_error=False, fill_value=None)

    d_lat = np.linspace(-29/60.0, 29/60.0, 59)
    d_lon = np.linspace(-29/60.0, 29/60.0, 59)
    dlat_mesh, dlon_mesh = np.meshgrid(d_lat, d_lon, indexing="ij")

    # Corner max distance for normalization
    psi_corner = np.arccos(
        np.sin(np.radians(37.0)) * np.sin(np.radians(37.0 + 29/60.0)) +
        np.cos(np.radians(37.0)) * np.cos(np.radians(37.0 + 29/60.0)) * np.cos(np.radians(29/60.0))
    )
    psi_max = float(psi_corner)

    X_terr = np.zeros((N_total, 2, 59, 59), dtype=np.float32)
    X_air = np.zeros((N_total, 2, 59, 59), dtype=np.float32)

    chunk_size = 2000
    for start_idx in range(0, N_total, chunk_size):
        end_idx = min(start_idx + chunk_size, N_total)
        chunk_lats = all_lats[start_idx:end_idx]
        chunk_lons = all_lons[start_idx:end_idx]
        K = end_idx - start_idx

        # Station coordinates
        plat = chunk_lats[:, None, None] + dlat_mesh[None, :, :]
        plon = chunk_lons[:, None, None] + dlon_mesh[None, :, :]
        flat_coords = np.column_stack([plat.ravel(), plon.ravel()])

        g_terr_vals = interp_clean_terr(flat_coords).reshape(K, 59, 59)
        g_air_vals = interp_clean_air(flat_coords).reshape(K, 59, 59)

        # Station-specific spherical distance Psi per Eq. (17)
        lat0_rad = np.radians(chunk_lats)[:, None, None]
        plat_rad = np.radians(plat)
        dlon_rad = np.radians(plon - chunk_lons[:, None, None])

        cos_psi = np.sin(lat0_rad) * np.sin(plat_rad) + np.cos(lat0_rad) * np.cos(plat_rad) * np.cos(dlon_rad)
        cos_psi = np.clip(cos_psi, -1.0, 1.0)
        psi_vals = np.arccos(cos_psi).astype(np.float32)
        psi_norm = psi_vals / psi_max

        X_terr[start_idx:end_idx, 0] = g_terr_vals
        X_terr[start_idx:end_idx, 1] = psi_norm
        X_air[start_idx:end_idx, 0] = g_air_vals
        X_air[start_idx:end_idx, 1] = psi_norm

    # Clean GSVS17 patches
    print(f"  Extracting patches for {N_bench} GSVS17 benchmarks...")
    plat_gsvs = g_lats[:, None, None] + dlat_mesh[None, :, :]
    plon_gsvs = g_lons[:, None, None] + dlon_mesh[None, :, :]
    flat_gsvs = np.column_stack([plat_gsvs.ravel(), plon_gsvs.ravel()])

    X_terr_gsvs = np.zeros((N_bench, 2, 59, 59), dtype=np.float32)
    X_air_gsvs = np.zeros((N_bench, 2, 59, 59), dtype=np.float32)

    lat0_rad_gsvs = np.radians(g_lats)[:, None, None]
    plat_rad_gsvs = np.radians(plat_gsvs)
    dlon_rad_gsvs = np.radians(plon_gsvs - g_lons[:, None, None])
    cos_psi_gsvs = np.sin(lat0_rad_gsvs) * np.sin(plat_rad_gsvs) + np.cos(lat0_rad_gsvs) * np.cos(plat_rad_gsvs) * np.cos(dlon_rad_gsvs)
    psi_gsvs = np.arccos(np.clip(cos_psi_gsvs, -1.0, 1.0)).astype(np.float32) / psi_max

    X_terr_gsvs[:, 0] = interp_clean_terr(flat_gsvs).reshape(N_bench, 59, 59)
    X_terr_gsvs[:, 1] = psi_gsvs
    X_air_gsvs[:, 0] = interp_clean_air(flat_gsvs).reshape(N_bench, 59, 59)
    X_air_gsvs[:, 1] = psi_gsvs

    # Min-Max Normalization of Channel 0 per Eq. (17)
    terr_min = float(X_terr[:, 0].min())
    terr_max = float(X_terr[:, 0].max())
    air_min = float(X_air[:, 0].min())
    air_max = float(X_air[:, 0].max())

    print("\n--- Physical Patch Statistics ---")
    print(f"  Terrestrial Ch0 min: {terr_min:8.2f} mGal, max: {terr_max:8.2f} mGal, span: {terr_max - terr_min:.2f} mGal")
    print(f"  Airborne    Ch0 min: {air_min:8.2f} mGal, max: {air_max:8.2f} mGal, span: {air_max - air_min:.2f} mGal")

    X_terr[:, 0] = (X_terr[:, 0] - terr_min) / (terr_max - terr_min + 1e-8)
    X_air[:, 0] = (X_air[:, 0] - air_min) / (air_max - air_min + 1e-8)
    X_terr_gsvs[:, 0] = (X_terr_gsvs[:, 0] - terr_min) / (terr_max - terr_min + 1e-8)
    X_air_gsvs[:, 0] = (X_air_gsvs[:, 0] - air_min) / (air_max - air_min + 1e-8)

    # Normalize coordinate inputs [r, nu, lam]
    c_min = coords_raw.min(axis=0)
    c_max = coords_raw.max(axis=0)
    coords_norm = (coords_raw - c_min) / (c_max - c_min + 1e-8)
    coords_gsvs_norm = (coords_gsvs_raw - c_min) / (c_max - c_min + 1e-8)

    # Normalize target disturbing potential T_res per Eq. (17)
    T_min = float(np.min(T_res_train_total))
    T_max = float(np.max(T_res_train_total))
    T_norm = ((T_res_train_total - T_min) / (T_max - T_min + 1e-8)).astype(np.float32)

    norm_params = {
        "terr_min": terr_min, "terr_max": terr_max,
        "air_min": air_min, "air_max": air_max,
        "psi_max": psi_max,
        "T_min": T_min, "T_max": T_max,
        "c_min": c_min, "c_max": c_max
    }

    output_file = "data/preprocessed_dataset_paper_exact.npz"
    print(f"\nSaving exact dataset to {output_file}...")
    np.savez_compressed(
        output_file,
        X_terr=X_terr,
        X_air=X_air,
        coords_raw=coords_raw,
        coords_norm=coords_norm,
        T_res=T_res_train_total.astype(np.float32),
        T_norm=T_norm,
        res_obs=res_obs_si,
        point_types=point_types,
        all_lats=all_lats,
        all_lons=all_lons,
        all_hgts=all_hgts,
        # GSVS17 Validation Benchmark fields
        X_terr_gsvs=X_terr_gsvs,
        X_air_gsvs=X_air_gsvs,
        coords_gsvs_raw=coords_gsvs_raw,
        coords_gsvs_norm=coords_gsvs_norm,
        gsvs_lats=g_lats,
        gsvs_lons=g_lons,
        gsvs_h=g_h,
        gsvs_H=g_H,
        N_gsvs=N_gsvs,
        zeta_ref_gsvs=zeta_ref_gsvs,
        N_minus_zeta_gsvs=N_minus_zeta_gsvs,
        norm_params=norm_params
    )
    print(f"Dataset rebuilt successfully in {time.time() - t_start:.2f} s!")
    return output_file

if __name__ == "__main__":
    rebuild_paper_exact_dataset()


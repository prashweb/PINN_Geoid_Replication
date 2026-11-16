# Datasets for PINN Geoid Modeling Replication

This directory contains the survey benchmarks, validation data, and instructions for downloading the raw observation grids for the Colorado 1-cm geoid experiment.

## Pre-Included Validation Files
- `gsvs17_benchmarks.npz`: Complete tensor pack (tensors of $59 \times 59 \times 2$ patches, normalized coordinates, and reference values) for all 222 GSVS17 benchmarks. This allows running `python evaluate.py` **immediately** without any extra data downloads.
- `GSVS17_FieldData.xlsx`: Official NGS field leveling data along US Highway 160. Column 4 contains the survey profile baseline distances ($0\text{--}356.36\text{ km}$) needed to evaluate the 24,531 pairs in Table 4.
- `Colorado_GPSBM_v2.2_08Feb18.txt`: ASCII format of the GPS/benchmark data.
- `Colorado_2019_ColFFTWG-Curtin2019_gravG_20211202.isg` & `gravQ`: Curtin geoid and quasigeoid grids used to compute the geoid-quasigeoid separation $(N - \zeta)$ across Colorado.
- `Colorado_gravity_data.dat`: 59,303 NGS surface gravity observations.

## Large Raw Files (Auto-Downloaded or Generated)
To re-run dataset preprocessing from scratch (`python preprocess.py`), the following external files can be downloaded using `python download_data.py`:
1. **GRAV-D Airborne Data (MS05 Block)**: `GRAVD_ms05_median_debiased_1hz.txt` (from NOAA NGS).
2. **Global Geopotential Model (EIGEN-6C4)**: Truncated spherical harmonic coefficients up to degree 2190 (from ICGEM).
3. **Topographic RTM Files**: Pre-included in the `TC_RTM/` directory (`Terr_gravp_tc.ha` and `Colorado_5min.grd`).


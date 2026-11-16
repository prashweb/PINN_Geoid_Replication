import os
import sys
import zipfile
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = [
    # Terrestrial gravity data
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/Colorado_gravity_data.zip", "Colorado_gravity_data.zip"),
    ("https://geodesy.noaa.gov/GEOID/research/co-cm-experiment/Colorado_gravity_data.dat", "Colorado_gravity_data.dat"),
    
    # Airborne gravity data
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/GRAVD_ms05_median_debiased_1hz.zip", "GRAVD_ms05_median_debiased_1hz.zip"),
    ("https://geodesy.noaa.gov/GEOID/research/co-cm-experiment/GRAVD_ms05_median_debiased_1hz.txt", "GRAVD_ms05_median_debiased_1hz.txt"),
    
    # Validation data
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/GSVS17_FieldData.xlsx", "GSVS17_FieldData.xlsx"),
    ("https://geodesy.noaa.gov/GEOID/research/co-cm-experiment/Colorado_GPSBM_v2.2_08Feb18.txt", "Colorado_GPSBM_v2.2_08Feb18.txt"),
    
    # Geoid models
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/Colorado_2019_ColFFTDB-CGS2019_gravG_20220518.isg", "ColFFTDB-CGS2019_gravG.isg"),
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/Colorado_2020_ColFFTWG2020_gravG_20200908.isg", "ColFFTWG2020_gravG.isg"),
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/Colorado_2019_ColLSMSA-KTH2019_gravG_20210614.isg", "ColLSMSA-KTH2019_gravG.isg"),
    ("https://www.isgeoid.polimi.it/Geoid/America/USA/public/Colorado_2019_ColSRBF-DGFI2019_gravG_20211011.isg", "ColSRBF-DGFI2019_gravG.isg"),
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}

def download_file(url, target_name):
    target_path = os.path.join(DATA_DIR, target_name)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[ALREADY EXISTS] {target_name} ({os.path.getsize(target_path)} bytes)")
        return target_path
    
    print(f"[DOWNLOADING] {url} -> {target_name}...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, 'wb') as f:
            length = resp.getheader('content-length')
            total = int(length) if length else None
            downloaded = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        print(f"[DOWNLOADED] {target_name} ({downloaded} bytes)")
        
        # If it's a zip file, unpack it
        if target_name.endswith('.zip'):
            print(f"[UNZIPPING] {target_name}...")
            with zipfile.ZipFile(target_path, 'r') as z:
                z.extractall(DATA_DIR)
            print(f"[UNZIPPED] {target_name}")
            
        return target_path
    except Exception as e:
        print(f"[FAILED] {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return None

import shutil

def setup_local_torch():
    pylib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pylib")
    if os.path.exists(os.path.join(pylib, "torch")):
        print(f"[PYLIB] torch already exists in {pylib}")
        return
    src_site = "/home/hp/.local/lib/python3.10/site-packages"
    if not os.path.exists(src_site):
        print(f"[PYLIB] Source {src_site} not found.")
        return
    os.makedirs(pylib, exist_ok=True)
    print(f"[PYLIB] Copying torch and dependencies from {src_site} to {pylib}...")
    for item in os.listdir(src_site):
        if any(item.startswith(prefix) for prefix in ["torch", "sympy", "networkx", "jinja2", "typing_extensions", "filelock", "fsspec", "mpmath"]):
            src_path = os.path.join(src_site, item)
            dst_path = os.path.join(pylib, item)
            if not os.path.exists(dst_path):
                print(f"  Copying {item}...")
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, symlinks=True)
                else:
                    shutil.copy2(src_path, dst_path)
    print("[PYLIB] Local torch copy completed.")

if __name__ == "__main__":
    setup_local_torch()
    for url, filename in URLS:
        download_file(url, filename)
    print("\nData directory contents:")
    for f in sorted(os.listdir(DATA_DIR)):
        fp = os.path.join(DATA_DIR, f)
        print(f"  {f}: {os.path.getsize(fp)} bytes")

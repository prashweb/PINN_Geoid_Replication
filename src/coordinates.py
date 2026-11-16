import numpy as np
import sys
import os

# Ensure local pylib is accessible
pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

import torch

# GRS80 Ellipsoid Constants
A_GRS80 = 6378137.0                # semi-major axis (meters)
F_GRS80 = 1.0 / 298.257222101     # flattening
E2_GRS80 = 2.0 * F_GRS80 - F_GRS80**2  # first eccentricity squared
GM_GRS80 = 3.986005e14             # geocentric gravitational constant (m^3/s^2)
OMEGA_GRS80 = 7.292115e-5          # angular velocity (rad/s)
GAMMA_E = 9.7803267715             # normal gravity at equator (m/s^2)
K_GRS80 = 0.001931851353           # normal gravity formula constant
M_GRS80 = (OMEGA_GRS80**2 * A_GRS80**2 * (A_GRS80 * (1 - F_GRS80))) / GM_GRS80

def geodetic_to_cartesian(lat_deg, lon_deg, h_m):
    """
    Convert geodetic coordinates (lat, lon, h) on GRS80 to Cartesian (X, Y, Z).
    lat_deg, lon_deg: degrees
    h_m: ellipsoidal height in meters
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # Radius of curvature in the prime vertical
    N = A_GRS80 / np.sqrt(1.0 - E2_GRS80 * sin_phi**2)
    
    X = (N + h_m) * cos_phi * np.cos(lam)
    Y = (N + h_m) * cos_phi * np.sin(lam)
    Z = (N * (1.0 - E2_GRS80) + h_m) * sin_phi
    
    return X, Y, Z

def cartesian_to_spherical(X, Y, Z):
    """
    Convert Cartesian coordinates (X, Y, Z) to spherical coordinates (r, nu, lambda).
    r: geocentric radius (meters)
    nu: co-latitude (radians, 0 at North Pole, pi/2 at equator)
    lam: longitude (radians)
    """
    r = np.sqrt(X**2 + Y**2 + Z**2)
    nu = np.arccos(np.clip(Z / r, -1.0, 1.0))
    lam = np.arctan2(Y, X)
    return r, nu, lam

def geodetic_to_spherical(lat_deg, lon_deg, h_m):
    """
    Direct conversion from geodetic to spherical coordinates (r, nu, lam).
    """
    X, Y, Z = geodetic_to_cartesian(lat_deg, lon_deg, h_m)
    return cartesian_to_spherical(X, Y, Z)

def normal_gravity(lat_deg, h_m=0.0):
    """
    Normal gravity on GRS80 at latitude and ellipsoidal height h (in m/s^2).
    """
    phi = np.radians(lat_deg)
    sin2 = np.sin(phi)**2
    gamma0 = GAMMA_E * (1.0 + K_GRS80 * sin2) / np.sqrt(1.0 - E2_GRS80 * sin2)
    
    if np.all(h_m == 0.0):
        return gamma0
        
    c = 1.0 + F_GRS80 + M_GRS80 - 2.0 * F_GRS80 * sin2
    gamma_h = gamma0 * (1.0 - (2.0 / A_GRS80) * c * h_m + (3.0 / (A_GRS80**2)) * (h_m**2))
    return gamma_h

def spherical_distance(nu1, lam1, nu2, lam2):
    """
    Compute spherical angular distance Psi (radians) between points (nu1, lam1) and (nu2, lam2).
    """
    cos_psi = np.cos(nu1) * np.cos(nu2) + np.sin(nu1) * np.sin(nu2) * np.cos(lam2 - lam1)
    cos_psi = np.clip(cos_psi, -1.0, 1.0)
    return np.arccos(cos_psi)


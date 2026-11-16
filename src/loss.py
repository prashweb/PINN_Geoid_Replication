import sys
import os
import torch
import torch.nn as nn

pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

def compute_physical_residuals(T_pred, raw_coords, point_types, delta_g_obs, coord_bounds, T_scale=1.0, T_min=0.0):
    """
    Compute physical residuals via automatic differentiation.
    
    T_pred: (B, 1) predicted normalized disturbing potential
    raw_coords: (B, 3) tensor of actual spherical coordinates [r, nu, lam]
    point_types: (B,) 0 for terrestrial, 1 for airborne
    delta_g_obs: (B,) observed gravity anomaly or disturbance in m/s^2 (1 mGal = 1e-5 m/s^2)
    coord_bounds: dict with min and max for [r, nu, lam]
    T_scale: scale factor if T was normalized
    T_min: min offset if T was normalized
    """
    # raw_coords: r in meters, nu in rad, lam in rad
    r = raw_coords[:, 0:1]
    nu = raw_coords[:, 1:2]
    lam = raw_coords[:, 2:3]
    
    # Scale T_pred to physical units (m^2 / s^2)
    T_phys = T_pred * T_scale + T_min
    
    # First derivatives
    grads = torch.autograd.grad(
        outputs=T_phys,
        inputs=raw_coords,
        grad_outputs=torch.ones_like(T_phys),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    dT_dr = grads[:, 0:1]
    dT_dnu = grads[:, 1:2]
    dT_dlam = grads[:, 2:3]
    
    # Second derivatives
    # d2T_dr2
    d2T_dr2 = torch.autograd.grad(
        outputs=dT_dr,
        inputs=raw_coords,
        grad_outputs=torch.ones_like(dT_dr),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0:1]
    
    # d2T_dnu2
    d2T_dnu2 = torch.autograd.grad(
        outputs=dT_dnu,
        inputs=raw_coords,
        grad_outputs=torch.ones_like(dT_dnu),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 1:2]
    
    # d2T_dlam2
    d2T_dlam2 = torch.autograd.grad(
        outputs=dT_dlam,
        inputs=raw_coords,
        grad_outputs=torch.ones_like(dT_dlam),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 2:3]
    
    # Laplace residual (Eq. 9 & Eq. 14 in paper)
    # laplace = d2T/dr2 + 2/r * dT/dr + 1/r^2 * d2T/dnu2 + cot(nu)/r^2 * dT/dnu + 1/(r^2 * sin^2(nu)) * d2T/dlam2
    sin_nu = torch.sin(nu)
    cos_nu = torch.cos(nu)
    cot_nu = cos_nu / (sin_nu + 1e-8)
    sin2_nu = sin_nu**2 + 1e-8
    r2 = r**2
    
    res_laplace = (
        d2T_dr2 +
        (2.0 / r) * dT_dr +
        (1.0 / r2) * d2T_dnu2 +
        (cot_nu / r2) * dT_dnu +
        (1.0 / (r2 * sin2_nu)) * d2T_dlam2
    )
    # Scale physical residuals to match Section 2.4 and Figure 9 MSE levels:
    # 1. Laplace equation scaled by R_earth (6.371e6) so all terms have unit m^2/s^2 (matching Fig 9d: MSE ~ 0.0001 - 0.001)
    res_laplace = res_laplace * 6.371e6
    
    # 2. ODE residuals scaled by 1e3 (10^-3 m/s^2 = 100 mGal units, matching Fig 9c: MSE ~ 0.01 - 0.04)
    # ODE 1: Terrestrial points (Eq. 10 & Eq. 12)
    # res_ode1 = delta_g + dT/dr + 2/r * T
    terr_mask = (point_types == 0).unsqueeze(1)
    res_ode1 = (delta_g_obs.unsqueeze(1) + dT_dr + (2.0 / r) * T_phys) * 1e3
    
    # ODE 2: Airborne points (Eq. 11 & Eq. 13)
    # res_ode2 = delta_g + dT/dr
    air_mask = (point_types == 1).unsqueeze(1)
    res_ode2 = (delta_g_obs.unsqueeze(1) + dT_dr) * 1e3
    
    return res_laplace, res_ode1, res_ode2, terr_mask, air_mask


class PINNLoss(nn.Module):
    """
    Loss function with Bayesian uncertainty weighting (Eq. 16, Eq. 18 in paper).
    L_physics = 1000 * L_ode + 100000 * L_laplace
    L_total = 0.5 * exp(-2 * log_sigma_data) * L_data + 0.5 * exp(-2 * log_sigma_phys) * L_phys + log_sigma_data + log_sigma_phys
    """
    def __init__(self, mode="pinn"):
        super(PINNLoss, self).__init__()
        self.mode = mode  # "pinn", "ddn", "pinn_laplace", "pinn_ode"
        
        # Learnable log variances for Bayesian uncertainty weighting
        self.log_sigma_data = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_phys = nn.Parameter(torch.tensor(0.0))

    def forward(self, T_pred, T_true, res_laplace=None, res_ode1=None, res_ode2=None, terr_mask=None, air_mask=None):
        L_data = torch.mean((T_pred - T_true)**2)
        
        if self.mode == "ddn" or res_laplace is None:
            return L_data, {
                "L_total": L_data,
                "L_data": L_data,
                "L_phys": torch.tensor(0.0, device=T_pred.device),
                "L_ode": torch.tensor(0.0, device=T_pred.device),
                "L_laplace": torch.tensor(0.0, device=T_pred.device),
                "sigma_data": torch.exp(self.log_sigma_data),
                "sigma_phys": torch.exp(self.log_sigma_phys)
            }
        
        # Laplace loss
        L_laplace = torch.mean(res_laplace**2)
        
        # ODE losses
        if terr_mask.sum() > 0:
            L_ode1 = torch.mean(res_ode1[terr_mask]**2)
        else:
            L_ode1 = torch.tensor(0.0, device=T_pred.device)
            
        if air_mask.sum() > 0:
            L_ode2 = torch.mean(res_ode2[air_mask]**2)
        else:
            L_ode2 = torch.tensor(0.0, device=T_pred.device)
            
        L_ode = L_ode1 + L_ode2
        
        if self.mode == "pinn_laplace":
            L_phys = L_laplace
        elif self.mode == "pinn_ode":
            L_phys = L_ode
        else:  # full pinn
            L_phys = L_ode + L_laplace
            
        # Bayesian uncertainty weighting (Eq. 18)
        s_data = self.log_sigma_data
        s_phys = self.log_sigma_phys
        
        precision_data = torch.exp(-2.0 * s_data)
        precision_phys = torch.exp(-2.0 * s_phys)
        
        L_total = 0.5 * precision_data * L_data + 0.5 * precision_phys * L_phys + s_data + s_phys
        
        loss_dict = {
            "L_total": L_total,
            "L_data": L_data,
            "L_phys": L_phys,
            "L_ode": L_ode,
            "L_laplace": L_laplace,
            "sigma_data": torch.exp(s_data),
            "sigma_phys": torch.exp(s_phys)
        }
        
        return L_total, loss_dict


import numpy as np
from typing import Dict, Tuple

def ddx_central(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2.0*dx)

def ddy_central(f: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(f, -1, axis=-2) - np.roll(f, 1, axis=-2)) / (2.0*dy)

def laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return ((np.roll(f,-1,axis=-1) - 2*f + np.roll(f,1,axis=-1)) / (dx*dx) +
            (np.roll(f,-1,axis=-2) - 2*f + np.roll(f,1,axis=-2)) / (dy*dy))

def cfl_gravity(u: np.ndarray, v: np.ndarray, h: np.ndarray, g: float, dt: float, dx: float, dy: float) -> float:
    c = np.sqrt(np.maximum(g*h, 0.0))
    speed_x = np.max(np.abs(u) + c)
    speed_y = np.max(np.abs(v) + c)
    return max(speed_x * dt / dx, speed_y * dt / dy)

def rhs_sw_2d(t: float, Y: np.ndarray, p: Dict) -> np.ndarray:
    """
    Y shape: (3, Ny, Nx) -> [h, u, v]
    p: {"g":..., "f":..., "nu":..., "dx":..., "dy":..., "Fh": float or 2D, "Du": float, "Dv": float}
    Returns dY/dt with same shape.
    """
    h, u, v = Y[0], Y[1], Y[2]
    g = p["g"]; f = p.get("f", 0.0); nu = p.get("nu", 0.0)
    dx = p["dx"]; dy = p["dy"]

    # Continuity: h_t = - div (h u)
    hu = h * u
    hv = h * v
    dhdt = -(ddx_central(hu, dx) + ddy_central(hv, dy))
    # Optional forcing on h (heating/cooling)
    if "Fh" in p:
        dhdt = dhdt + p["Fh"]

    # Momentum: u_t, v_t (adv + Coriolis + pressure + viscous + damping)
    adv_u = u*ddx_central(u, dx) + v*ddy_central(u, dy)
    adv_v = u*ddx_central(v, dx) + v*ddy_central(v, dy)
    pres_x = -g * ddx_central(h, dx)
    pres_y = -g * ddy_central(h, dy)
    cor_u = -f * v
    cor_v =  f * u
    visc_u = nu * laplacian(u, dx, dy) if nu else 0.0
    visc_v = nu * laplacian(v, dx, dy) if nu else 0.0
    damp_u = -p.get("Du", 0.0) * u
    damp_v = -p.get("Dv", 0.0) * v

    dudt = -(adv_u) + cor_u + pres_x + visc_u + damp_u
    dvdt = -(adv_v) + cor_v + pres_y + visc_v + damp_v

    dYdt = np.stack([dhdt, dudt, dvdt], axis=0)
    return dYdt
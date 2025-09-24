import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from src.models.shallow_water import rhs_sw_2d, cfl_gravity

def make_initial_sw(Ny: int, Nx: int, h0: float = 1.0, jet_amp: float = 0.1) -> np.ndarray:
    y = np.linspace(0.0, 1.0, Ny, endpoint=False)
    x = np.linspace(0.0, 1.0, Nx, endpoint=False)
    X, Y = np.meshgrid(x, y)
    h = h0 + 0.0*X
    u = jet_amp * np.sin(2*np.pi*Y)  # simple zonal jet
    v = 0.0 * X
    return np.stack([h, u, v], axis=0)

def rhs(t, Y, params: Dict):
    return rhs_sw_2d(t, Y, params)

def sw_cfl(Y: np.ndarray, params: Dict, dt: float) -> float:
    h, u, v = Y[0], Y[1], Y[2]
    return cfl_gravity(u, v, h, params["g"], dt, params["dx"], params["dy"])
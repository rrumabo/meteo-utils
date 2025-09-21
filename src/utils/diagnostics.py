#norms, CFL, JSON export

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, Any, List, Iterable
def _normalize_grid(grid: Union[Sequence[int], int]) -> Union[int, List[int]]:
    if isinstance(grid, (int, np.integer)):
        return int(grid)
    if isinstance(grid, (list, tuple, np.ndarray)):
        return [int(x) for x in list(grid)]
    try:
        return [int(x) for x in grid]  # type: ignore[arg-type]
    except Exception as e:
        raise TypeError(f"Unsupported grid type: {type(grid)!r}") from e


import numpy as np
import numpy.typing as npt


def compute_norms(u: np.ndarray, dx: float = 1.0, dy: Optional[float] = None) -> Dict[str, float]:
    """
    Compute L1, L2, Linf norms for 1D or 2D arrays.
    For 1D: measure = dx
    For 2D: measure = dx * dy
    """
    uabs = np.abs(u)
    if u.ndim == 1:
        measure = dx
        l1 = measure * np.sum(uabs)
        l2 = math.sqrt(measure * np.sum(u * u))
    elif u.ndim == 2:
        if dy is None:
            dy = dx
        measure = dx * dy
        l1 = measure * np.sum(uabs)
        l2 = math.sqrt(measure * np.sum(u * u))
    else:
        raise ValueError("compute_norms expects 1D or 2D array")
    linf = float(np.max(uabs))
    return {"L1": float(l1), "L2": float(l2), "Linf": float(linf)}


def cfl_number_advection(dt: float, dx: float, a: Optional[float] = None, u: Optional[npt.ArrayLike] = None) -> float:
    """
    CFL for advection-type terms: |a|max * dt/dx.
    Provide either a constant speed 'a' or a field 'u' (uses max|u|).
    """
    if a is None:
        if u is None:
            raise ValueError("Provide 'a' (scalar) or 'u' (array) to compute advection CFL.")
        u_arr = np.asarray(u)
        vmax = float(np.max(np.abs(u_arr)))
    else:
        vmax = float(abs(a))
    return vmax * dt / dx


def cfl_number_diffusion(dt: float, dx: float, nu: float, dim: int = 1) -> float:
    """
    Diffusion stability number ~ nu * dt / dx^2 (1D).
    For equal grid spacing, a common conservative check is:
      1D: nu*dt/dx^2 <= 0.5 (explicit FTCS)
      2D: 2*nu*dt/dx^2 <= 0.5  -> nu*dt/dx^2 <= 0.25
    We return the 'raw' factor; the caller decides the threshold.
    """
    if dim not in (1, 2):
        raise ValueError("dim must be 1 or 2")
    factor = nu * dt / (dx * dx)
    return factor if dim == 1 else 2.0 * factor


def write_metrics_json(
    out_dir: Path | str,
    scheme: str,
    grid: Union[Sequence[int], int],
    dt: float,
    cfl: Dict[str, float],
    norms_over_time: Dict[str, Iterable[float]],
    extras: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Persist a compact metrics.json into the run output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "scheme": scheme,
        "grid": _normalize_grid(grid),
        "dt": dt,
        "cfl": {k: float(v) for k, v in cfl.items()},
        "norms": {k: list(map(float, v)) for k, v in norms_over_time.items()},
    }
    if extras:
        payload["extras"] = {k: float(v) for k, v in extras.items()}

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return metrics_path
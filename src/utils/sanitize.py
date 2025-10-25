# src/utils/sanitize.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Mapping
import numpy as np

def _ensure_params(params_or_dx: Any, args: tuple) -> Dict[str, float]:
    """
    Accept either a params dict {dx,dy,g} *or* positional (dx, dy, g).
    Returns a normalized dict with float values.
    """
    if isinstance(params_or_dx, Mapping):
        p = params_or_dx
        return {"dx": float(p["dx"]), "dy": float(p["dy"]), "g": float(p["g"])}
    else:
        if len(args) < 2:
            raise TypeError("sanitize_ic_and_dt expected params dict or dx, dy, g (positional).")
        dx = float(params_or_dx); dy = float(args[0]); g = float(args[1])
        return {"dx": dx, "dy": dy, "g": g}


def sanitize_ic_and_dt(
    Y0: np.ndarray,
    params_or_dx: Any,
    *args,
    dt_request: float | None = None,
    cfl_target: float = 0.20,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Flexible API to avoid call-site churn while we stabilize the CLI.

    Supported call signatures:
      1) sanitize_ic_and_dt(Y0, params_dict, [dt_request], [cfl_target])
      2) sanitize_ic_and_dt(Y0, dx, dy, g, [dt_request], [cfl_target])

    Returns: (Y0_sanitized, dt_main, spinup_info)
    - Y0_sanitized: finite h,u,v with h clipped to [1e-6, 10]
    - dt_main: CFL-safe step (min of requested and computed)
    - spinup_info: tiny very-stable spinup step {'T', 'dt'}
    """
    # If called with a params dict, extra positional args may be dt_request, cfl_target
    if isinstance(params_or_dx, Mapping):
        extra = list(args)
        if extra and dt_request is None:
            dt_request = float(extra.pop(0))
        if extra:
            cfl_target = float(extra.pop(0))

    # Normalize params (handles both dict and numeric triple)
    p = _ensure_params(params_or_dx, args)

    # If called with numeric (dx,dy,g), remaining extras are [dt_request, cfl_target]
    if not isinstance(params_or_dx, Mapping):
        tail = list(args[2:]) if len(args) >= 2 else []
        if tail and dt_request is None:
            dt_request = float(tail.pop(0))
        if tail:
            cfl_target = float(tail.pop(0))

    # Sanitize initial condition
    Y0 = np.asarray(Y0, dtype=np.float64)
    h, u, v = Y0[0].copy(), Y0[1].copy(), Y0[2].copy()

    h = np.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)
    h = np.clip(h, 1e-6, 10.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    Y0_s = Y0.copy()
    Y0_s[0], Y0_s[1], Y0_s[2] = h, u, v

    # CFL-based dt
    dx_min = min(p["dx"], p["dy"])
    c = np.sqrt(p["g"] * h)
    umax = float(np.nanmax(np.abs(u)))
    cmax = float(np.nanmax(c))
    vmax = umax + cmax
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.sqrt(p["g"] * float(np.nanmean(h))))

    dt_cfl = float(cfl_target * dx_min / max(vmax, 1e-12))
    dt_main = float(min(dt_request, dt_cfl)) if dt_request is not None else dt_cfl

    # Conservative spin-up (highly stable tiny step)
    spinup = {"T": 0.02, "dt": max(1e-6, 0.02 * dt_cfl)}

    return Y0_s, dt_main, spinup

__all__ = ["sanitize_ic_and_dt"]
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from src.core import time_integrators as ti
from src.utils.diagnostic_manager import DiagnosticManager
from src.utils.diagnostics import (
    compute_norms,
    cfl_number_advection,
    cfl_number_diffusion,
    write_metrics_json,
)


@dataclass
class Solution:
    t: np.ndarray           # shape (Ns,)
    y: np.ndarray           # shape (Ns, dim) or (Ns, ...) for PDE states
    meta: Dict[str, Any]    # method, dt, steps, elapsed, user records


def _select_stepper(method: str):
    """
    Return a function with signature: (f, t, y, dt, params) -> y_next
    by adapting integrators that use: step(y, rhs(u,t), t, dt, diagnostics_fn=None).
    """
    name = method.strip().lower()

    def make_adapter(step_fn):
        def _adapt(f, t, y, dt, params):
            # bind params and flip argument order for integrators: rhs(u, t)
            def rhs(u, tt):
                return f(tt, u, params)
            return step_fn(y, rhs, t, dt, None)
        return _adapt

    if name == "euler":
        base = getattr(ti, "euler_step", None)
    elif name in ("rk2", "heun"):
        base = getattr(ti, "rk2_step", None)
    elif name == "rk4":
        base = getattr(ti, "rk4_step", None)
    else:
        raise ValueError("Unknown method. Use: euler, rk2, heun, rk4.")
    if base is None:
        raise ValueError(f"Integrator for '{method}' not found in time_integrators.py")

    return make_adapter(base)

def solve_fixed_step(
    f: Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray],
    *,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    dt: float,
    method: str = "rk4",
    params: Optional[Dict[str, Any]] = None,
    save_every: int = 1,
    callbacks: Optional[Iterable[Callable[[float, np.ndarray, Dict[str, Any]], Optional[bool]]]] = None,
    diagnostics: Optional[DiagnosticManager] = None,
    metrics_out_dir: Optional[Path] = None,
    norm_grid: Optional[Tuple[float, Optional[float]]] = None,
    cfl_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    norm_save_every: int = 1,
) -> Solution:
    """
    Fixed-step time integration for ODEs or semi-discrete PDEs (Method of Lines).
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 <= t0:
        raise ValueError("t_span must satisfy t1 > t0")

    stepper = _select_stepper(method)
    p = params or {}
    cbs = list(callbacks) if callbacks is not None else []

    # Ensure arrays
    y = np.array(y0, dtype=float, copy=True)
    t = t0

    # Diagnostics
    dm = diagnostics or DiagnosticManager()
    dm.reset()
    dm.start()

    # Preallocate conservative upper bound for number of saves
    n_steps_total = int(np.ceil((t1 - t0) / dt))
    n_saves_est = n_steps_total // save_every + 2  # (not strictly needed, kept for clarity)

    t_hist: List[float] = [t]
    y_hist: List[np.ndarray] = [np.array(y, copy=True)]

    # --- Norms and metrics initialization ---
    l2_hist: List[float] = []
    dx_val: Optional[float] = None
    dy_val: Optional[float] = None
    if norm_grid is not None:
        dx_val = float(norm_grid[0])
        dy_val = None if len(norm_grid) == 1 or norm_grid[1] is None else float(norm_grid[1])
        try:
            n0 = compute_norms(y, dx=dx_val, dy=dy_val)
            l2_hist.append(n0["L2"]) 
        except Exception:
            pass

    steps = 0
    while t < t1 - 1e-15:
        # Adjust last dt to land exactly on t1
        dt_eff = min(dt, t1 - t)

        # Advance one step
        y = stepper(f, t, y, dt_eff, p)
        t = t + dt_eff
        steps += 1
        dm.tick()

        # Save history
        if (steps % save_every) == 0 or t >= t1 - 1e-15:
            t_hist.append(t)
            y_hist.append(np.array(y, copy=True))
            if norm_grid is not None and (steps % norm_save_every == 0 or t >= t1 - 1e-15):
                try:
                    nnow = compute_norms(y, dx=dx_val if dx_val is not None else 1.0, dy=dy_val)
                    l2_hist.append(nnow["L2"]) 
                except Exception:
                    pass

        # Callbacks can stop early
        if cbs:
            meta = {"step": steps, "t": t, "dt": dt_eff, "method": method}
            if any(bool(cb(t, y, meta)) for cb in cbs):
                break

    dm.stop()

    # --- Metrics output and CFL computation ---
    cfls: Dict[str, float] = {}
    if cfl_specs:
        for key, spec in cfl_specs.items():
            stype = spec.get("type", "diffusion")
            if stype == "advection":
                cfls[key] = float(
                    cfl_number_advection(
                        dt=float(spec["dt"]),
                        dx=float(spec["dx"]),
                        a=spec.get("a"),
                        u=spec.get("u"),
                    )
                )
            elif stype == "diffusion":
                cfls[key] = float(
                    cfl_number_diffusion(
                        dt=float(spec["dt"]),
                        dx=float(spec["dx"]),
                        nu=float(spec["nu"]),
                        dim=int(spec.get("dim", 1)),
                    )
                )
            elif "value" in spec:
                cfls[key] = float(spec["value"])

    if metrics_out_dir is not None:
        last_state = y_hist[-1]
        if isinstance(last_state, np.ndarray):
            if last_state.ndim == 1:
                grid_info = last_state.shape[0]
            elif last_state.ndim == 2:
                grid_info = list(last_state.shape)
            else:
                grid_info = int(last_state.size)
        else:
            grid_info = 0
        try:
            write_metrics_json(
                out_dir=metrics_out_dir,
                scheme=method,
                grid=grid_info,
                dt=float(dt),
                cfl=cfls,
                norms_over_time={"L2": l2_hist} if l2_hist else {},
                extras={"elapsed_s": float(dm.summary().get("elapsed_s", 0.0))},
            )
        except Exception:
            pass

    T = np.asarray(t_hist, dtype=float)
    Y = np.stack(y_hist, axis=0)  # shape (Ns, ...)

    meta = {
        "method": method,
        "dt": float(dt),
        "steps": steps,
        **dm.summary(),
        "norms": {"L2": l2_hist} if l2_hist else {},
        "cfl": cfls,
    }
    return Solution(t=T, y=Y, meta=meta)
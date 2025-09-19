from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
import numpy as np

from de_solver.core import time_integrators as ti
from de_solver.utils.diagnostic_manager import DiagnosticManager


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

        # Callbacks can stop early
        if cbs:
            meta = {"step": steps, "t": t, "dt": dt_eff, "method": method}
            if any(bool(cb(t, y, meta)) for cb in cbs):
                break

    dm.stop()

    T = np.asarray(t_hist, dtype=float)
    Y = np.stack(y_hist, axis=0)  # shape (Ns, ...)

    meta = {
        "method": method,
        "dt": float(dt),
        "steps": steps,
        **dm.summary(),
    }
    return Solution(t=T, y=Y, meta=meta)
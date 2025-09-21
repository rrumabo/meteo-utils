import numpy as np
from src.core.operators import apply_op


def run_heat_solver_2d(L_op, u0, Nx, Ny, T, dt, dx=None, dy=None, step_func=None, rhs_func=None):
    """2D heat solver; supports operator-aware and rhs-based steppers."""
    if step_func is None:
        raise ValueError("step_func must be provided (e.g. rk4_step_op or rk4_step)")

    u = u0.copy()
    u_history = [u.copy()]
    t = 0.0
    steps = int(np.ceil(T / dt)) if T > 0 else 0

    def _rhs(u_in, t_in):
        lin = apply_op(L_op, u_in)
        if rhs_func is not None:
            lin = lin + rhs_func(u_in, t_in)
        return lin

    for _ in range(steps):
        try:
            # operator-aware: step(u, *, t, dt, L_op, rhs_func=None)
            u = step_func(u=u, t=t, dt=dt, L_op=L_op, rhs_func=rhs_func)
        except TypeError:
            # rhs-based: step(u, rhs_func, t, dt)
            u = step_func(u, _rhs, t, dt)
        u_history.append(u.copy())
        t += dt

    return u_history
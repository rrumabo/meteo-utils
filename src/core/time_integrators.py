import numpy as np
from scipy.sparse.linalg import splu, SuperLU
from scipy.sparse import issparse, csc_matrix, identity
from typing import Optional, TypedDict, cast

# Explicit Euler (rhs-based)
def euler_step(u, rhs_func, t, dt, diagnostics_fn=None):
    if diagnostics_fn: diagnostics_fn(u, t)
    return u + dt * rhs_func(u, t)

# Classical RK4 (rhs-based)
def rk4_step(u, rhs_func, t, dt, diagnostics_fn=None):
    if diagnostics_fn: diagnostics_fn(u, t)
    k1 = rhs_func(u, t)
    k2 = rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_func(u + dt * k3, t + dt)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Operator-aware Euler: du/dt = L u + f(u,t)
def euler_step_op(u, *, t, dt, L_op, rhs_func=None, diagnostics_fn=None):
    if diagnostics_fn: diagnostics_fn(u, t)
    rhs = L_op @ u
    if rhs_func is not None:
        rhs = rhs + rhs_func(u, t)
    return u + dt * rhs

# Operator-aware RK4: du/dt = L u + f(u,t)
def rk4_step_op(u, *, t, dt, L_op, rhs_func=None, diagnostics_fn=None):
    if diagnostics_fn: diagnostics_fn(u, t)
    def F(s, tau):
        r = L_op @ s
        if rhs_func is not None:
            r = r + rhs_func(s, tau)
        return r
    k1 = F(u, t)
    k2 = F(u + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = F(u + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = F(u + dt * k3,      t + dt)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Crank–Nicolson factory for du/dt = L u (dt-aware cached LU)
def make_crank_nicolson_step(L_op):
    L_csc_any = L_op if (issparse(L_op) and getattr(L_op, "format", None) == "csc") else csc_matrix(L_op)

    # Validate operator shape early (helps when a tuple or 1D array is passed by mistake)
    shape = getattr(L_csc_any, "shape", None)
    if shape is None:
        raise TypeError(f"make_crank_nicolson_step: L_op of type {type(L_op).__name__} has no 'shape' attribute")
    if not (len(shape) == 2):
        raise ValueError(f"make_crank_nicolson_step: L_op must be 2D, got shape={shape}")
    nrows, ncols = int(shape[0]), int(shape[1])
    if nrows != ncols:
        raise ValueError(f"make_crank_nicolson_step: L_op must be square, got {nrows}x{ncols}")

    L_csc: csc_matrix = cast(csc_matrix, L_csc_any)
    N = nrows

    class _CNCache(TypedDict):
        dt: Optional[float]
        solver: Optional[SuperLU]
        B: Optional[csc_matrix]

    cached: _CNCache = {"dt": None, "solver": None, "B": None}
    def _prepare(dt):
        if cached["dt"] == dt and cached["solver"] is not None: return
        I = identity(N, format="csc")
        A = (I - 0.5 * dt * L_csc).tocsc()
        B = (I + 0.5 * dt * L_csc).tocsc()
        cached["solver"] = splu(A)
        cached["B"] = B
        cached["dt"] = dt
    def step(u, rhs_func, t, dt):
        _prepare(dt)
        u = np.asarray(u).reshape(-1)
        B_opt = cached["B"]
        solver_opt = cached["solver"]
        if B_opt is None or solver_opt is None:
            raise RuntimeError("Crank–Nicolson cache not prepared; call _prepare(dt) first")
        B = cast(csc_matrix, B_opt)
        solver = cast(SuperLU, solver_opt)
        rhs = B @ u
        return solver.solve(rhs)
    return step

# RK4 with post-step renormalization (e.g., NLSE)
def make_renormalized_rk4_step(norm_func):
    def step(u, rhs_func, t, dt, diagnostics_fn=None):
        if diagnostics_fn: diagnostics_fn(u, t)
        k1 = rhs_func(u, t)
        k2 = rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs_func(u + dt * k3, t + dt)
        u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        nrm = norm_func(u_next)
        if nrm > 0: u_next = u_next / nrm
        return u_next
    return step

# Forward Euler in imaginary time (optional renormalization)
def make_ite_step(norm_func=None):
    def step(u, rhs_func, t, dt, diagnostics_fn=None):
        if diagnostics_fn: diagnostics_fn(u, t)
        u_next = u - dt * rhs_func(u, t)
        if norm_func is not None:
            nrm = norm_func(u_next)
            if nrm > 0: u_next = u_next / nrm
        return u_next
    return step

# RK4 in imaginary time with renormalization
def make_renormalized_rk4_step_ite(norm_func):
    def step(u, rhs_func, t, dt, diagnostic_manager=None):
        k1 = -rhs_func(u, t)
        k2 = -rhs_func(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = -rhs_func(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = -rhs_func(u + dt * k3, t + dt)
        u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        nrm = norm_func(u_next)
        if nrm != 0: u_next = u_next / nrm
        if diagnostic_manager is not None:
            residual = float(np.linalg.norm(u_next - u))
            diagnostic_manager.track_step(u_next, t + dt, residual=residual)
        return u_next
    return step
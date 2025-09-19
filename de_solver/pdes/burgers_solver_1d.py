import numpy as np

# 1D viscous Burgers: u_t + (1/2 u^2)_x = nu * u_xx
# Discretization:
#   • Advection: local Lax–Friedrichs (Rusanov) numerical flux
#   • Diffusion: either provided Laplacian operator L_op (preferred)
#                or centered second-difference if L_op is None
# Notes:
#   • Works with periodic or simple reflective boundary (see below).
#   • Designed to be drop-in with your explicit time steppers:
#       step(u, rhs, t, dt) -> u_next
# ==================================================

def burgers_rhs_llf(u: np.ndarray, dx: float, nu: float, bc: str = "periodic", L_op=None) -> np.ndarray:
    """
    Compute RHS of 1D Burgers with LLF flux + diffusion.

    Parameters
    ----------
    u : np.ndarray
        State array (shape: [N]).
    dx : float
        Grid spacing.
    nu : float
        Viscosity coefficient.
    bc : {"periodic", "reflect"}
        Boundary condition. "periodic" is recommended.
    L_op : scipy.sparse.spmatrix or ndarray, optional
        Discrete Laplacian operator. If provided, diffusion uses L_op @ u.

    Returns
    -------
    np.ndarray
        Time derivative du/dt.
    """
    bc = str(bc).lower()

    # ---------- Advection term:  -(F_{i+1/2} - F_{i-1/2}) / dx,  with F(u)=0.5*u^2 ----------
    if bc == "periodic":
        up = np.roll(u, -1)                  # u_{i+1}
        F  = 0.5 * u * u
        Fp = 0.5 * up * up
        a  = np.maximum(np.abs(u), np.abs(up))
        F_iphalf = 0.5 * (F + Fp) - 0.5 * a * (up - u)
        F_imhalf = np.roll(F_iphalf, 1)
    else:
        # "reflect" fallback: copy-edge ghosts (simple, dissipative)
        up = np.r_[u[1:], u[-1]]
        F  = 0.5 * u * u
        Fp = 0.5 * up * up
        a  = np.maximum(np.abs(u), np.abs(up))
        F_iphalf = 0.5 * (F + Fp) - 0.5 * a * (up - u)
        F_imhalf = np.r_[F_iphalf[0], F_iphalf[:-1]]

    adv = -(F_iphalf - F_imhalf) / dx

    # ---------- Diffusion term: nu * u_xx ----------
    if L_op is not None:
        diff = nu * (L_op @ u)
    else:
        if bc == "periodic":
            um = np.roll(u, 1)
            up = np.roll(u, -1)
        else:
            um = np.r_[u[0], u[:-1]]
            up = np.r_[u[1:], u[-1]]
        diff = nu * (up - 2.0 * u + um) / (dx * dx)

    return adv + diff


# --------------------------------------------------
# Minimal explicit driver compatible with step_func
# --------------------------------------------------
# step_func must have signature: step(u, rhs, t, dt) -> u_next

def run_burgers_solver_1d(u0, T, dt, dx, nu=0.01, bc: str = "periodic", step_func=None, L_op=None, return_times=True):
    """
    Time-integrate 1D Burgers from u0 to time T.

    Parameters
    ----------
    u0 : array_like
        Initial condition (length N).
    T : float
        Final time.
    dt : float
        Time step.
    dx : float
        Grid spacing.
    nu : float
        Viscosity.
    bc : str
        Boundary condition ("periodic" or "reflect").
    step_func : callable
        Explicit time stepper: step(u, rhs, t, dt) -> u_next
    L_op : optional
        Discrete Laplacian to use for diffusion if provided.
    return_times : bool
        If True, return (history, times); else just history.

    Returns
    -------
    (list[np.ndarray], np.ndarray) or list[np.ndarray]
        Solution snapshots and time vector (if return_times).
    """
    if step_func is None:
        raise ValueError("step_func must be provided for time integration")

    def rhs(u, t):
        return burgers_rhs_llf(u, dx=dx, nu=nu, bc=bc, L_op=L_op)

    u = np.array(u0, dtype=float).copy()
    t = 0.0
    out = [u.copy()]
    times = [t]
    nsteps = int(np.ceil(T / dt))

    for _ in range(nsteps):
        u = step_func(u, rhs, t, dt)
        t += dt
        out.append(u.copy())
        times.append(t)

    return (out, np.asarray(times)) if return_times else out


# ----------------------------------------
# Thin class wrapper (keeps old import API)
# ----------------------------------------
class BurgersSolver1D:
    def __init__(self, dx, nu=0.01, bc: str = "periodic", step_func=None, L_op=None):
        """
        Parameters mirror run_burgers_solver_1d, but bound to an instance.
        """
        self.dx = float(dx)
        self.nu = float(nu)
        self.bc = str(bc).lower()
        self.step_func = step_func
        self.L_op = L_op

    def run(self, u0, T, dt):
        return run_burgers_solver_1d(
            u0=u0,
            T=T,
            dt=dt,
            dx=self.dx,
            nu=self.nu,
            bc=self.bc,
            step_func=self.step_func,
            L_op=self.L_op,
            return_times=True,
        )
import os
from typing import Any, Dict, List, Tuple, Union, Optional

import yaml

Number = Union[int, float]
NumOrList = Union[Number, List[Number], Tuple[Number, Number]]

def _as_float_list(v: Any) -> List[float]:
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(v)]

def _as_int_list(v: Any) -> List[int]:
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)]

def _lower_str(x: Any, default: str) -> str:
    return str(x if x is not None else default).strip().lower()

def load_config(path: str = "config.yaml", tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Load + normalize a solver config.
    Accepts both older and newer schemas; supports 1D and 2D.
    Normalized output keys:

      grid: { dim: int, L: float|[float,float], N: int|[int,int] }
      bc:   { type: 'periodic'|'dirichlet'|'neumann' }
      time: { dt: float, T: float }
      physics: { pde: 'heat'|'advection'|'burgers', params: {...} }
      integrator: { method: 'euler'|'rk4'|'euler_op'|'rk4_op' }
      initial_condition: { type: 'gaussian'|..., center: float|[float,float],
                           sigma: float|[float,float], amp: float, periodic_dist: bool }
      io: { outdir: str }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg: Dict[str, Any] = {}

    # ---------- grid ----------
    grid = raw.get("grid", {})
    # allow legacy top-level keys
    dim = grid.get("dim", raw.get("dimension", 1))
    L_in = grid.get("L", raw.get("L", 1.0))
    N_in = grid.get("N", raw.get("N", 256))

    L_list = _as_float_list(L_in)
    N_list = _as_int_list(N_in)

    # infer dimension from lengths if not provided
    if dim is None:
        dim = 2 if max(len(L_list), len(N_list)) >= 2 else 1
    dim = int(dim)

    # normalize L, N scalars vs lists
    if dim == 1:
        L_norm: Union[float, List[float]] = float(L_list[0])
        N_norm: Union[int,   List[int]]   = int(N_list[0])
    elif dim == 2:
        if len(L_list) == 1: L_list = [L_list[0], L_list[0]]
        if len(N_list) == 1: N_list = [N_list[0], N_list[0]]
        L_norm = [float(L_list[0]), float(L_list[1])]
        N_norm = [int(N_list[0]), int(N_list[1])]
    else:
        raise ValueError(f"Only dim=1 or dim=2 supported (got {dim})")

    cfg["grid"] = {"dim": dim, "L": L_norm, "N": N_norm}

    # ---------- boundary conditions ----------
    bc = raw.get("bc", {})
    cfg["bc"] = {"type": _lower_str(bc.get("type", "periodic"), "periodic")}

    # ---------- time ----------
    tm = raw.get("time", {})
    cfg["time"] = {"dt": float(tm.get("dt", 1e-4)), "T": float(tm.get("T", 1e-1))}

    # ---------- physics / PDE ----------
    physics = raw.get("physics", {})
    # accept legacy { "pde": { "type": ... , "params": {...} } } or top-level "pde"
    pde_block = raw.get("pde", {})
    pde_name = physics.get("pde", pde_block.get("type", "heat"))
    pde = _lower_str(pde_name, "heat")
    params = dict(physics.get("params", pde_block.get("params", {})))

    # common legacy passthroughs
    for k in ("alpha", "nu", "c", "velocity", "advection_velocity"):
        if k in raw and k not in params:
            params[k] = raw[k]
    cfg["physics"] = {"pde": pde, "params": params}

    # ---------- integrator ----------
    integ = raw.get("integrator", raw.get("integrator_settings", {}))
    method = _lower_str(integ.get("method", integ.get("name", "rk4")), "rk4")
    # allow a few synonyms
    aliases = {
        "rk4op": "rk4_op", "eulerop": "euler_op",
        "rk-4": "rk4", "rk_4": "rk4"
    }
    method = aliases.get(method.replace("-", "").replace("_", ""), method)
    cfg["integrator"] = {"method": method}

    # ---------- initial condition ----------
    ic_raw = raw.get("initial_condition", raw.get("ic", {}))
    ic_type = _lower_str(ic_raw.get("type", "gaussian"), "gaussian")

    def _to_num_or_list(v: Any) -> Union[float, List[float]]:
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return float(v)

    center = _to_num_or_list(ic_raw.get("center", 0.5))
    sigma  = _to_num_or_list(ic_raw.get("sigma", 0.05))
    amp    = float(ic_raw.get("amp", ic_raw.get("amplitude", 1.0)))
    periodic_dist = bool(ic_raw.get("periodic_dist", True))

    # reduce to scalar in 1D
    if dim == 1:
        if isinstance(center, list): center = float(center[0])
        if isinstance(sigma,  list): sigma  = float(sigma[0])

    cfg["initial_condition"] = {
        "type": ic_type,
        "center": center,
        "sigma": sigma,
        "amp": amp,
        "periodic_dist": periodic_dist,
    }

    # ---------- io ----------
    io = raw.get("io", {})
    cfg["io"] = {"outdir": str(io.get("outdir", "./outputs"))}

    # ---------- final validation ----------
    if "N" not in cfg["grid"] or "L" not in cfg["grid"]:
        raise ValueError("grid must include N and L")

    ic = cfg["initial_condition"]
    if "type" not in ic:
        raise ValueError("initial_condition requires 'type'")
    if "amp" not in ic:
        raise ValueError("initial_condition requires 'amp'")
    if "center" not in ic or "sigma" not in ic:
        raise ValueError("initial_condition requires 'center' and 'sigma'")

    return cfg
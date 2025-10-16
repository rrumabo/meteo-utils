import numpy as np
from typing import Callable, Optional, Union, Dict
from typing import Any, cast
import numpy.typing as npt
from numbers import Real

Scalar = Union[float, int]
Field  = npt.NDArray[np.float64]
MaybeF = Optional[Union[Scalar, Field, Callable[[float], Union[Scalar, Field]]]]

def to_field_or_scalar(x: MaybeF, Ny: int, Nx: int) -> MaybeF:
    if x is None:
        return None
    if callable(x):
        return x
    # Accept only real-valued scalars (int/float/numpy scalars), not complex/strings
    if isinstance(x, Real) and not isinstance(x, bool):
        return float(x)
    arr: npt.NDArray[np.float64] = np.asarray(cast(Any, x), dtype=np.float64)
    if np.iscomplexobj(arr):
        raise TypeError("Forcing fields must be real-valued (no complex dtype).")
    if arr.shape != (Ny, Nx):
        raise ValueError(f"Expected {(Ny, Nx)} got {arr.shape}")
    return arr

def sinusoid_Fh(amplitude: float, period: float, phase: float = 0.0,
                spatial: Optional[Field] = None) -> Callable[[float], Union[float, np.ndarray]]:
    """Fh(t) = A*sin(2π t/period + phase) [× spatial pattern]"""
    if spatial is not None:
        S: npt.NDArray[np.float64] = np.asarray(spatial, dtype=np.float64)
        def Fh(t: float):
            return amplitude * np.sin(2*np.pi*(t/period) + phase) * S
        return Fh
    else:
        return lambda t: amplitude * np.sin(2*np.pi*(t/period) + phase)
    
def make_forcings(Ny: int, Nx: int, cfg: Dict) -> Dict:
    """Normalize {'Fh','Du','Dv'} from scalars/fields/callables."""
    out = {}
    out["Fh"] = to_field_or_scalar(cfg.get("Fh"), Ny, Nx)
    _du = to_field_or_scalar(cfg.get("Du"), Ny, Nx)
    _dv = to_field_or_scalar(cfg.get("Dv"), Ny, Nx)
    out["Du"] = 0.0 if _du is None else _du
    out["Dv"] = 0.0 if _dv is None else _dv
    return out
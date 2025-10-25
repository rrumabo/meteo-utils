from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Sequence, Dict
from typing import Any, Mapping, cast
from dataclasses import is_dataclass, asdict as _dc_asdict
import hashlib

BBox = Dict[str, float]

def _norm_bbox(bbox: Optional[Union[BBox, Sequence[float]]]) -> Optional[BBox]:
    """
    Accept dict with lon0/lon1/lat0/lat1 or a 4-seq [lon0, lon1, lat0, lat1];
    return a dict with those keys (floats). None -> None.
    """
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        # copy + coerce to float
        out: BBox = {
            "lon0": float(bbox["lon0"]),
            "lon1": float(bbox["lon1"]),
            "lat0": float(bbox["lat0"]),
            "lat1": float(bbox["lat1"]),
        }
        return out
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        lon0, lon1, lat0, lat1 = map(float, bbox)
        return {"lon0": lon0, "lon1": lon1, "lat0": lat0, "lat1": lat1}
    raise ValueError("bbox must be a dict {lon0,lon1,lat0,lat1} or a 4-sequence")


def _cache_basename(
    stem: str,
    Nx: int,
    Ny: int,
    bbox: Optional[Union[BBox, Sequence[float]]] = None,
    key: Optional[str] = None,
) -> str:
    """
    Stable short cache filename for a given (file stem, grid, bbox, key).
    Includes an 8-char SHA1 of the identifying stamp to avoid very long names.
    """
    stamp = f"{stem}|{Nx}x{Ny}"
    b = _norm_bbox(bbox)
    if b is not None:
        stamp += f"|{b['lon0']:.3f}|{b['lon1']:.3f}|{b['lat0']:.3f}|{b['lat1']:.3f}"
    if key:
        stamp += f"|{key}"
    h = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{Nx}x{Ny}_{h}.npz"


def find_cache(
    nc_path: Optional[Union[str, Path]],
    bbox: Optional[Union[BBox, Sequence[float]]],
    Nx: int,
    Ny: int,
    cache_dir: Union[str, Path],
    key: Optional[str] = None,
) -> Optional[Path]:
    """
    Return existing cache Path if present, else None.
    Naming is deterministic across runs for the same (file stem, grid, bbox, key).
    """
    stem = Path(nc_path).stem if nc_path else "era5_default"
    cache_dir = Path(cache_dir)
    name = _cache_basename(stem, Nx, Ny, bbox, key)
    p = cache_dir / name
    return p if p.exists() else None



# as_dict utility for serialization/logging
def as_dict(obj: Any) -> dict:
    """
    Lightweight converter for logging/serialization:
    - dataclasses -> dataclasses.asdict
    - mappings (dict-like) -> plain dict
    - objects with __dict__ -> shallow dict of public attrs
    - otherwise: wrap as {'value': obj}
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return _dc_asdict(cast(Any, obj))
    # dataclass *class* (not instance): represent by name
    if is_dataclass(obj) and isinstance(obj, type):
        return {"__dataclass__": getattr(obj, "__name__", str(obj))}
    if isinstance(obj, dict):
        # ensure plain dict, recursively handle dataclass values
        return {k: as_dict(v) for k, v in obj.items()}
    # Mapping but not plain dict
    if isinstance(obj, Mapping):
        return {k: as_dict(v) for k, v in obj.items()}
    # numpy / sequences are left as-is (caller decides)
    if hasattr(obj, "__dict__"):
        # take only public attributes to avoid noise
        return {k: as_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"value": obj}

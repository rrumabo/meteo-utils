from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import xarray as xr  # for type hints only


# Mock data generator (safe, no deps)
def mock_height_anomaly(Ny: int, Nx: int, amp: float = 0.05) -> np.ndarray:
    """Generate a synthetic sinusoidal anomaly for testing."""
    y = np.linspace(0, 1, Ny, endpoint=False)
    x = np.linspace(0, 1, Nx, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return amp * np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)


# ERA5/CMIP helpers (optional; require xarray at call time)
def open_dataset(path: str) -> "xr.Dataset":
    """Open a NetCDF/GRIB dataset with xarray (requires xarray)."""
    try:
        import xarray as xr
    except ModuleNotFoundError as e:
        raise ImportError(
            "xarray is required for open_dataset(). Install with: pip install xarray netCDF4"
        ) from e
    return xr.open_dataset(path)


def subset_to_domain(
    ds: "xr.Dataset", lon0: float, lon1: float, lat0: float, lat1: float
) -> "xr.Dataset":
    """Subset dataset to [lon0, lon1] × [lat0, lat1]."""
    # This will work when ds is an xarray.Dataset; no import needed here.
    return ds.sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))


def interp_to_grid(da: "xr.DataArray", Nx: int, Ny: int) -> np.ndarray:
    """Interpolate a DataArray to (Ny, Nx) grid resolution and return a NumPy array."""
    # Coordinates are assumed to be named 'longitude' and 'latitude'.
    lon_min = float(da.longitude.min())
    lon_max = float(da.longitude.max())
    lat_min = float(da.latitude.min())
    lat_max = float(da.latitude.max())

    lon_new = np.linspace(lon_min, lon_max, Nx)
    lat_new = np.linspace(lat_min, lat_max, Ny)

    da_interp = da.interp(longitude=lon_new, latitude=lat_new)
    return np.asarray(da_interp.values)
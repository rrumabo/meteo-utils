# Meteo Utils — PDE + ML Climate Downscaling Toolkit

Numerical PDE + lightweight ML plumbing for climate & urban microclimate experiments.  
Built incrementally: **core solvers → climate relevance → urban heat island (UHI) → data coupling → ML downscaling.**

---

## What’s in here (v3)

- **Core PDE engine**
  - 1D/2D heat (diffusion), Burgers, **shallow‑water (SW) 1‑layer** with periodic BCs.
  - Time steppers: RK2/RK4 (explicit), diagnostics & early‑abort on NaN/overflow.

- **Urban physics (new in v3)**
  - Minimal **UHI layer**: surface energy balance (toy net radiation, sensible/latent cooling), roughness/drag.
  - Works with 4‑field state **[h, u, v, T]**. Backwards‑compatible with 3‑field **[h, u, v]**.

- **Diagnostics & outputs**
  - **CFL** (diffusion & gravity‑wave) logging; mass conservation; metrics persisted to `metrics.json`.
  - Artifacts: `final_state.npz`, `diag_*.csv`, quick PNGs.

- **I/O & data**
  - `src/io/reanalysis.py`: mock anomaly + **xarray** stubs (ERA5/CMIP ready).
  - `src/io/landuse.py`: tiny land‑use generator (asphalt / mixed / park bands).
  - `scripts/get_era5_cyprus.py`: small ERA5 sample downloader (NetCDF).

- **Notebooks**
  - `notebooks/validation_heat1d.ipynb` — heat‑equation convergence (2nd‑order slope).
  - `notebooks/toy_climate_circulation.ipynb` — SW toy jet / Rossby‑like waves.
  - `notebooks/era5_to_sw.ipynb` — ERA5 slice → SW initialization (mock + real file).
  - `notebooks/uhi_toy.ipynb` — UHI toy with land‑use bands; Cyprus‑like vs Rotterdam‑like sweep.

---

## Install

```bash
# Python 3.10–3.12 recommended
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
# Optional extras for data:
pip install xarray netCDF4 cdsapi
```

> Tip: Large artifacts (`outputs/`, `data/`) are big. Consider Git LFS or add them to `.gitignore`.

---

## 🏃 Quick starts

### 1) Shallow‑water toy (3‑field)

```python
from src.runner import solve_fixed_step
from src.models.sw_adapter import rhs, make_initial_sw
import numpy as np

Ny, Nx = 64, 64
Y0 = make_initial_sw(Ny, Nx, h0=1.0, jet_amp=0.1)
dx, dy = 1.0/Nx, 1.0/Ny
params = {"g": 9.81, "f": 1e-4, "nu": 1e-4, "dx": dx, "dy": dy}

# CFL-safe dt ≈ 0.25*min(dx,dy)/sqrt(g*h)
dt = 0.25 * min(dx,dy) / np.sqrt(params["g"]*Y0[0].mean())

sol = solve_fixed_step(
    f=rhs, t_span=(0.0, 0.5), y0=Y0, dt=dt, method="rk4", params=params,
    metrics_out_dir=None, save_every=50, norm_grid=(dx,dy),
    cfl_specs={"gw":{"type":"advection","dt":dt,"dx":min(dx,dy),
                     "u": np.abs(Y0[1]) + np.sqrt(params["g"]*Y0[0])}}
)
print("Snapshots:", len(sol.t))
```

### 2) UHI toy (4‑field)

```python
# Run the notebook: notebooks/uhi_toy.ipynb
# or, minimal python:
import numpy as np
from src.io.landuse import load_landuse
from src.models.sw_adapter import rhs
from src.runner import solve_fixed_step

Ny, Nx = 128, 192
dx, dy = 1.0/Nx, 1.0/Ny
albedo, veg, rough = load_landuse(Ny, Nx)

Y0 = np.zeros((4, Ny, Nx)); Y0[0] = 1.0
params = {"dx":dx,"dy":dy,"g":9.81,"f":1e-4,"nu":1e-4,
          "kT":1e-3,"Q0":1.0,"day_len":1.0,"C_heat":1.0,"Hc":0.5,"LEc":0.4,
          "albedo":albedo,"veg":veg,"roughness":rough}
dt = 0.25 * min(dx,dy) / np.sqrt(params["g"]*Y0[0].mean())

sol = solve_fixed_step(rhs,(0.0,1.0),Y0,dt,"rk4",params,save_every=200,
                       metrics_out_dir=None,norm_grid=(dx,dy),
                       cfl_specs={"gw":{"type":"advection","dt":dt,"dx":min(dx,dy),
                                        "u": np.abs(Y0[1]) + np.sqrt(params["g"]*Y0[0])}})
```

---

## Notebooks & data tips

- Launch Jupyter **from repo root** (`meteo_utils/`). If you open from `notebooks/`, prepend:
  ```python
  import sys, pathlib
  ROOT = pathlib.Path.cwd()
  if not (ROOT / "src").exists():
      ROOT = ROOT.parent
  sys.path.insert(0, str(ROOT))
  ```

- ERA5 sample file:
  ```bash
  # Configure ~/.cdsapirc once, then:
  python scripts/get_era5_cyprus.py
  # Output → data/era5_cyprus_2020-07-01.nc
  ```
  Open `notebooks/era5_to_sw.ipynb` and point the loader at `data/*.nc`.

---

## Repo layout

```
src/
  core/time_integrators.py         # RK steppers, hooks
  models/
    shallow_water.py               # SW 3-field RHS (+ CFL helpers)
    sw_adapter.py                  # routes 3-field vs 4-field states
    urban_physics.py               # UHI SEB sources & drag
    sw_balance.py                  # geostrophic init, smoothing utils
  io/
    reanalysis.py                  # mock + xarray stubs for ERA5/CMIP
    landuse.py                     # tiny banded land-use generator
  utils/diagnostics.py             # CFL, norms, metrics writing
notebooks/
  validation_heat1d.ipynb
  toy_climate_circulation.ipynb
  era5_to_sw.ipynb
  uhi_toy.ipynb
scripts/
  get_era5_cyprus.py
tests/
  ... (pytest unit tests)
outputs/                           # artifacts (may be gitignored)
```

---

## Metrics & artifacts

Each run can write:
- `metrics.json` — run meta + diagnostics (CFL maxima, mean‑h drift, extras).
- `diag_*.csv` — time series (t, CFL, max|u|, mean h, etc.).
- `final_state.npz` — last `[h,u,v,(T)]`.
- PNGs — quick looks (height/temperature + diagnostics).

---

## Version history & roadmap

- **v0–v1**: Repo scaffolding, 1D/2D heat & Burgers, basic diagnostics.
- **v2**: SW core stabilized; validation notebook; ERA5 stubs.
- **v3 (current)**: UHI toy physics (SEB + drag), land‑use bands, two‑city sweep.
- **Next**
  - **v4**: ERA5/CMIP coupling & unit‑aware T/energy; small region cases.
  - **v5**: Urban datasets (OSM/land cover), roughness/ET maps.
  - **v6–v7**: ML downscaling & bias correction (CNN/FNO), validation vs stations.
  - **v8**: Packaging, docs site, gallery demos.

---

## Tests & CI

```bash
pytest -q                      # run unit tests locally
# CI: .github/workflows/ci.yml runs pytest on push
```

---

## Troubleshooting

- **`ModuleNotFoundError: src` in notebooks** → ensure repo root on `sys.path` (see snippet above).
- **Blow‑ups / NaNs** → reduce `dt` (CFL target ≲ 0.3), add small `nu`, ensure positive `h` & sensible params.
- **ERA5 lon 0..360 vs −180..180** → use the auto‑detection in the notebook cell; convert with `lon % 360` if needed.
- **Large files rejected by GitHub** → use Git LFS or add `outputs/` & `data/` to `.gitignore`.

---

## 📜 License

TBD.

## 🙌 Cite / credit

If you build on this, please link back to this repo. When we stabilize v1.0, we’ll add a proper citation entry.
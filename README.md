# 🌍 Meteo Utils

Numerical PDE + ML framework for climate and atmospheric modeling.  
This project is being built step by step — from core PDE solvers to climate downscaling tools.

---

## 🚀 Features (so far)

- **Core PDE engine**
  - 1D/2D heat equation (diffusion)
  - Burgers’ equation
  - Shallow-water equations (1-layer, periodic BCs)

- **Diagnostics**
  - CFL stability checks (diffusion + gravity waves)
  - Convergence tests (2nd-order accuracy confirmed)
  - Mass conservation tracking

- **Notebooks**
  - `notebooks/validation_heat1d.ipynb` → heat eq convergence demo
  - `notebooks/toy_climate_circulation.ipynb` → shallow-water toy jet / Rossby-like circulation

- **I/O**
  - `src/io/reanalysis.py` with a synthetic anomaly generator
  - ERA5/CMIP stubs (using xarray, optional)

- **Repo setup**
  - Modular code in `src/`
  - Tests in `tests/` (pytest + GitHub Actions CI)
  - Example configs in `examples/`
  - Outputs written to `outputs/`

---

## 📂 Example usage

Run a shallow-water toy circulation (64×64 grid, RK4):

```bash
pip install -r requirements.txt

Then in Python:
from src.runner import solve_fixed_step
from src.models.sw_adapter import rhs, make_initial_sw

Ny, Nx = 64, 64
Y0 = make_initial_sw(Ny, Nx, h0=1.0, jet_amp=0.1)
params = {"g": 9.81, "f": 1e-4, "nu": 1e-4, "dx": 1/Nx, "dy": 1/Ny}
sol = solve_fixed_step(rhs, t_span=(0.0, 0.5), y0=Y0, dt=1e-3,
                       method="rk4", params=params)
print("Simulation finished, snapshots:", len(sol.t))
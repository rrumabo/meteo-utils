# Meteo Utils — Toy PDE & Urban Microclimate Sandbox

Small, testable code for shallow-water PDEs and a minimal Urban Heat Island (UHI) layer, with an optional ERA5 path. Goal: build a reliable backbone for city-scale digital-twin experiments.

## Quick start (no data)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/uhi_toy.ipynb

Outputs → outputs/uhi_toy/ (T_final.png, metrics.json).

Optional ERA5
pip install xarray netCDF4 cdsapi
python scripts/get_era5_cyprus.py   # writes data/era5_cyprus_2020-07-01.nc
jupyter lab notebooks/era5_to_sw.ipynb

What works (v5)
	•	PDE: 1D/2D heat, Burgers, 1-layer shallow-water (periodic), RK2/RK4.
	•	UHI toy: 4-field [h,u,v,T] with simple surface-energy source + drag.
	•	Data: ERA5 stub + tiny Cyprus slice; robust subsetting + normalization.
	•	Diagnostics: CFL guardrails, mass conservation, metrics.json.

Repo: 
src/ (core, models, io, utils)  notebooks/  scripts/  tests/
data/ (gitignored)              outputs/ (gitignored)
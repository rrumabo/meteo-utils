# meteo_utils (v2 in progress)

Run a demo:
```bash
pip install -r requirements.txt
python - <<'PY'
from pathlib import Path
import numpy as np
from src.runner import solve_fixed_step

def lap(u, dx): return (np.roll(u,-1)-2*u+np.roll(u,1))/(dx*dx)
def rhs(t,u,p): return p["nu"]*lap(u,p["dx"])

N=128; L=1.0; dx=L/N; x=np.linspace(0,L,N,endpoint=False)
u0=np.exp(-((x-0.5)**2)/(2*(0.05**2)))
dt=0.0005; T=0.02
solve_fixed_step(rhs, t_span=(0,T), y0=u0, dt=dt, method="rk4",
                 params={"nu":1e-3,"dx":dx},
                 metrics_out_dir=Path("outputs/heat1d"), norm_grid=(dx,None),
                 cfl_specs={"diff":{"type":"diffusion","dt":dt,"dx":dx,"nu":1e-3,"dim":1}})
print("Wrote outputs/heat1d/metrics.json")
PY
from src.utils.diagnostics import cfl_number_diffusion

def test_cfl_heat_stable_threshold():
    dx, dt, nu = 1.0/64, 1e-4, 1e-3
    factor = cfl_number_diffusion(dt=dt, dx=dx, nu=nu, dim=1)
    assert factor <= 0.5

def test_cfl_heat_violation_detectable():
    dx, dt, nu = 1.0/64, 1e-2, 1e-3  # intentionally too large dt
    factor = cfl_number_diffusion(dt=dt, dx=dx, nu=nu, dim=1)
    assert factor > 0.5
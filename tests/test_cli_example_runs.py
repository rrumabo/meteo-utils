import subprocess, sys
from pathlib import Path
import pytest

def test_cli_example_runs_if_present(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = root / "examples" / "heat1d.yaml"
    main = root / "main.py"

    if not (cfg.exists() and main.exists()):
        pytest.skip("CLI example not available yet (no main.py or heat1d.yaml).")

    proc = subprocess.run(
        [sys.executable, str(main), "--config", str(cfg)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
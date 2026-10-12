import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", ["01_quickstart.py", "02_capacity_sweep.py", "04_naive_vs_real.py"])
def test_example_runs(name):
    out = subprocess.run([sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip()

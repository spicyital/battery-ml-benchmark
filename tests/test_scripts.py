import subprocess
import sys
from pathlib import Path


def test_generation_script_runs_without_editable_install(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_synthetic_data.py",
            "--config",
            "configs/quick.yaml",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_init_creates_expected_structure(tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "init"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    assert (tmp_path / "configs" / "experiments" / "dev.yaml").exists()
    assert (tmp_path / "data" / "raw" / ".gitkeep").exists()
    assert (tmp_path / "outputs" / "runs").exists()

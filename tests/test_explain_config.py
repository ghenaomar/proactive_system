from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_explain_config_prints_summary(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "x.yaml"
    cfg.write_text("exp_name: ex\nsystem:\n  seed: 1\n", encoding="utf-8")

    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "explain-config", "--config", str(cfg), "--set", "system.seed=77"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "CONFIG EXPLAIN" in r.stdout
    assert "exp_name: ex" in r.stdout
    assert "seed: 77" in r.stdout


from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_with_set_overrides_seed(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "x.yaml"
    cfg.write_text("exp_name: ovr\nsystem:\n  seed: 1\n", encoding="utf-8")

    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "run", "--config", str(cfg), "--set", "system.seed=42"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    run_dir = [p for p in (tmp_path / "outputs" / "runs").iterdir() if p.is_dir()][0]
    resolved = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    assert int(resolved["system"]["seed"]) == 42

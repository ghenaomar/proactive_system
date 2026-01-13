from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_show_config_prints_resolved_json(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "x.yaml"
    cfg.write_text("exp_name: sc\nsystem:\n  seed: 1\n", encoding="utf-8")

    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "show-config", "--config", str(cfg), "--set", "system.seed=99"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    obj = json.loads(r.stdout)
    assert int(obj["system"]["seed"]) == 99
    prov = obj.get("_provenance", {})
    assert isinstance(prov, dict)

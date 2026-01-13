from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_meta_contains_config_provenance(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "x.yaml"
    cfg.write_text(
        "\n".join(
            [
                "exp_name: prov_test",
                "system:",
                "  seed: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "run", "--config", str(cfg), "--set", "system.seed=42"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    run_dir = [p for p in (tmp_path / "outputs" / "runs").iterdir() if p.is_dir()][0]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    prov = meta.get("config_provenance", {})
    assert isinstance(prov, dict)
    assert prov.get("config_path") is not None
    assert isinstance(prov.get("includes_resolved", []), list)
    overrides = prov.get("overrides_applied", {})
    assert isinstance(overrides, dict)
    assert int(overrides.get("system.seed", -1)) == 42

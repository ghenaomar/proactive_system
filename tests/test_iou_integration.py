from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_3_images(img_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    for i in range(3):
        arr = (np.zeros((16, 20, 3), dtype=np.uint8) + i).astype(np.uint8)
        imageio.imwrite(str(img_dir / ("%03d.png" % i)), arr)


def test_iou_tracker_stable_ids_in_run(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    _make_3_images(img_dir)

    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "dev.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "exp_name: iou_track_test",
                "system:",
                "  seed: 1",
                "  max_frames: 50",
                "  assumed_fps: 30",
                "models:",
                "  person_detector:",
                "    name: fake",
                "    num_detections: 2",
                "  tracker:",
                "    name: iou",
                "    iou_threshold: 0.3",
                "    max_age: 10",
                "    start_id: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    r = subprocess.run(
        [sys.executable, "-m", "proctor_ai", "run", "--config", str(cfg_path), "--input", str(img_dir)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    run_dir = [p for p in (tmp_path / "outputs" / "runs").iterdir() if p.is_dir()][0]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

    tr = meta.get("tracker_report", {})
    assert tr.get("enabled") is True
    assert int(tr.get("unique_track_ids", -1)) == 2

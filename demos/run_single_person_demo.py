from __future__ import annotations

"""
Single-person demo (recorded video):
- Runs the standard proctor_ai pipeline using a dedicated config (single-person, full-frame seat)
- Detects behaviors
- Renders the same "minimal" demo overlay (colors + risk + sidebar/trail) used in the main project

This stays SEPARATE from the exam-hall pipeline: only new config is used.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(p.returncode)
    return p.stdout or ""


def _extract_run_dir(output: str) -> Path | None:
    # cli prints: OK: wrote run to <path>
    m = re.search(r"OK:\s+wrote run to\s+(.+)", output)
    if m:
        return Path(m.group(1).strip()).resolve()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video (.mp4/.avi/...)")
    ap.add_argument(
        "--config",
        default="configs/experiments/single_person_demo.yaml",
        help="Config path (default: configs/experiments/single_person_demo.yaml)",
    )
    ap.add_argument("--max-frames", type=int, default=900, help="Max frames to process/render")
    ap.add_argument("--assumed-fps", type=float, default=30.0, help="Used by the runner when FPS cannot be inferred")
    ap.add_argument("--no-sidebar", action="store_true", help="Disable right sidebar")
    ap.add_argument("--no-trail", action="store_true", help="Disable bbox trail")
    args = ap.parse_args()

    project_root = Path.cwd()
    inp = Path(args.input).resolve()
    cfg = Path(args.config).resolve()

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    if not cfg.exists():
        raise SystemExit(f"Config not found: {cfg}")

    sets: list[str] = []
    if int(args.max_frames) > 0:
        sets.append(f"system.max_frames={int(args.max_frames)}")
    if float(args.assumed_fps) > 0:
        sets.append(f"system.assumed_fps={float(args.assumed_fps)}")
    # Ensure mediapipe enabled + dynamic ROI on
    sets.append("features.mediapipe.enabled=true")
    sets.append("features.mediapipe.dynamic_roi=true")

    # 1) Run pipeline
    cmd_run = [sys.executable, "-m", "proctor_ai", "run", "--config", str(cfg), "--input", str(inp)]
    for s in sets:
        cmd_run += ["--set", s]

    out = _run(cmd_run)
    run_dir = _extract_run_dir(out)
    if run_dir is None:
        # fallback: newest outputs/runs/*
        runs = sorted((project_root / "outputs" / "runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print(out)
            raise SystemExit("Could not determine run_dir from CLI output.")
        run_dir = runs[0].resolve()

    feats = run_dir / "features" / "mediapipe.jsonl"
    if not feats.exists():
        raise SystemExit(f"Missing features file: {feats}")

    # 2) Detect behaviors
    beh_out = run_dir / "events" / "behaviors.jsonl"
    beh_out.parent.mkdir(parents=True, exist_ok=True)

    cmd_beh = [
        sys.executable,
        "tools/detect_behaviors.py",
        "--in",
        str(feats),
        "--out",
        str(beh_out),
        "--allow_pose_turn",
        "--min_face_turn_px", "16.0",
        "--min_face_turn_px_pose", "14.0",
        # Demo-friendly thresholds (more responsive than CCTV defaults)
        "--turn_on", "0.35",
        "--turn_off", "0.25",
        "--turn_enter_ms", "150",
        "--turn_exit_ms", "220",
    ]
    _run(cmd_beh)

    if (not beh_out.exists()) or beh_out.stat().st_size == 0:
        print(f"WARN: behaviors file is empty: {beh_out}")
        print("Tip: run render with --show_scores to verify pose/face signals.")

    # 3) Render "old style" minimal overlay (colored + risk + sidebar/trail)
    out_video = run_dir / "previews" / "demo_annotated.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    cmd_r = [
        sys.executable,
        "tools/render_demo_video.py",
        "--run_dir",
        str(run_dir),
        "--style",
        "minimal",
        "--max_frames",
        str(int(args.max_frames)),
        "--use_last_features",
        "--show_risk_text",
        "--out",
        str(out_video),
    ]
    if not args.no_sidebar:
        cmd_r.append("--sidebar")
    if not args.no_trail:
        cmd_r.append("--trail")

    _run(cmd_r)

    print(f"OK: run_dir={run_dir}")
    print(f"OK: demo_video={out_video}")


if __name__ == "__main__":
    main()

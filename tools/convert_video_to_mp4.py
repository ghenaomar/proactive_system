"""Convert CCTV / vendor video formats (e.g. .dav) to .mp4 using ffmpeg.

Why:
  - Some CCTV formats won't decode via imageio.
  - OpenCV may or may not open them depending on codecs.
  - Converting once to H.264 MP4 makes the rest of the pipeline stable.

Usage:
  python tools/convert_video_to_mp4.py --in data/raw/examhall.dav --out data/raw/examhall.mp4

Notes:
  - Requires `ffmpeg` available on PATH.
  - Audio is optional; this keeps it if present.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input video path (e.g. .dav)")
    ap.add_argument("--out", dest="out", required=True, help="Output .mp4 path")
    ap.add_argument("--crf", type=int, default=23, help="H.264 CRF (lower=better, bigger file). Default 23")
    ap.add_argument("--preset", type=str, default="veryfast", help="ffmpeg preset (ultrafast..slow). Default veryfast")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH. Install ffmpeg, or convert the video to .mp4 manually.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(inp),
        "-c:v",
        "libx264",
        "-preset",
        str(args.preset),
        "-crf",
        str(int(args.crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]

    print("[CMD]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        raise SystemExit(f"ffmpeg failed with code {p.returncode}")

    if not out.exists() or out.stat().st_size <= 0:
        raise SystemExit("Conversion finished but output file is missing/empty.")

    print(f"OK: wrote {out} ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()

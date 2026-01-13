"""Interactive seat-zone picker for exam hall videos.

This tool is meant as a *one-time calibration step per exam hall*.

Usage (video):
  python tools/pick_zones.py --in data/raw/exam_demo.mp4 --out configs/zones/exam_demo_seats_px.yaml --zones 23

Controls (OpenCV window):
  - Drag a rectangle with the mouse.
  - Press ENTER/SPACE to accept.
  - Press 'c' to cancel current selection and redraw.
  - Press ESC to abort early.

Output YAML format:
  zones:
    - id: seat_1
      bbox: [x1, y1, x2, y2]   # pixels
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import yaml


def _read_first_frame(inp: Path) -> Tuple[Any, int, int]:
    p = str(inp)
    # Prefer treating the input as a video first. This is important for
    # vendor CCTV formats such as `.dav` which OpenCV may be able to open,
    # but which won't be recognized by a simple suffix allow-list.
    cap = cv2.VideoCapture(p)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"Failed to read first frame from: {inp}")
    else:
        # Fallback: treat as image.
        frame = cv2.imread(p)
        if frame is None:
            raise SystemExit(f"Failed to read image/video: {inp}")

    h, w = int(frame.shape[0]), int(frame.shape[1])
    return frame, w, h


def _draw_label(img: Any, x: int, y: int, text: str) -> None:
    cv2.putText(
        img,
        text,
        (int(x), int(max(16, y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (int(x), int(max(16, y))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Video (.mp4/...) or image (.png/.jpg/...) path")
    ap.add_argument("--out", dest="out", required=True, help="Output YAML path")
    ap.add_argument("--zones", type=int, default=23, help="Number of zones to draw")
    ap.add_argument("--prefix", type=str, default="seat", help="Zone id prefix")
    ap.add_argument("--resize", type=float, default=1.0, help="Resize factor for UI only (e.g. 0.75)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frame_bgr, w0, h0 = _read_first_frame(inp)

    ui_scale = float(args.resize)
    ui_scale = 1.0 if ui_scale <= 0 else ui_scale
    if abs(ui_scale - 1.0) > 1e-6:
        frame_ui = cv2.resize(frame_bgr, (int(w0 * ui_scale), int(h0 * ui_scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_ui = frame_bgr.copy()

    zones: List[Dict[str, Any]] = []

    # We draw accepted rectangles onto the UI image so the user avoids overlaps.
    ui = frame_ui

    win = "Pick Zones (drag box, ENTER=accept, c=redo, ESC=abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, ui)

    for i in range(int(args.zones)):
        zid = f"{args.prefix}_{i+1}"
        print(f"\n[{i+1}/{int(args.zones)}] Draw zone: {zid}")
        roi = cv2.selectROI(win, ui, showCrosshair=False, fromCenter=False)
        x, y, ww, hh = map(int, roi)

        # ESC aborts: OpenCV returns (0,0,0,0)
        if ww <= 0 or hh <= 0:
            print("Selection cancelled/empty. Stopping early.")
            break

        # Convert UI coords -> original pixels
        x1 = int(round(x / ui_scale))
        y1 = int(round(y / ui_scale))
        x2 = int(round((x + ww) / ui_scale))
        y2 = int(round((y + hh) / ui_scale))

        x1 = max(0, min(w0 - 1, x1))
        y1 = max(0, min(h0 - 1, y1))
        x2 = max(x1 + 1, min(w0, x2))
        y2 = max(y1 + 1, min(h0, y2))

        zones.append({"id": zid, "bbox": [int(x1), int(y1), int(x2), int(y2)]})

        # Draw onto UI preview
        ux1 = int(round(x1 * ui_scale))
        uy1 = int(round(y1 * ui_scale))
        ux2 = int(round(x2 * ui_scale))
        uy2 = int(round(y2 * ui_scale))
        cv2.rectangle(ui, (ux1, uy1), (ux2, uy2), (0, 255, 0), 1)
        _draw_label(ui, ux1 + 2, uy1 - 6, zid)
        cv2.imshow(win, ui)

    cv2.destroyAllWindows()

    payload: Dict[str, Any] = {
        "meta": {
            "source": str(inp),
            "width": int(w0),
            "height": int(h0),
            "zones": int(len(zones)),
        },
        "zones": zones,
    }

    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"\nOK: wrote zones YAML -> {out}")
    print(f"Zones saved: {len(zones)}")


if __name__ == "__main__":
    main()

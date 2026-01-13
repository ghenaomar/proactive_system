from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2


def _zone_id(z: Any) -> Optional[str]:
    if isinstance(z, dict):
        v = z.get("id") or z.get("zone_id")
        return str(v) if v is not None else None
    for name in ("zone_id", "id"):
        if hasattr(z, name):
            v = getattr(z, name)
            return str(v) if v is not None else None
    return None


def _maybe_dict(z: Any) -> Optional[Dict[str, Any]]:
    # pydantic / dataclass-ish
    if hasattr(z, "model_dump"):
        try:
            d = z.model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(z, "__dict__"):
        try:
            d = dict(getattr(z, "__dict__"))
            if isinstance(d, dict) and d:
                return d
        except Exception:
            pass
    return None


def _as_xyxy(v: Any) -> Optional[Tuple[int, int, int, int]]:
    if not (isinstance(v, (list, tuple)) and len(v) == 4):
        return None
    try:
        x1, y1, x2, y2 = int(v[0]), int(v[1]), int(v[2]), int(v[3])
        return x1, y1, x2, y2
    except Exception:
        return None


def _zone_bbox(z: Any) -> Optional[Tuple[int, int, int, int]]:
    # Try common places
    if isinstance(z, dict):
        for k in ("bbox", "xyxy", "bbox_xyxy", "rect", "box"):
            bb = z.get(k)
            xyxy = _as_xyxy(bb)
            if xyxy:
                return xyxy

    for name in ("bbox", "xyxy", "bbox_xyxy", "xyxy_px", "rect", "box", "coords"):
        if hasattr(z, name):
            bb = getattr(z, name)
            xyxy = _as_xyxy(bb)
            if xyxy:
                return xyxy

    # Try methods
    for m in ("to_xyxy", "as_xyxy"):
        if hasattr(z, m):
            try:
                bb = getattr(z, m)()
                xyxy = _as_xyxy(bb)
                if xyxy:
                    return xyxy
            except Exception:
                pass

    # Fallback: scan __dict__ for any 4-list that looks like bbox
    d = _maybe_dict(z)
    if isinstance(d, dict):
        for k, vv in d.items():
            xyxy = _as_xyxy(vv)
            if xyxy:
                return xyxy

    return None


def _clamp_bbox(xyxy: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 + 1:
        x2 = min(w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(h, y1 + 2)
    return x1, y1, x2, y2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/experiments/exam_demo.yaml OR outputs/runs/<id>/config_resolved.json")
    ap.add_argument("--input", required=True, help="video path")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frames_dir", default="", help="optional: save annotated frames as PNGs")
    args = ap.parse_args()

    # Load config (yaml via project loader or json)
    cfg: Dict[str, Any]
    if args.config.endswith(".json"):
        cfg = json.load(open(args.config, "r", encoding="utf-8"))
    else:
        # project config loader expects full cfg dict already inside pipeline normally,
        # but zones loader reads from cfg keys; we load yaml using the project's config system.
        # simplest: read the yaml file text and let existing loader handle inside load_zones(cfg)
        # Project config resolver (current codebase): proctor_ai.config
        from proctor_ai.config import load_config  # type: ignore

        cfg = load_config(Path(args.config))

    from proctor_ai.zones.mapping import load_zones  # type: ignore
    from proctor_ai.students.registry import load_students, zone_to_student_map  # type: ignore

    zones = load_zones(cfg)
    students = load_students(cfg)
    z2s = zone_to_student_map(students)

    # Prepare video IO
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (W, H))

    frames_dir = args.frames_dir.strip()
    if frames_dir:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)

    # Pre-extract zone draw items
    draw_items = []
    for z in zones:
        zid = _zone_id(z)
        if not zid:
            continue
        bb = _zone_bbox(z)
        if not bb:
            continue
        sid = z2s.get(str(zid), None)
        draw_items.append((str(zid), str(sid) if sid is not None else "?", bb))

    n = 0
    while n < int(args.max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # draw all zones (static)
        for zid, sid, bb in draw_items:
            x1, y1, x2, y2 = _clamp_bbox(bb, W, H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"{zid} -> {sid}"
            # label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = x1, max(20, y1 - 6)
            cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 8, ly + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

        if frames_dir:
            cv2.imwrite(str(Path(frames_dir) / f"frame_{n:05d}.png"), frame)

        n += 1

    cap.release()
    writer.release()
    print(f"OK: wrote seat-mapping-only video -> {out_path}")
    if frames_dir:
        print(f"OK: wrote frames -> {frames_dir} (count={n})")


if __name__ == "__main__":
    main()

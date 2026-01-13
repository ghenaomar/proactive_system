from __future__ import annotations

"""Live (webcam) single-person demo: auto bbox + tracking + behavior overlay.

This demo is intentionally separate from the exam-hall pipeline.
It does NOT require manual bbox drawing.

Run:
  python demos/run_single_person_live.py --cam 0

Keys:
  - q or ESC: quit
  - r: toggle recording preview video (mp4)
  - u: unlock (allow re-selecting the biggest person)

Notes:
  - OpenCV captures in BGR; the detector is configured accordingly.
  - By default we "lock" on a track ID once chosen to avoid switching to a
    second person in the background.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# Allow running from repo root without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _bbox_area(bb: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bb
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _behavior_lines(row: Dict[str, Any], *, thr: float) -> list[str]:
    look_down = _safe_float(row.get("look_down_score", 0.0))
    turn_l = _safe_float(row.get("head_turn_left_score", 0.0))
    turn_r = _safe_float(row.get("head_turn_right_score", 0.0))
    hands_up = _safe_float(row.get("hands_up_score", 0.0))
    hand_face = _safe_float(row.get("hand_to_face_score", 0.0))

    flags: list[str] = []
    if look_down >= thr:
        flags.append(f"LOOK_DOWN {look_down:.2f}")
    if turn_l >= thr:
        flags.append(f"TURN_LEFT {turn_l:.2f}")
    if turn_r >= thr:
        flags.append(f"TURN_RIGHT {turn_r:.2f}")
    if hands_up >= thr:
        flags.append(f"HANDS_UP {hands_up:.2f}")
    if hand_face >= thr:
        flags.append(f"HAND_TO_FACE {hand_face:.2f}")

    if not flags:
        flags.append("OK")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    ap.add_argument(
        "--config",
        default="configs/experiments/single_person_demo.yaml",
        help="Config path (default: configs/experiments/single_person_demo.yaml)",
    )
    ap.add_argument("--behavior-threshold", type=float, default=0.60)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--record-dir", default="outputs/live_preview", help="Where to save recorded preview mp4")
    ap.add_argument("--window", default="ProctorAI Live", help="OpenCV window title")

    args = ap.parse_args()

    import cv2  # type: ignore

    from proctor_ai.config.loading import load_config
    from proctor_ai.features.mediapipe_signals import MediaPipeSignals
    from proctor_ai.io.media_reader import Frame
    from proctor_ai.perception.detector.factory import create_person_detector
    from proctor_ai.perception.tracker.factory import create_tracker
    from proctor_ai.pipeline.runner import _clamp_xyxy_to_frame, _expand_xyxy, _upper_body_bbox

    cfg = load_config(Path(args.config), project_root=Path.cwd(), overrides=[])

    # --- Build detector/tracker ---
    models = cfg.get("models", {}) if isinstance(cfg.get("models", {}), dict) else {}

    det_cfg = models.get("person_detector", {}) if isinstance(models.get("person_detector", {}), dict) else {}
    det_cfg = dict(det_cfg)  # copy
    # OpenCV frame is BGR. Let detector convert.
    det_cfg.setdefault("input_color", "bgr")

    trk_cfg = models.get("tracker", {}) if isinstance(models.get("tracker", {}), dict) else {}

    detector = create_person_detector(det_cfg)
    tracker = create_tracker(trk_cfg)

    mp_cfg = cfg.get("features", {}).get("mediapipe", {}) if isinstance(cfg.get("features", {}), dict) else {}
    mp_cfg = dict(mp_cfg) if isinstance(mp_cfg, dict) else {}
    mp_cfg["enabled"] = True

    # ROI controls (same keys as runner)
    dyn_roi_mode = str(mp_cfg.get("dynamic_roi_mode", "upper_body") or "upper_body")
    roi_scale = float(mp_cfg.get("roi_scale", 1.05))
    roi_pad = int(mp_cfg.get("roi_pad", 8))

    # --- Webcam ---
    cap = cv2.VideoCapture(int(args.cam))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        # fallback
        w, h = 1280, 720

    # Recording toggle
    record_dir = Path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    writer: Optional[Any] = None
    recording = False

    # Track lock (avoid switching to a second person)
    locked_tid: Optional[int] = None
    miss_frames = 0
    max_miss_frames = 15

    t0 = time.perf_counter()
    idx = 0

    with MediaPipeSignals(mp_cfg) as mp_ctx:
        while True:
            if args.max_seconds and (time.perf_counter() - t0) >= float(args.max_seconds):
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            ts_ms = int(round(1000.0 * (time.perf_counter() - t0)))

            fr = Frame(index=int(idx), ts_ms=int(ts_ms), image=frame_bgr, shape_hw=(H, W))

            dets = detector.predict(fr)
            tracks = tracker.update(fr, dets) if detector is not None else []

            # Select track for overlay/ROI
            best = None
            if tracks:
                # Keep locked track if present
                if locked_tid is not None:
                    found = None
                    for tr in tracks:
                        if int(getattr(tr, "track_id", -1)) == int(locked_tid):
                            found = tr
                            break
                    if found is not None:
                        best = found
                        miss_frames = 0
                    else:
                        miss_frames += 1
                        if miss_frames > max_miss_frames:
                            locked_tid = None
                            miss_frames = 0

                if best is None:
                    # Choose largest bbox
                    best = max(tracks, key=lambda t: _bbox_area(getattr(t, "bbox_xyxy")))
                    try:
                        locked_tid = int(getattr(best, "track_id"))
                    except Exception:
                        locked_tid = None

            overlay = frame_bgr.copy()

            row: Optional[Dict[str, Any]] = None
            if best is not None:
                bb = tuple(map(float, getattr(best, "bbox_xyxy")))

                # Draw all tracks lightly
                for tr in tracks:
                    tbb = tuple(map(float, getattr(tr, "bbox_xyxy")))
                    x1, y1, x2, y2 = map(int, map(round, tbb))
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)

                # Primary bbox
                x1, y1, x2, y2 = map(int, map(round, bb))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Upper-body ROI (stabilize FaceMesh)
                roi_bb = bb
                if dyn_roi_mode.lower() in ("upper_body", "upper", "head"):
                    roi_bb = _upper_body_bbox(roi_bb, upper_frac=float(mp_cfg.get("dynamic_roi_upper_frac", 0.72)), x_expand_frac=float(mp_cfg.get("dynamic_roi_x_expand_frac", 0.08)))

                roi_bb = _clamp_xyxy_to_frame(roi_bb, (H, W))
                rx1, ry1, rx2, ry2 = (int(round(roi_bb[0])), int(round(roi_bb[1])), int(round(roi_bb[2])), int(round(roi_bb[3])))
                rx1, ry1, rx2, ry2 = _expand_xyxy((rx1, ry1, rx2, ry2), W, H, scale=float(roi_scale), pad=int(roi_pad))

                # Draw ROI
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

                crop_bgr = frame_bgr[ry1:ry2, rx1:rx2]
                if crop_bgr is not None and getattr(crop_bgr, "size", 0) != 0:
                    crop_rgb = crop_bgr[:, :, ::-1]

                    row = mp_ctx.compute_row(
                        roi_rgb=crop_rgb,
                        roi_xyxy=(rx1, ry1, rx2, ry2),
                        student_id="me",
                        zone_id="solo",
                        frame_index=int(idx),
                        ts_ms=int(ts_ms),
                    )

            # Overlay behaviors
            y = 30
            if row is not None:
                lines = _behavior_lines(row, thr=float(args.behavior_threshold))
                for ln in lines[:6]:
                    cv2.putText(overlay, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y += 26
            else:
                cv2.putText(overlay, "NO_PERSON", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # UI: lock status + record status
            cv2.putText(
                overlay,
                f"track={locked_tid if locked_tid is not None else '-'}  rec={'ON' if recording else 'OFF'}",
                (12, H - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Show
            cv2.imshow(str(args.window), overlay)

            # Recording
            if recording:
                if writer is None:
                    out_path = record_dir / f"live_preview_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (W, H))
                try:
                    writer.write(overlay)
                except Exception:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
                    writer = None
            if key == ord("u"):
                locked_tid = None
                miss_frames = 0

            idx += 1

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    try:
        if writer is not None:
            writer.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()

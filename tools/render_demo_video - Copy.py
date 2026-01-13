from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _zone_id(z: Any) -> Optional[str]:
    if isinstance(z, dict):
        v = z.get("id") or z.get("zone_id")
        return str(v) if v is not None else None
    for name in ("zone_id", "id"):
        if hasattr(z, name):
            v = getattr(z, name)
            return str(v) if v is not None else None
    return None


def _coerce_xyxy(bb: Any) -> Optional[Tuple[float, float, float, float]]:
    if bb is None:
        return None

    if is_dataclass(bb):
        try:
            bb = asdict(bb)
        except Exception:
            pass

    if isinstance(bb, dict):
        if "xyxy" in bb:
            return _coerce_xyxy(bb.get("xyxy"))
        if all(k in bb for k in ("x1", "y1", "x2", "y2")):
            try:
                return float(bb["x1"]), float(bb["y1"]), float(bb["x2"]), float(bb["y2"])
            except Exception:
                return None

    if all(hasattr(bb, k) for k in ("x1", "y1", "x2", "y2")):
        try:
            return float(getattr(bb, "x1")), float(getattr(bb, "y1")), float(getattr(bb, "x2")), float(getattr(bb, "y2"))
        except Exception:
            return None

    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        try:
            return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        except Exception:
            return None

    return None


def _xyxy_to_int_px(bb4: Tuple[float, float, float, float], shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    H, W = int(shape_hw[0]), int(shape_hw[1])
    x1, y1, x2, y2 = bb4
    # normalized?
    if (0.0 <= x1 <= 1.0) and (0.0 <= x2 <= 1.0) and (0.0 <= y1 <= 1.0) and (0.0 <= y2 <= 1.0):
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(1, min(int(round(x2)), W))
    y2 = max(1, min(int(round(y2)), H))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return x1, y1, x2, y2


def _zone_bbox_px(z: Any, shape_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    for attr in ("xyxy", "bbox_xyxy", "roi_xyxy", "rect", "bbox"):
        if hasattr(z, attr):
            bb = _coerce_xyxy(getattr(z, attr))
            if bb is not None:
                return _xyxy_to_int_px(bb, shape_hw)
    for m in ("as_xyxy", "to_xyxy", "xyxy_px", "bbox_xyxy", "get_xyxy"):
        if hasattr(z, m) and callable(getattr(z, m)):
            fn = getattr(z, m)
            for args in ((), (shape_hw,), (int(shape_hw[0]), int(shape_hw[1]))):
                try:
                    bb = _coerce_xyxy(fn(*args))
                    if bb is not None:
                        return _xyxy_to_int_px(bb, shape_hw)
                except TypeError:
                    continue
                except Exception:
                    continue
    return None


def _put_label(
    img_bgr,
    x: int,
    y: int,
    text: str,
    *,
    scale: float = 0.55,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x2, y2 = x + tw + 8, y + 6
    y1 = y - th - 8
    x1 = x
    # alpha background
    if alpha > 0:
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0.0, img_bgr)
    cv2.putText(img_bgr, text, (x + 4, y), font, scale, fg, thickness, cv2.LINE_AA)


def _draw_corner_box(
    img_bgr,
    bb: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    *,
    thickness: int = 1,
    corner: int = 12,
) -> None:
    x1, y1, x2, y2 = bb
    corner = max(6, int(corner))
    # top-left
    cv2.line(img_bgr, (x1, y1), (x1 + corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x1, y1), (x1, y1 + corner), color, thickness, cv2.LINE_AA)
    # top-right
    cv2.line(img_bgr, (x2, y1), (x2 - corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x2, y1), (x2, y1 + corner), color, thickness, cv2.LINE_AA)
    # bottom-left
    cv2.line(img_bgr, (x1, y2), (x1 + corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x1, y2), (x1, y2 - corner), color, thickness, cv2.LINE_AA)
    # bottom-right
    cv2.line(img_bgr, (x2, y2), (x2 - corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x2, y2), (x2, y2 - corner), color, thickness, cv2.LINE_AA)


def _risk_to_color(risk: float) -> Tuple[int, int, int]:
    # 0 -> green, 1 -> red
    r = max(0.0, min(1.0, float(risk)))
    g = int(round(255 * (1.0 - r)))
    rr = int(round(255 * r))
    return (0, g, rr)


def _behavior_to_text(beh: str) -> str:
    m = {
        "look_down": "Looking Down",
        "head_turn_left": "Looking Left",
        "head_turn_right": "Looking Right",
        "hand_to_face": "Hand To Face",
    }
    return m.get(beh, beh)


def _parse_beh_list(s: str) -> List[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    return [p for p in parts if p]


def _mmss(ts_ms: int) -> str:
    t = max(0, int(ts_ms))
    s = t // 1000
    return f"{s//60:02d}:{s%60:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a clean/modern annotated demo video from a proctor_ai run.")
    ap.add_argument("--run_dir", required=True, help="outputs/runs/<RUN_ID>")
    ap.add_argument("--out", default=None, help="output mp4 path (default: <run_dir>/debug/demo_annotated.mp4)")
    ap.add_argument("--max_frames", type=int, default=900, help="max frames to render")
    ap.add_argument("--start_frame", type=int, default=0, help="start frame index")
    ap.add_argument("--fps", type=float, default=0.0, help="override output fps (0 => use input fps)")

    ap.add_argument(
        "--style",
        choices=["minimal", "legacy"],
        default="minimal",
        help="minimal = modern clean overlay, legacy = old dense overlay",
    )
    ap.add_argument("--sidebar", action="store_true", help="add right sidebar with last alerts")
    ap.add_argument("--sidebar_w", type=int, default=360)

    ap.add_argument(
        "--behaviors",
        default="look_down,head_turn_left,head_turn_right,hand_to_face",
        help="comma-separated behaviors to display",
    )
    ap.add_argument("--show_scores", action="store_true", help="(legacy) show raw scores per student")
    ap.add_argument("--show_roi", action="store_true", help="(legacy) draw roi_xyxy from features")

    ap.add_argument("--stale_ms", type=int, default=600, help="hide active labels if features stale")
    ap.add_argument("--use_last_features", action="store_true", help="reuse last feature row for HUD if missing")

    # Bbox smoothing (visual)
    ap.add_argument("--bbox_smooth_alpha", type=float, default=0.35, help="EMA alpha for bbox smoothing (0 disables)")

    # Motion trail (helps illustrate tracking in a modern way)
    ap.add_argument("--trail", action="store_true", help="draw a short motion trail for each seat")
    ap.add_argument("--trail_len", type=int, default=18, help="trail length in frames")

    # Risk
    ap.add_argument("--risk_alert", type=float, default=0.65)
    ap.add_argument("--risk_high", type=float, default=0.85)
    ap.add_argument("--w_look", type=float, default=0.35)
    ap.add_argument("--w_turn", type=float, default=0.25)
    ap.add_argument("--w_htf", type=float, default=0.40)
    ap.add_argument("--show_risk_text", action="store_true", help="show numeric risk above bbox")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    meta_path = run_dir / "meta.json"
    cfg_path = run_dir / "config_resolved.json"
    feats_path = run_dir / "features" / "mediapipe.jsonl"
    beh_path = run_dir / "events" / "behaviors.jsonl"

    if not meta_path.exists():
        raise SystemExit(f"meta.json not found: {meta_path}")
    if not cfg_path.exists():
        raise SystemExit(f"config_resolved.json not found: {cfg_path}")
    if not feats_path.exists():
        raise SystemExit(f"features not found: {feats_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    input_path = meta.get("io_report", {}).get("input_path") or cfg.get("input_path") or cfg.get("input", None)
    if not input_path:
        raise SystemExit("Could not infer input video path from meta.json/config_resolved.json")
    input_path = str(input_path)

    # Load zones/students mapping
    from proctor_ai.zones.mapping import load_zones
    from proctor_ai.students.registry import load_students, zone_to_student_map

    zones = load_zones(cfg)
    students = load_students(cfg)
    zone2student = zone_to_student_map(students)

    # Features: frame_index -> student_id -> row
    feat_rows = _read_jsonl(feats_path)
    feats_by_frame: Dict[int, Dict[str, Dict[str, Any]]] = {}
    frame_ts_ms: Dict[int, int] = {}
    for r in feat_rows:
        fi = int(r.get("frame_index", 0))
        sid = str(r.get("student_id", ""))
        if not sid:
            continue
        if fi not in feats_by_frame:
            feats_by_frame[fi] = {}
        feats_by_frame[fi][sid] = r
        if fi not in frame_ts_ms:
            try:
                frame_ts_ms[fi] = int(r.get("ts_ms", fi * 33))
            except Exception:
                frame_ts_ms[fi] = fi * 33

    # Behaviors events
    beh_rows = _read_jsonl(beh_path) if beh_path.exists() else []
    beh_rows.sort(key=lambda x: (int(x.get("ts_ms", 0)), int(x.get("frame_index", 0))))

    show_beh_set = set(_parse_beh_list(args.behaviors))
    all_beh = ["look_down", "head_turn_left", "head_turn_right", "hand_to_face"]

    # Active states per student
    active: Dict[str, Dict[str, bool]] = {}
    for sid in {s.student_id for s in students} | {str(r.get("student_id", "")) for r in beh_rows}:
        if sid:
            active[sid] = {b: False for b in all_beh}

    # Track last feature seen per student (stale gating + HUD)
    last_feat_ts: Dict[str, int] = {}
    last_feat_row: Dict[str, Dict[str, Any]] = {}

    # Visual bbox smoothing per student
    bbox_smooth: Dict[str, Tuple[float, float, float, float]] = {}

    # Motion trail per student (list of recent center points)
    trail: Dict[str, List[Tuple[int, int]]] = {}

    # Video IO
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or in_fps <= 1e-6:
        in_fps = 30.0
    out_fps = float(args.fps) if args.fps and args.fps > 0 else float(in_fps)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W <= 0 or H <= 0:
        ok, fr = cap.read()
        if not ok:
            raise SystemExit("Could not read first frame for size.")
        H, W = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    shape_hw = (H, W)

    # Zone bboxes (for legacy fallback)
    zone_bb: Dict[str, Tuple[int, int, int, int]] = {}
    for z in zones:
        zid = _zone_id(z)
        if not zid:
            continue
        bb = _zone_bbox_px(z, shape_hw)
        if bb is not None:
            zone_bb[zid] = bb

    out_path = Path(args.out) if args.out else (run_dir / "debug" / "demo_annotated.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sidebar_w = int(args.sidebar_w) if args.sidebar else 0
    out_size = (W + sidebar_w, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, out_size)
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for: {out_path}")

    start_frame = max(0, int(args.start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    beh_i = 0
    rendered = 0
    frame_index = start_frame

    # Sidebar alerts
    alerts: List[str] = []
    last_alert_ts: Dict[str, int] = {}
    alert_cooldown_ms = 1200

    def compute_risk_from_active(sid: str) -> float:
        st = active.get(sid, {})
        r = 0.0
        if st.get("look_down", False):
            r += float(args.w_look)
        if st.get("head_turn_left", False) or st.get("head_turn_right", False):
            r += float(args.w_turn)
        if st.get("hand_to_face", False):
            r += float(args.w_htf)
        return max(0.0, min(1.0, r))

    try:
        while rendered < int(args.max_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts_ms = frame_ts_ms.get(frame_index)
            if ts_ms is None:
                ts_ms = int(round(frame_index * 1000.0 / out_fps))

            # Update last feature cache for this frame
            if frame_index in feats_by_frame:
                for sid, row in feats_by_frame[frame_index].items():
                    try:
                        last_feat_ts[sid] = int(row.get("ts_ms", ts_ms))
                    except Exception:
                        last_feat_ts[sid] = ts_ms
                    last_feat_row[sid] = row

            # Update active behaviors up to this time
            while beh_i < len(beh_rows):
                ev = beh_rows[beh_i]
                ev_ts = int(ev.get("ts_ms", 0))
                if ev_ts > ts_ms:
                    break
                sid = str(ev.get("student_id", ""))
                beh = str(ev.get("behavior", ""))
                et = str(ev.get("event_type", ""))
                if sid in active and beh in active[sid] and (beh in show_beh_set):
                    if et == "start":
                        active[sid][beh] = True
                    elif et == "end":
                        active[sid][beh] = False
                beh_i += 1

            if args.style == "legacy":
                # -----------------------------
                # Legacy rendering (old dense HUD)
                # -----------------------------
                for zid, sid in zone2student.items():
                    zid = str(zid)
                    sid = str(sid)
                    bb = zone_bb.get(zid)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb
                    last_ts = last_feat_ts.get(sid)
                    is_fresh = (last_ts is not None) and ((ts_ms - int(last_ts)) <= int(args.stale_ms))

                    st = active.get(sid, {})
                    act_list = [b for b, on in st.items() if on and (b in show_beh_set)]
                    is_active = bool(act_list) and is_fresh

                    color = (0, 255, 255) if is_active else (255, 255, 0)
                    thickness = 3 if is_active else 2
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
                    _put_label(frame_bgr, x1, max(16, y1 + 18), f"{zid} -> {sid}", scale=0.55, fg=(255, 255, 255))

                    r = feats_by_frame.get(frame_index, {}).get(sid)
                    if r is None and args.use_last_features:
                        r = last_feat_row.get(sid)

                    if args.show_roi and r is not None:
                        roi = r.get("roi_xyxy")
                        if isinstance(roi, list) and len(roi) == 4:
                            try:
                                rx1, ry1, rx2, ry2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                                cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                            except Exception:
                                pass

                    y_cursor = y1 + 42
                    if args.show_scores and r is not None:
                        face_ok = bool(r.get("face_detected", False))
                        pose_ok = bool(r.get("pose_detected", False))
                        l = f"{float(r.get('look_down_score', 0.0)):.2f}"
                        hl = f"{float(r.get('head_turn_left_score', 0.0)):.2f}"
                        hr = f"{float(r.get('head_turn_right_score', 0.0)):.2f}"
                        htf = f"{float(r.get('hand_to_face_score', 0.0)):.2f}"
                        _put_label(frame_bgr, x1, y_cursor, f"pose={int(pose_ok)} face={int(face_ok)}", scale=0.48); y_cursor += 18
                        _put_label(frame_bgr, x1, y_cursor, f"look={l}  turnL={hl}  turnR={hr}  htf={htf}", scale=0.48); y_cursor += 18

                    if is_active:
                        _put_label(frame_bgr, x1, y_cursor, "ACTIVE: " + ", ".join(act_list), scale=0.52)

            else:
                # -----------------------------
                # Minimal / modern rendering
                # -----------------------------
                for zid, sid in zone2student.items():
                    zid = str(zid)
                    sid = str(sid)

                    # Feature row for this frame (or last known)
                    r = feats_by_frame.get(frame_index, {}).get(sid)
                    if r is None and args.use_last_features:
                        r = last_feat_row.get(sid)

                    last_ts = last_feat_ts.get(sid)
                    is_fresh = (last_ts is not None) and ((ts_ms - int(last_ts)) <= int(args.stale_ms))

                    # Determine seat bbox (static) and tracker bbox (dynamic)
                    seat_bb: Optional[Tuple[int, int, int, int]] = None
                    track_bb: Optional[Tuple[int, int, int, int]] = None
                    track_stale = False
                    if r is not None:
                        seat = r.get("seat_bbox_xyxy")
                        if isinstance(seat, list) and len(seat) == 4:
                            try:
                                seat_bb = (int(seat[0]), int(seat[1]), int(seat[2]), int(seat[3]))
                            except Exception:
                                seat_bb = None

                        tr_bb = r.get("track_bbox_xyxy")
                        if isinstance(tr_bb, list) and len(tr_bb) == 4:
                            try:
                                track_bb = (int(tr_bb[0]), int(tr_bb[1]), int(tr_bb[2]), int(tr_bb[3]))
                                track_stale = bool(r.get("track_bbox_is_stale", False))
                            except Exception:
                                track_bb = None

                        if seat_bb is None:
                            roi = r.get("roi_xyxy")
                            if isinstance(roi, list) and len(roi) == 4:
                                try:
                                    seat_bb = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
                                except Exception:
                                    seat_bb = None

                    if seat_bb is None:
                        seat_bb = zone_bb.get(zid)
                    if seat_bb is None:
                        continue

                    # Visual smoothing only for the dynamic (track) bbox
                    bb = track_bb if track_bb is not None else seat_bb
                    a = float(args.bbox_smooth_alpha)
                    if (track_bb is not None) and (a > 0.0):
                        prev = bbox_smooth.get(sid)
                        if prev is None:
                            bbox_smooth[sid] = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                        else:
                            bbox_smooth[sid] = (
                                a * float(bb[0]) + (1.0 - a) * float(prev[0]),
                                a * float(bb[1]) + (1.0 - a) * float(prev[1]),
                                a * float(bb[2]) + (1.0 - a) * float(prev[2]),
                                a * float(bb[3]) + (1.0 - a) * float(prev[3]),
                            )
                        bb = _xyxy_to_int_px(bbox_smooth[sid], shape_hw)

                    # Active behavior label
                    st = active.get(sid, {})
                    act_list = [b for b, on in st.items() if on and (b in show_beh_set)]
                    is_active = bool(act_list) and is_fresh

                    # Priority label
                    label_beh = None
                    if "hand_to_face" in act_list:
                        label_beh = "hand_to_face"
                    elif "head_turn_left" in act_list:
                        label_beh = "head_turn_left"
                    elif "head_turn_right" in act_list:
                        label_beh = "head_turn_right"
                    elif "look_down" in act_list:
                        label_beh = "look_down"

                    risk = compute_risk_from_active(sid) if is_fresh else 0.0

                    # Color rules
                    if risk >= float(args.risk_high):
                        color = (0, 0, 255)  # red
                        thick = 2
                    elif is_active:
                        color = (0, 255, 255)  # yellow
                        thick = 1
                    else:
                        color = (0, 200, 0)  # green
                        thick = 1

                    # If the tracker bbox is stale (occluded), draw it more neutrally.
                    if track_bb is not None and track_stale:
                        color = (160, 160, 160)
                        thick = 1

                    # Optional: draw the static seat zone faintly to help orientation.
                    # This keeps the video readable even when the dynamic track jitter crosses zones.
                    if seat_bb is not None:
                        _draw_corner_box(frame_bgr, seat_bb, (55, 55, 55), thickness=1, corner=10)

                    # Motion trail (illustrates tracking / smoothing)
                    if args.trail and (track_bb is not None) and is_fresh:
                        cx = int((bb[0] + bb[2]) * 0.5)
                        cy = int((bb[1] + bb[3]) * 0.5)
                        pts = trail.get(sid, [])
                        pts.append((cx, cy))
                        pts = pts[-max(2, int(args.trail_len)) :]
                        trail[sid] = pts
                        if len(pts) >= 2:
                            for p1, p2 in zip(pts[:-1], pts[1:]):
                                cv2.line(frame_bgr, p1, p2, (70, 70, 70), 1, cv2.LINE_AA)

                    # Draw dynamic box (corner-style)
                    _draw_corner_box(frame_bgr, bb, color, thickness=thick, corner=14)

                    # Seat label always (small)
                    x1, y1, x2, y2 = bb
                    if bool(args.show_risk_text):
                        rt = f"risk {risk:.2f}"
                        # place above bbox if possible
                        ry = max(2, y1 - 22)
                        _put_label(frame_bgr, x1, ry, rt, scale=0.45, fg=(255, 255, 255), bg=_risk_to_color(risk), alpha=0.65)

                    _put_label(frame_bgr, x1, max(14, y1 + 16), f"{zid}", scale=0.5, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.35)

                    # Risk bar
                    if is_fresh:
                        bar_w = max(10, x2 - x1)
                        bar_h = 5
                        fill_w = int(round(bar_w * max(0.0, min(1.0, risk))))
                        bx1, by1 = x1, max(0, y1 - 8)
                        cv2.rectangle(frame_bgr, (bx1, by1), (bx1 + bar_w, by1 + bar_h), (30, 30, 30), -1)
                        cv2.rectangle(frame_bgr, (bx1, by1), (bx1 + fill_w, by1 + bar_h), _risk_to_color(risk), -1)

                    # Show behavior label only when active
                    if is_active and label_beh is not None:
                        txt = _behavior_to_text(label_beh)
                        if risk >= float(args.risk_high):
                            txt = "HIGH RISK: " + txt
                        _put_label(frame_bgr, x1, min(H - 6, y2 - 6), txt, scale=0.52, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.6)

                        # Sidebar alert (on starts / high risk)
                        if args.sidebar and risk >= float(args.risk_alert):
                            last_a = int(last_alert_ts.get(sid, -10_000))
                            if (ts_ms - last_a) >= alert_cooldown_ms:
                                last_alert_ts[sid] = ts_ms
                                entry = f"{_mmss(ts_ms)} - {zid}: {txt} (risk {risk:.2f})"
                                alerts.append(entry)
                                alerts[:] = alerts[-5:]

            # Sidebar rendering
            if args.sidebar:
                canvas = cv2.copyMakeBorder(frame_bgr, 0, 0, 0, sidebar_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # Title
                _put_label(canvas, W + 12, 22, "ALERTS", scale=0.7, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.0)
                y = 56
                for a in alerts[-5:][::-1]:
                    _put_label(canvas, W + 12, y, a, scale=0.48, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.0)
                    y += 22
                _put_label(canvas, 10, 22, f"frame={frame_index}  t={_mmss(ts_ms)}", scale=0.62, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.35)
                writer.write(canvas)
            else:
                _put_label(frame_bgr, 10, 22, f"frame={frame_index}  t={_mmss(ts_ms)}", scale=0.62, fg=(255, 255, 255), bg=(0, 0, 0), alpha=0.35)
                writer.write(frame_bgr)

            rendered += 1
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    print(f"OK: wrote demo video -> {out_path}")
    print(f"Rendered frames: {rendered} (start_frame={start_frame})")
    print(f"Input video: {input_path}")


if __name__ == "__main__":
    main()

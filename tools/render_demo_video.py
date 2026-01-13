"""Professional Demo Video Renderer for Proctor AI.

Features:
    - IDENTITY VERIFICATION: Shows real student names or "UNKNOWN"
    - HIGHLY VISIBLE status indicators on each person
    - Professional color-coded risk system
    - Clear behavior labels

Author: Proctor AI Team
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# COLORS
# =============================================================================

class Colors:
    # Status colors (BGR)
    SAFE = (80, 200, 100)           # Green
    WARNING = (50, 200, 240)        # Yellow  
    ALERT = (50, 150, 255)          # Orange
    DANGER = (60, 60, 240)          # Red
    IDENTIFIED = (230, 180, 60)     # Cyan - identified student
    UNKNOWN = (80, 80, 180)         # Dark red - unknown person
    
    # UI
    BG_DARK = (20, 20, 25)
    BG_PANEL = (30, 30, 35)
    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)
    
    # Glow
    GLOW_GREEN = (100, 255, 100)
    GLOW_RED = (100, 100, 255)


def _risk_color(risk: float) -> Tuple[int, int, int]:
    if risk < 0.3:
        return Colors.SAFE
    elif risk < 0.5:
        return Colors.WARNING
    elif risk < 0.7:
        return Colors.ALERT
    return Colors.DANGER


# =============================================================================
# IDENTITY SYSTEM
# =============================================================================

class SimpleIdentityMatcher:
    """Match faces to enrolled students using embeddings."""
    
    def __init__(self, db_path: str = "data/students"):
        self.db_path = Path(db_path)
        self.students: Dict[str, Dict] = {}  # folder_name -> {name, embedding}
        self.embeddings: Optional[np.ndarray] = None
        self.student_ids: List[str] = []
        self._load()
    
    def _load(self):
        """Load student database."""
        if not self.db_path.exists():
            print(f"Warning: Student database not found at {self.db_path}")
            return
        
        embeddings_list = []
        
        for folder in sorted(self.db_path.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            
            info_path = folder / "info.json"
            emb_path = folder / "embeddings.npy"
            
            if not info_path.exists():
                continue
            
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                name = data.get("name", folder.name)
                student_id = data.get("student_id", folder.name)
                
                self.students[folder.name] = {
                    "name": name,
                    "student_id": student_id,
                }
                
                # Load embeddings if available
                if emb_path.exists():
                    emb = np.load(str(emb_path))
                    if len(emb.shape) == 2:
                        mean_emb = np.mean(emb, axis=0)
                    else:
                        mean_emb = emb
                    # Normalize
                    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
                    embeddings_list.append(mean_emb)
                    self.student_ids.append(folder.name)
                    
            except Exception as e:
                print(f"Error loading {folder}: {e}")
        
        if embeddings_list:
            self.embeddings = np.stack(embeddings_list, axis=0)
            print(f"✓ Loaded {len(self.students)} students, {len(self.student_ids)} with embeddings")
        else:
            print(f"✓ Loaded {len(self.students)} students (no embeddings)")
    
    def get_name(self, folder_name: str) -> str:
        """Get student name by folder name."""
        if folder_name in self.students:
            return self.students[folder_name]["name"]
        return folder_name
    
    def get_all_names(self) -> Dict[str, str]:
        """Get all student names."""
        return {k: v["name"] for k, v in self.students.items()}


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    out.append(json.loads(s))
    return out


def _draw_thick_box(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 6,
    glow: bool = True,
    reduce_glow: bool = False,
) -> None:
    """Draw HIGHLY VISIBLE bounding box with optional reduced glow for overlapping boxes."""
    x1, y1, x2, y2 = bbox
    
    # Glow effect (reduced if overlapping)
    if glow:
        glow_range = 2 if reduce_glow else 4
        for i in range(glow_range, 0, -1):
            glow_color = tuple(max(0, int(c * 0.6)) for c in color)
            cv2.rectangle(img, (x1-i*2, y1-i*2), (x2+i*2, y2+i*2), glow_color, max(1, thickness//2))
    
    # Main box - THICKER
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    # Inner highlight for 3D effect
    highlight = tuple(min(255, c + 60) for c in color)
    cv2.rectangle(img, (x1+4, y1+4), (x2-4, y2-4), highlight, 2, cv2.LINE_AA)


def _draw_label_plate(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.8,
    thickness: int = 2,
    padding: int = 12,
) -> Tuple[int, int]:
    """Draw a BIG label with colored background plate."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    
    # Background plate
    x1, y1 = x, y - th - padding
    x2, y2 = x + tw + padding * 2, y + padding
    
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), Colors.WHITE, 2)
    
    # Text
    cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return (x2 - x1, y2 - y1)


def _draw_status_bar(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    status_text: str,
    risk: float,
    is_identified: bool,
    active_behaviors: List[str] = None,
) -> None:
    """Draw ENHANCED status bar with better behavior visibility."""
    x1, y1, x2, y2 = bbox
    
    # Calculate number of behavior lines
    num_behaviors = len(active_behaviors) if active_behaviors else 0
    bar_height = 35 + (15 * min(num_behaviors, 3))  # Taller for multiple behaviors
    
    # Determine color
    if risk >= 0.5:
        bar_color = _risk_color(risk)
    elif is_identified:
        bar_color = Colors.IDENTIFIED
    else:
        bar_color = Colors.SAFE
    
    # Draw bar background at bottom of bbox
    bar_y1 = y2 - bar_height
    bar_y2 = y2
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, bar_y1), (x2, bar_y2), bar_color, -1)
    cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Status text (main behavior or MONITORING) - LARGER and BOLDER
    text = status_text if status_text else "MONITORING"
    cv2.putText(img, text, (x1 + 10, bar_y1 + 22), font, 0.70, Colors.WHITE, 2, cv2.LINE_AA)
    
    # Show additional active behaviors (up to 2 more lines)
    if active_behaviors and len(active_behaviors) > 1:
        y_offset = bar_y1 + 38
        for beh in active_behaviors[1:3]:  # Show up to 2 more behaviors
            beh_display = beh.replace("_", " ").upper()
            cv2.putText(img, f"+ {beh_display}", (x1 + 10, y_offset), font, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
            y_offset += 15
    
    # Risk indicator on right - LARGER
    risk_text = f"{int(risk*100)}%"
    risk_color = Colors.WHITE if risk < 0.7 else (100, 100, 255)  # Red for high risk
    cv2.putText(img, risk_text, (x2 - 60, bar_y2 - 12), font, 0.75, risk_color, 2, cv2.LINE_AA)


def _draw_name_badge(
    img: np.ndarray,
    name: str,
    is_known: bool,
    bbox: Tuple[int, int, int, int],
) -> None:
    """Draw name badge ABOVE the bbox."""
    x1, y1, x2, y2 = bbox
    
    if is_known:
        bg_color = Colors.IDENTIFIED
        icon = "✓ "
    else:
        bg_color = Colors.UNKNOWN
        icon = "? "
        name = "UNKNOWN"
    
    full_text = f"{icon}{name}"
    
    _draw_label_plate(
        img, full_text,
        (x1, y1 - 5),
        bg_color=bg_color,
        font_scale=0.75,
        thickness=2,
        padding=10,
    )


def _draw_header(
    img: np.ndarray,
    frame_idx: int,
    ts_ms: int,
    detected: int,
    identified: int,
    alerts: int,
    session: str,
) -> None:
    """Draw header bar."""
    H, W = img.shape[:2]
    header_h = 55
    
    cv2.rectangle(img, (0, 0), (W, header_h), Colors.BG_DARK, -1)
    cv2.line(img, (0, header_h), (W, header_h), Colors.GRAY, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 38
    
    # Session name
    cv2.putText(img, session, (20, y), font, 0.8, Colors.WHITE, 2, cv2.LINE_AA)
    
    # Time
    mins, secs = ts_ms // 60000, (ts_ms // 1000) % 60
    cv2.putText(img, f"TIME: {mins:02d}:{secs:02d}", (280, y), font, 0.65, Colors.IDENTIFIED, 2, cv2.LINE_AA)
    
    # Frame
    cv2.putText(img, f"FRAME: {frame_idx}", (480, y), font, 0.55, Colors.GRAY, 1, cv2.LINE_AA)
    
    # Detected
    cv2.putText(img, f"DETECTED: {detected}", (650, y), font, 0.65, Colors.SAFE, 2, cv2.LINE_AA)
    
    # Identified
    cv2.putText(img, f"IDENTIFIED: {identified}", (850, y), font, 0.65, Colors.IDENTIFIED, 2, cv2.LINE_AA)
    
    # Alerts
    alert_color = Colors.DANGER if alerts > 0 else Colors.SAFE
    cv2.putText(img, f"ALERTS: {alerts}", (1070, y), font, 0.7, alert_color, 2, cv2.LINE_AA)


def _draw_sidebar(
    img: np.ndarray,
    known_students: List[Tuple[str, str]],  # (name, status)
    unknown_count: int,
    alerts: List[str],
    width: int,
) -> None:
    """Draw sidebar with student list and alerts."""
    H, W = img.shape[:2]
    sx = W - width
    
    cv2.rectangle(img, (sx, 0), (W, H), Colors.BG_PANEL, -1)
    cv2.line(img, (sx, 0), (sx, H), Colors.GRAY, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = sx + 15
    y = 80
    
    # STUDENTS section
    cv2.putText(img, "STUDENTS IN FRAME", (x, y), font, 0.65, Colors.IDENTIFIED, 2, cv2.LINE_AA)
    y += 35
    
    # Known students
    for name, status in known_students:
        color = Colors.SAFE if status == "OK" else Colors.WARNING
        cv2.putText(img, f"[IDENTIFIED] {name}", (x, y), font, 0.5, color, 1, cv2.LINE_AA)
        y += 25
    
    # Unknown
    if unknown_count > 0:
        cv2.putText(img, f"[UNKNOWN] {unknown_count} person(s)", (x, y), font, 0.5, Colors.UNKNOWN, 1, cv2.LINE_AA)
        y += 25
    
    y += 30
    
    # ALERTS section
    cv2.putText(img, "RECENT ALERTS", (x, y), font, 0.65, Colors.WARNING, 2, cv2.LINE_AA)
    y += 35
    
    for alert in alerts[-8:]:
        display = alert[:38] + "..." if len(alert) > 40 else alert
        cv2.putText(img, f"> {display}", (x, y), font, 0.45, Colors.WHITE, 1, cv2.LINE_AA)
        y += 22


def _calculate_bbox_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU (Intersection over Union) between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def _check_any_overlap(bbox: Tuple[int, int, int, int], other_bboxes: List[Tuple[int, int, int, int]], threshold: float = 0.1) -> bool:
    """Check if bbox overlaps with any other bbox above threshold."""
    for other in other_bboxes:
        if _calculate_bbox_overlap(bbox, other) > threshold:
            return True
    return False


def _behavior_text(beh: str) -> str:
    """Convert behavior name to display text with icon."""
    m = {
        "look_down": "LOOKING DOWN",
        "head_turn_left": "TURNING LEFT",
        "head_turn_right": "TURNING RIGHT",
        "hand_to_face": "HAND TO FACE",
    }
    return m.get(beh, beh.upper())


def _zone_id(z: Any) -> Optional[str]:
    if isinstance(z, dict):
        return str(z.get("id") or z.get("zone_id") or "")
    for attr in ("zone_id", "id"):
        if hasattr(z, attr):
            return str(getattr(z, attr) or "")
    return None


def _zone_bbox(z: Any, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    bb = None
    for attr in ("xyxy", "bbox_xyxy", "bbox", "rect"):
        if hasattr(z, attr):
            bb = getattr(z, attr)
            break
    if bb is None and isinstance(z, dict):
        bb = z.get("bbox") or z.get("xyxy")
    
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        vals = [float(v) for v in bb]
        if all(0 <= v <= 1 for v in vals):
            vals = [vals[0]*W, vals[1]*H, vals[2]*W, vals[3]*H]
        return tuple(int(v) for v in vals)
    return None


# =============================================================================
# MAIN
# =============================================================================

# #region agent log
import json as _json_debug
_DEBUG_LOG_PATH = "/home/user/proactive_system/.cursor/debug.log"
def _debug_log(hyp_id, msg, data=None):
    import time, os
    os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
    with open(_DEBUG_LOG_PATH, "a") as f:
        f.write(_json_debug.dumps({"hypothesisId": hyp_id, "message": msg, "data": data, "timestamp": int(time.time()*1000)}) + "\n")
# #endregion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0)
    ap.add_argument("--sidebar", action="store_true", default=True)
    ap.add_argument("--sidebar_w", type=int, default=350)
    ap.add_argument("--session_name", default="Exam Monitoring")
    ap.add_argument("--behaviors", default="look_down,head_turn_left,head_turn_right,hand_to_face", help="Comma-separated behavior names to show (NOT a file path)")
    ap.add_argument("--behaviors_file", default=None, help="Optional: explicit path to behaviors.jsonl (auto-detected from run_dir)")
    ap.add_argument("--bbox_smooth", type=float, default=0.3)
    ap.add_argument("--stale_ms", type=int, default=1200)
    ap.add_argument("--trail", action="store_true")
    ap.add_argument("--trail_len", type=int, default=15)
    
    # Risk weights
    ap.add_argument("--w_look", type=float, default=0.30)
    ap.add_argument("--w_turn", type=float, default=0.35)
    ap.add_argument("--w_htf", type=float, default=0.35)
    
    args = ap.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    cfg = json.loads((run_dir / "config_resolved.json").read_text())
    
    # #region agent log
    _debug_log("INIT", "render_started", {"run_dir": str(run_dir), "stale_ms": args.stale_ms, "behaviors_arg": args.behaviors})
    # #endregion
    
    input_path = meta.get("io_report", {}).get("input_path") or cfg.get("input")
    
    # Load identity mapping (supports track-based, zone-based, and face-based)
    identity_map_path = run_dir / "identity_mapping.json"
    track_identities = {}  # Maps track/student ID (s1, s2...) to name
    zone_identities = {}   # Maps zone ID (seat_1, seat_2...) to name
    face_positions = {}    # Maps name -> (x, y) for position-based matching
    face_bboxes = {}       # Maps name -> [x1, y1, x2, y2] for direct face drawing!
    
    if identity_map_path.exists():
        with open(identity_map_path, "r") as f:
            id_data = json.load(f)
        
        method = id_data.get("method", "zone-based")
        
        # Face-based mapping (from calibrate_identities.py)
        if "students" in id_data and method == "face-calibration":
            for name, info in id_data["students"].items():
                pos = info.get("position", {})
                if "x" in pos and "y" in pos:
                    face_positions[name] = (pos["x"], pos["y"])
                # Also load face bbox if available!
                fb = info.get("face_bbox")
                if fb and len(fb) == 4:
                    face_bboxes[name] = fb
            print(f"✓ Loaded FACE-based identity mapping: {len(face_positions)} students")
            for name, (x, y) in face_positions.items():
                bbox_str = f", bbox {face_bboxes[name]}" if name in face_bboxes else ""
                print(f"  {name} at ({x:.0f}, {y:.0f}){bbox_str}")
        
        # Track-based mapping (from identify_by_tracks.py)
        if "track_identities" in id_data:
            track_identities = id_data["track_identities"]
            if not face_positions:  # Only print if not using face-based
                print(f"✓ Loaded TRACK-based identity mapping: {len(track_identities)} tracks")
                for tid, info in track_identities.items():
                    name = info.get("name", "UNKNOWN") if isinstance(info, dict) else info
                    conf = info.get("confidence", 0) if isinstance(info, dict) else 0
                    print(f"  {tid} → {name} ({conf:.0%})" if conf else f"  {tid} → {name}")
        
        # Zone-based mapping (fallback)
        if "zone_identities" in id_data:
            zone_identities = id_data["zone_identities"]
            if not track_identities and not face_positions:
                print(f"✓ Loaded ZONE-based identity mapping: {len(zone_identities)} zones")
    else:
        print(f"⚠ No identity mapping found at {identity_map_path}")
        print("  Run: python tools/calibrate_identities.py --video ... --out $RUN_DIR/identity_mapping.json")
    
    # Fallback: load from student database
    identity = SimpleIdentityMatcher("data/students")
    enrolled_names = identity.get_all_names()
    print(f"Enrolled students in database: {list(enrolled_names.values())}")
    
    # Load zones (optional - may not exist in auto_track mode)
    zones = []
    zone2student = {}
    auto_track_mode = cfg.get("auto_track_mode", False)
    
    try:
        from proctor_ai.zones.mapping import load_zones
        from proctor_ai.students.registry import load_students, zone_to_student_map
        
        zones = load_zones(cfg)
        students = load_students(cfg)
        zone2student = zone_to_student_map(students)
    except Exception:
        pass
    
    if auto_track_mode or not zone2student:
        print("✓ Auto-track mode: Processing all detected tracks")
    
    # Load features
    feats_path = run_dir / "features" / "mediapipe.jsonl"
    feat_rows = _read_jsonl(feats_path)
    
    feats_by_frame: Dict[int, Dict[str, Dict]] = {}
    frame_ts: Dict[int, int] = {}
    for r in feat_rows:
        fi = int(r.get("frame_index", 0))
        sid = str(r.get("student_id", ""))
        if sid:
            feats_by_frame.setdefault(fi, {})[sid] = r
            frame_ts[fi] = int(r.get("ts_ms", fi * 33))
    
    # Load behaviors
    # Auto-detect behaviors file (support both new --behaviors_file and old usage)
    if args.behaviors_file:
        beh_path = Path(args.behaviors_file)
    else:
        # Auto-detect from run_dir
        beh_path = run_dir / "events" / "behaviors.jsonl"
    
    beh_rows = _read_jsonl(beh_path)
    beh_rows.sort(key=lambda x: int(x.get("ts_ms", 0)))
    
    # #region agent log
    _debug_log("H2", "behaviors_loaded", {"path": str(beh_path), "exists": beh_path.exists(), "count": len(beh_rows), "sample": beh_rows[:3] if beh_rows else []})
    # #endregion
    
    # Parse behavior names (handle both comma-separated and file paths gracefully)
    behaviors_input = str(args.behaviors).strip()
    
    # If it looks like a file path, use default behaviors instead
    if "/" in behaviors_input or "\\" in behaviors_input or behaviors_input.endswith(".jsonl"):
        behaviors_input = "look_down,head_turn_left,head_turn_right,hand_to_face"
        # #region agent log
        _debug_log("H1_FIX", "behaviors_arg_was_path", {"original": args.behaviors, "using_default": behaviors_input})
        # #endregion
    
    show_behs = set(b.strip() for b in behaviors_input.split(",") if b.strip())
    all_behs = ["look_down", "head_turn_left", "head_turn_right", "hand_to_face"]
    
    # #region agent log
    _debug_log("H1_FIX", "show_behs_parsed", {"show_behs": list(show_behs), "all_behs": all_behs})
    # #endregion
    
    active: Dict[str, Dict[str, bool]] = {}
    for sid in set(str(r.get("student_id", "")) for r in beh_rows) | set(zone2student.values()):
        if sid:
            active[sid] = {b: False for b in all_behs}
    
    # Video
    cap = cv2.VideoCapture(str(input_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out_fps = args.fps if args.fps > 0 else in_fps
    
    zone_bb = {}
    for z in zones:
        zid = _zone_id(z)
        if zid:
            bb = _zone_bbox(z, H, W)
            if bb:
                zone_bb[zid] = bb
    
    # In auto-track mode, build zone2student from features
    if auto_track_mode or not zone2student:
        # Get all unique student IDs from features
        all_sids = set()
        for frame_data in feats_by_frame.values():
            all_sids.update(frame_data.keys())
        
        # Create mapping: each student IS their own zone
        for sid in all_sids:
            zone2student[sid] = sid
            zone2student[f"track_{sid[1:]}" if sid.startswith("s") else sid] = sid
    
    # Output
    sidebar_w = args.sidebar_w if args.sidebar else 0
    out_w, out_h = W + sidebar_w, H
    
    out_path = Path(args.out) if args.out else (run_dir / "demo" / "demo_pro.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
    
    # State
    last_feat_ts: Dict[str, int] = {}
    last_feat_row: Dict[str, Dict] = {}
    bbox_smooth: Dict[str, Tuple[float, ...]] = {}
    trail_pts: Dict[str, List[Tuple[int, int]]] = {}
    alerts: List[str] = []
    last_alert_ts: Dict[str, int] = {}
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    frame_idx = args.start_frame
    beh_i = 0
    rendered = 0
    
    print(f"Rendering {args.max_frames} frames...")
    
    try:
        while rendered < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            
            ts_ms = frame_ts.get(frame_idx, int(frame_idx * 1000 / out_fps))
            
            # Update features
            if frame_idx in feats_by_frame:
                for sid, row in feats_by_frame[frame_idx].items():
                    last_feat_ts[sid] = int(row.get("ts_ms", ts_ms))
                    last_feat_row[sid] = row
            
            # Update behaviors
            while beh_i < len(beh_rows):
                ev = beh_rows[beh_i]
                if int(ev.get("ts_ms", 0)) > ts_ms:
                    break
                sid = str(ev.get("student_id", ""))
                beh = str(ev.get("behavior", ""))
                et = str(ev.get("event_type", ""))
                if sid in active and beh in active[sid]:
                    active[sid][beh] = (et == "start")
                    # #region agent log
                    if frame_idx < 50:  # Log only first 50 frames
                        _debug_log("H1", "behavior_updated", {"frame": frame_idx, "sid": sid, "beh": beh, "et": et, "active_state": dict(active.get(sid, {}))})
                    # #endregion
                else:
                    # #region agent log
                    if frame_idx < 50:
                        _debug_log("H1", "behavior_SKIPPED", {"frame": frame_idx, "sid": sid, "beh": beh, "sid_in_active": sid in active, "beh_in_sid": beh in active.get(sid, {})})
                    # #endregion
                beh_i += 1
            
            # Canvas
            if args.sidebar:
                canvas = cv2.copyMakeBorder(frame, 0, 0, 0, sidebar_w, cv2.BORDER_CONSTANT, value=Colors.BG_PANEL)
            else:
                canvas = frame.copy()
            
            frame_detected = 0
            frame_identified = 0
            frame_alerts = 0
            known_in_frame: List[Tuple[str, str]] = []
            unknown_in_frame = 0
            draw_data: List[Dict[str, Any]] = []
            
            # Process each person (from features, not zones)
            # In auto-track mode, iterate over all students in current frame
            frame_sids = set(feats_by_frame.get(frame_idx, {}).keys()) | set(last_feat_row.keys())
            
            # ONE-TO-ONE face matching (if face_positions available)
            sid_to_name_override = {}
            if face_positions:
                # Collect all tracks with head positions
                track_heads = []
                for sid in frame_sids:
                    r = feats_by_frame.get(frame_idx, {}).get(sid) or last_feat_row.get(sid)
                    if not r:
                        continue
                    last_ts = last_feat_ts.get(sid)
                    if not (last_ts and (ts_ms - last_ts) <= args.stale_ms):
                        continue
                    track_bb = r.get("track_bbox_xyxy")
                    if not (isinstance(track_bb, list) and len(track_bb) == 4):
                        continue
                    # Head position = top 15% of bbox
                    head_x = (track_bb[0] + track_bb[2]) / 2
                    head_y = track_bb[1] + (track_bb[3] - track_bb[1]) * 0.15
                    track_heads.append((sid, head_x, head_y))
                
                # Calculate all distances
                all_pairs = []
                for sid, hx, hy in track_heads:
                    for name, (fx, fy) in face_positions.items():
                        dist = ((hx - fx)**2 + (hy - fy)**2) ** 0.5
                        # Increased from 350 to 800 for better matching in wide shots
                        if dist < 800:  # Max distance (was 350)
                            all_pairs.append((dist, sid, name))
                
                all_pairs.sort(key=lambda x: x[0])  # Sort by distance
                
                # #region agent log
                if frame_idx < 30:
                    _debug_log("H3", "face_matching_pairs", {"frame": frame_idx, "all_pairs": [(d, s, n) for d, s, n in all_pairs[:10]], "track_heads_count": len(track_heads), "face_positions_count": len(face_positions)})
                # #endregion
                
                # Greedy one-to-one assignment
                used_sids = set()
                used_names = set()
                for dist, sid, name in all_pairs:
                    if sid not in used_sids and name not in used_names:
                        sid_to_name_override[sid] = name
                        used_sids.add(sid)
                        used_names.add(name)
                
                # #region agent log
                if frame_idx < 30:
                    _debug_log("H3", "face_matching_result", {"frame": frame_idx, "sid_to_name_override": dict(sid_to_name_override)})
                # #endregion
            
            for sid in frame_sids:
                r = feats_by_frame.get(frame_idx, {}).get(sid) or last_feat_row.get(sid)
                if not r:
                    continue
                
                last_ts = last_feat_ts.get(sid)
                is_fresh = last_ts and (ts_ms - last_ts) <= args.stale_ms
                
                # #region agent log
                if frame_idx < 30:
                    _debug_log("H5", "freshness_check", {"frame": frame_idx, "sid": sid, "last_ts": last_ts, "ts_ms": ts_ms, "stale_ms": args.stale_ms, "is_fresh": is_fresh})
                # #endregion
                
                if not is_fresh:
                    continue
                
                frame_detected += 1
                
                # Get track bbox and zone id
                track_bb = r.get("track_bbox_xyxy")
                zid = r.get("zone_id", sid)
                
                # #region agent log
                if frame_idx < 30:
                    _debug_log("H4", "track_bbox_check", {"frame": frame_idx, "sid": sid, "track_bb": track_bb, "zid": zid, "has_bbox": track_bb is not None})
                # #endregion
                
                # FIRST: Determine identity (needed for face_bbox lookup!)
                student_name = None
                is_known = False
                
                # 1. Try face-based override (one-to-one matched)
                if sid in sid_to_name_override:
                    student_name = sid_to_name_override[sid]
                    is_known = True
                
                # 2. Try track-based identity
                if not is_known and sid in track_identities:
                    info = track_identities[sid]
                    name = info.get("name", "") if isinstance(info, dict) else str(info)
                    if name and name != "UNKNOWN":
                        student_name = name
                        is_known = True
                
                # 3. Fallback: zone-based identity
                if not is_known and zid in zone_identities:
                    info = zone_identities[zid]
                    name = info.get("name", "") if isinstance(info, dict) else str(info)
                    if name and name != "UNKNOWN":
                        student_name = name
                        is_known = True
                
                # 4. Default: unknown
                if not student_name:
                    student_name = "UNKNOWN"
                    is_known = False
                
                # SECOND: Determine bbox (using identity for face_bbox lookup!)
                cur_bb = None
                bbox_source = "none"
                
                if isinstance(track_bb, list) and len(track_bb) == 4:
                    # Check if we have calibrated face_bbox for this KNOWN person
                    if is_known and student_name in face_bboxes:
                        # Use calibrated face bbox - EXACT location, NO overlap!
                        fb = face_bboxes[student_name]
                        fw = fb[2] - fb[0]
                        fh = fb[3] - fb[1]
                        cur_bb = (
                            max(0, fb[0] - int(fw * 0.15)),
                            max(0, fb[1] - int(fh * 0.05)),
                            min(W, fb[2] + int(fw * 0.15)),
                            min(H, fb[3] + int(fh * 1.2))  # Extend down for shoulders
                        )
                        bbox_source = "face_calib"
                    else:
                        # Fallback: HEAD REGION from track bbox (top 35%)
                        x1, y1, x2, y2 = track_bb
                        height = y2 - y1
                        width = x2 - x1
                        
                        head_height = int(height * 0.35)
                        y2_new = y1 + head_height
                        
                        center_x = (x1 + x2) / 2
                        head_width = width * 0.8
                        x1_new = max(0, int(center_x - head_width / 2))
                        x2_new = min(W, int(center_x + head_width / 2))
                        
                        cur_bb = (int(x1_new), int(y1), int(x2_new), int(y2_new))
                        bbox_source = "track_head"
                
                # Fallback 1: Use zone bbox directly (for Zone Tracker)
                if cur_bb is None and zid in zone_bb:
                    cur_bb = zone_bb[zid]
                    bbox_source = "zone"
                
                # Fallback 2: Try to find zone by student ID pattern (s1 -> seat_1)
                if cur_bb is None:
                    # Try different zone ID patterns
                    possible_zone_ids = [
                        zid,  # Original zone_id
                        f"seat_{sid[1:]}" if sid.startswith("s") else None,  # s1 -> seat_1
                        f"track_{sid[1:]}" if sid.startswith("s") else None,  # s1 -> track_1
                        sid,  # Just the sid
                    ]
                    for pzid in possible_zone_ids:
                        if pzid and pzid in zone_bb:
                            cur_bb = zone_bb[pzid]
                            bbox_source = f"zone_fallback_{pzid}"
                            break
                
                # Fallback 3: Use seat_bbox_xyxy from features if available
                if cur_bb is None:
                    seat_bb = r.get("seat_bbox_xyxy")
                    if isinstance(seat_bb, list) and len(seat_bb) == 4:
                        cur_bb = tuple(int(v) for v in seat_bb)
                        bbox_source = "seat_bbox"
                
                # Still no bbox - skip this person
                if cur_bb is None:
                    # #region agent log
                    if frame_idx < 50:
                        _debug_log("H6", "no_bbox_skip", {"frame": frame_idx, "sid": sid, "zid": zid, "track_bb": track_bb, "zone_bb_keys": list(zone_bb.keys())})
                    # #endregion
                    continue
                
                # #region agent log
                if frame_idx < 30:
                    _debug_log("H7", "bbox_resolved", {"frame": frame_idx, "sid": sid, "bbox_source": bbox_source, "cur_bb": cur_bb})
                # #endregion
                
                # Smooth
                if args.bbox_smooth > 0:
                    prev = bbox_smooth.get(sid)
                    if prev:
                        a = args.bbox_smooth
                        sm = tuple(a * c + (1 - a) * p for c, p in zip(cur_bb, prev))
                        bbox_smooth[sid] = sm
                        cur_bb = tuple(int(v) for v in sm)
                    else:
                        bbox_smooth[sid] = tuple(float(v) for v in cur_bb)
                
                if is_known:
                    frame_identified += 1
                    known_in_frame.append((student_name, "OK"))
                else:
                    unknown_in_frame += 1
                
                # Calculate risk
                st = active.get(sid, {})
                act_list = [b for b, on in st.items() if on and b in show_behs]
                
                # #region agent log
                if frame_idx < 50 and frame_idx % 10 == 0:
                    _debug_log("H1", "active_behaviors", {"frame": frame_idx, "sid": sid, "student_name": student_name, "active_state": dict(st), "act_list": act_list, "show_behs": list(show_behs)})
                # #endregion
                
                risk = 0.0
                if st.get("look_down"):
                    risk += args.w_look
                if st.get("head_turn_left") or st.get("head_turn_right"):
                    risk += args.w_turn
                if st.get("hand_to_face"):
                    risk += args.w_htf
                risk = min(1.0, risk)
                
                is_alert = bool(act_list)
                if is_alert:
                    frame_alerts += 1
                
                # Determine status text
                status_text = ""
                if act_list:
                    status_text = _behavior_text(act_list[0])
                    # #region agent log
                    _debug_log("H1_POST", "status_text_set", {"frame": frame_idx, "sid": sid, "student_name": student_name, "status_text": status_text, "act_list": act_list, "risk": risk, "show_behs": list(show_behs)})
                    # #endregion
                
                # Box color based on risk
                if risk >= 0.6:
                    box_color = Colors.DANGER
                elif risk >= 0.3:
                    box_color = Colors.WARNING
                elif is_known:
                    box_color = Colors.IDENTIFIED
                else:
                    box_color = Colors.SAFE
                
                # Store for sorting and overlap detection
                draw_data.append({
                    'sid': sid,
                    'bb': cur_bb,
                    'color': box_color,
                    'is_alert': is_alert,
                    'name': student_name,
                    'is_known': is_known,
                    'status': status_text,
                    'risk': risk,
                    'behaviors': act_list,
                })
            
            # Sort draw data by risk (high risk on top) then by y-coordinate
            draw_data.sort(key=lambda x: (-x['risk'], x['bb'][1]))
            
            # Draw trails first (under everything)
            for dd in draw_data:
                sid = dd['sid']
                cur_bb = dd['bb']
                if args.trail:
                    cx = (cur_bb[0] + cur_bb[2]) // 2
                    cy = (cur_bb[1] + cur_bb[3]) // 2
                    pts = trail_pts.get(sid, [])
                    pts.append((cx, cy))
                    pts = pts[-args.trail_len:]
                    trail_pts[sid] = pts
                    for i in range(1, len(pts)):
                        alpha = int(180 * i / len(pts))
                        cv2.line(canvas, pts[i-1], pts[i], (alpha, alpha, alpha), 3)
            
            # Draw bboxes with overlap detection
            drawn_bboxes = []
            for dd in draw_data:
                cur_bb = dd['bb']
                box_color = dd['color']
                is_alert = dd['is_alert']
                student_name = dd['name']
                is_known = dd['is_known']
                status_text = dd['status']
                risk = dd['risk']
                act_list = dd['behaviors']
                
                # Check for overlap with already drawn boxes
                has_overlap = _check_any_overlap(cur_bb, drawn_bboxes, threshold=0.15)
                
                # Draw THICK bounding box (reduced glow if overlapping)
                _draw_thick_box(canvas, cur_bb, box_color, thickness=5, glow=is_alert, reduce_glow=has_overlap)
                
                # Draw name badge ABOVE bbox
                _draw_name_badge(canvas, student_name, is_known, cur_bb)
                
                # Draw status bar AT BOTTOM of bbox (with all active behaviors)
                _draw_status_bar(canvas, cur_bb, status_text, risk, is_known, act_list)
                
                drawn_bboxes.append(cur_bb)
                
                # Record alert
                if is_alert and ts_ms - last_alert_ts.get(sid, -10000) > 3000:
                    mins, secs = ts_ms // 60000, (ts_ms // 1000) % 60
                    alerts.append(f"{mins:02d}:{secs:02d} {student_name}: {status_text}")
                    alerts = alerts[-15:]
                    last_alert_ts[sid] = ts_ms
            
            # Header
            _draw_header(canvas, frame_idx, ts_ms, frame_detected, frame_identified, frame_alerts, args.session_name)
            
            # Sidebar
            if args.sidebar:
                _draw_sidebar(canvas, known_in_frame, unknown_in_frame, alerts, sidebar_w)
            
            writer.write(canvas)
            rendered += 1
            frame_idx += 1
            
            if rendered % 100 == 0:
                print(f"  {rendered}/{args.max_frames}...")
    
    finally:
        cap.release()
        writer.release()
    
    print(f"\n✅ Output: {out_path}")
    print(f"   Rendered: {rendered} frames")


if __name__ == "__main__":
    main()

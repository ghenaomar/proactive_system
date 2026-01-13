#!/usr/bin/env python3
"""
Face-Based Professional Renderer

This renderer uses face detection to identify students directly,
bypassing the body tracker entirely. No more ID switching!

Features:
    - Face-based identity (no tracker confusion)
    - Shows ALL detected faces (enrolled + unknown)
    - Professional visual design (from render_demo_video.py)
    - Behavior indicators from MediaPipe features
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2


# =============================================================================
# COLORS - Same as render_demo_video.py
# =============================================================================

class Colors:
    SAFE = (80, 200, 100)           # Green
    WARNING = (50, 200, 240)        # Yellow  
    ALERT = (50, 150, 255)          # Orange
    DANGER = (60, 60, 240)          # Red
    IDENTIFIED = (230, 180, 60)     # Cyan
    UNKNOWN = (80, 80, 180)         # Dark red
    BG_DARK = (20, 20, 25)
    BG_PANEL = (30, 30, 35)
    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)


def _risk_color(risk: float) -> Tuple[int, int, int]:
    if risk < 0.3:
        return Colors.SAFE
    elif risk < 0.5:
        return Colors.WARNING
    elif risk < 0.7:
        return Colors.ALERT
    return Colors.DANGER


# =============================================================================
# DRAWING FUNCTIONS - Same style as render_demo_video.py
# =============================================================================

def _draw_thick_box(img, bbox, color, thickness=5, glow=True):
    """Draw HIGHLY VISIBLE bounding box with glow effect."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    if glow:
        for i in range(3, 0, -1):
            glow_color = tuple(max(0, c // 2) for c in color)
            cv2.rectangle(img, (x1-i*2, y1-i*2), (x2+i*2, y2+i*2), glow_color, thickness//2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    highlight = tuple(min(255, c + 50) for c in color)
    cv2.rectangle(img, (x1+3, y1+3), (x2-3, y2-3), highlight, 2, cv2.LINE_AA)


def _draw_label_plate(img, text, pos, bg_color, font_scale=0.8, thickness=2, padding=12):
    """Draw label with colored background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = int(pos[0]), int(pos[1])
    x1, y1 = x, y - th - padding
    x2, y2 = x + tw + padding * 2, y + padding
    
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), Colors.WHITE, 2)
    cv2.putText(img, text, (x + padding, y), font, font_scale, Colors.WHITE, thickness, cv2.LINE_AA)


def _draw_status_bar(img, bbox, behaviors, risk, is_identified):
    """Draw status bar at bottom of bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bar_height = 40
    
    if risk >= 0.5:
        bar_color = _risk_color(risk)
    elif is_identified:
        bar_color = Colors.IDENTIFIED
    else:
        bar_color = Colors.SAFE
    
    bar_y1 = y2 - bar_height
    bar_y2 = y2
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, bar_y1), (x2, bar_y2), bar_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Behavior text (ASCII only!)
    indicators = []
    if behaviors.get("look_down", 0) > 0.5:
        indicators.append("LOOK")
    if behaviors.get("head_turn_left", 0) > 0.5:
        indicators.append("LEFT")
    if behaviors.get("head_turn_right", 0) > 0.5:
        indicators.append("RIGHT")
    if behaviors.get("hand_to_face", 0) > 0.5:
        indicators.append("HAND")
    
    status_text = " | ".join(indicators) if indicators else "OK"
    cv2.putText(img, status_text, (x1 + 8, bar_y2 - 12), font, 0.55, Colors.WHITE, 2, cv2.LINE_AA)
    
    risk_text = f"{int(risk*100)}%"
    cv2.putText(img, risk_text, (x2 - 55, bar_y2 - 12), font, 0.6, Colors.WHITE, 2, cv2.LINE_AA)


def _draw_name_badge(img, name, is_known, bbox):
    """Draw name badge above bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    if is_known:
        bg_color = Colors.IDENTIFIED
    else:
        bg_color = Colors.UNKNOWN
        name = "UNKNOWN"
    
    _draw_label_plate(img, name, (x1, max(60, y1 - 5)), bg_color, font_scale=0.7, thickness=2, padding=10)


def _draw_header(img, frame_idx, detected, identified, alerts, session):
    """Draw header bar."""
    H, W = img.shape[:2]
    header_h = 55
    
    cv2.rectangle(img, (0, 0), (W, header_h), Colors.BG_DARK, -1)
    cv2.line(img, (0, header_h), (W, header_h), Colors.GRAY, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 38
    
    cv2.putText(img, session, (20, y), font, 0.7, Colors.WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, f"FRAME: {frame_idx}", (350, y), font, 0.55, Colors.GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, f"DETECTED: {detected}", (520, y), font, 0.6, Colors.SAFE, 2, cv2.LINE_AA)
    cv2.putText(img, f"IDENTIFIED: {identified}", (720, y), font, 0.6, Colors.IDENTIFIED, 2, cv2.LINE_AA)
    
    alert_color = Colors.DANGER if alerts > 0 else Colors.SAFE
    cv2.putText(img, f"ALERTS: {alerts}", (940, y), font, 0.65, alert_color, 2, cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Face-based professional video rendering")
    parser.add_argument("--video", required=True)
    parser.add_argument("--students", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--behaviors", help="Behaviors JSONL for stable indicators")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--face_every", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--session_name", default="EXAM MONITORING")
    args = parser.parse_args()

    print("=" * 60)
    print("FACE-BASED PROFESSIONAL RENDERING")
    print("=" * 60)

    # 1. Load InsightFace
    print("\n1. Loading face recognition...")
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print("   OK")

    # 2. Load enrolled students
    print("\n2. Loading enrolled students...")
    students_dir = Path(args.students)
    enrolled = {}
    
    for folder in students_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        info_file = folder / "info.json"
        emb_file = folder / "embeddings.npy"
        
        if info_file.exists() and emb_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            name = info.get("name", folder.name)
            embeddings = np.load(emb_file)
            if embeddings.ndim == 2:
                emb = np.mean(embeddings, axis=0)
            else:
                emb = embeddings
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            enrolled[name] = emb
            print(f"   - {name}")
    
    print(f"   Total: {len(enrolled)} students")

    # 3. Load features
    print("\n3. Loading features...")
    features_by_frame: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    
    with open(args.features) as f:
        for line in f:
            row = json.loads(line)
            fidx = row.get("frame_index")
            sid = row.get("student_id")
            if fidx is not None and sid:
                features_by_frame[fidx][sid] = row
    
    print(f"   {len(features_by_frame)} frames")
    
    # 3b. Load behaviors (events) - for stable indicators like render_demo_video.py
    behaviors_by_ts = []
    if args.behaviors and Path(args.behaviors).exists():
        print("\n3b. Loading behaviors (events)...")
        with open(args.behaviors) as f:
            for line in f:
                behaviors_by_ts.append(json.loads(line))
        behaviors_by_ts.sort(key=lambda x: int(x.get("ts_ms", 0)))
        print(f"   {len(behaviors_by_ts)} events")
    else:
        print("\n3b. No behaviors file (will use scores from features)")

    # 4. Open video
    print("\n4. Opening video...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   {width}x{height} @ {fps:.1f} FPS")

    # 5. Setup output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    # 6. Render
    print(f"\n5. Rendering...")
    
    frame_idx = 0
    rendered = 0
    cached_faces = []  # (name, bbox, is_enrolled)
    
    # Track active behaviors (like render_demo_video.py)
    # Maps: track_id -> {behavior: is_active}
    active_behaviors: Dict[str, Dict[str, bool]] = {}
    beh_idx = 0
    
    # Map name to nearest track_id (for behavior lookup)
    name_to_track: Dict[str, str] = {}
    
    def find_nearest_track(face_cx, face_cy, frame_features):
        """Find nearest track ID for behavior lookup."""
        best_dist = 500
        best_sid = None
        
        for sid, feats in frame_features.items():
            bbox = feats.get("track_bbox_xyxy")
            if not bbox:
                continue
            tx = (bbox[0] + bbox[2]) / 2
            ty = bbox[1] + (bbox[3] - bbox[1]) * 0.15
            
            dist = ((face_cx - tx)**2 + (face_cy - ty)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_sid = sid
        
        return best_sid
    
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate timestamp for this frame
        ts_ms = int(frame_idx * 1000 / fps)
        
        # Update active behaviors from events (like render_demo_video.py)
        while beh_idx < len(behaviors_by_ts):
            ev = behaviors_by_ts[beh_idx]
            if int(ev.get("ts_ms", 0)) > ts_ms:
                break
            sid = str(ev.get("student_id", ""))
            beh = str(ev.get("behavior", ""))
            et = str(ev.get("event_type", ""))
            
            if sid not in active_behaviors:
                active_behaviors[sid] = {"look_down": False, "head_turn_left": False, "head_turn_right": False, "hand_to_face": False}
            
            if beh in active_behaviors[sid]:
                active_behaviors[sid][beh] = (et == "start")
            
            beh_idx += 1
        
        # Detect faces periodically
        if frame_idx % args.face_every == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_app.get(frame_rgb)
            
            cached_faces = []
            used_names = set()
            
            # Sort by face size (larger first)
            face_data = []
            for face in faces:
                if face.embedding is None:
                    continue
                bbox = face.bbox.astype(int).tolist()
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]
                if face_w < 15 or face_h < 15:
                    continue
                face_data.append((face, bbox, face_w * face_h))
            
            face_data.sort(key=lambda x: -x[2])
            
            # Filter overlapping faces
            filtered = []
            for face, bbox, size in face_data:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                too_close = False
                for _, eb, _ in filtered:
                    ex = (eb[0] + eb[2]) / 2
                    ey = (eb[1] + eb[3]) / 2
                    if ((cx - ex)**2 + (cy - ey)**2) ** 0.5 < 80:
                        too_close = True
                        break
                
                if not too_close:
                    filtered.append((face, bbox, size))
            
            # Match faces to students (one-to-one)
            for face, bbox, _ in filtered:
                face_emb = face.embedding / np.linalg.norm(face.embedding)
                
                best_name = None
                best_sim = args.threshold
                
                for name, emb in enrolled.items():
                    if name in used_names:
                        continue
                    sim = float(np.dot(face_emb, emb))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                
                # Expand face bbox to upper body
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                expanded = [
                    max(0, x1 - int(w * 0.4)),
                    max(60, y1 - int(h * 0.15)),
                    min(width, x2 + int(w * 0.4)),
                    min(height, y2 + int(h * 2.0))
                ]
                
                if best_name:
                    used_names.add(best_name)
                    cached_faces.append((best_name, expanded, True))
                else:
                    cached_faces.append(("UNKNOWN", expanded, False))
        
        # Get features
        frame_features = features_by_frame.get(frame_idx, {})
        
        # Stats
        enrolled_count = sum(1 for _, _, e in cached_faces if e)
        unknown_count = sum(1 for _, _, e in cached_faces if not e)
        alert_count = 0
        
        # Draw each face
        for name, bbox, is_enrolled in cached_faces:
            face_cx = (bbox[0] + bbox[2]) / 2
            face_cy = bbox[1] + 50
            
            # Find nearest track for behavior lookup
            track_id = find_nearest_track(face_cx, face_cy, frame_features)
            
            # Get active behaviors (stable states, not flickering scores!)
            if track_id and track_id in active_behaviors:
                active = active_behaviors[track_id]
                behaviors = {
                    "look_down": 1.0 if active.get("look_down") else 0.0,
                    "head_turn_left": 1.0 if active.get("head_turn_left") else 0.0,
                    "head_turn_right": 1.0 if active.get("head_turn_right") else 0.0,
                    "hand_to_face": 1.0 if active.get("hand_to_face") else 0.0,
                }
            else:
                behaviors = {"look_down": 0, "head_turn_left": 0, "head_turn_right": 0, "hand_to_face": 0}
            
            risk = max(
                behaviors.get("look_down", 0) * 0.3,
                behaviors.get("head_turn_left", 0) * 0.35,
                behaviors.get("head_turn_right", 0) * 0.35,
                behaviors.get("hand_to_face", 0) * 0.35,
            )
            
            if risk >= 0.5:
                alert_count += 1
            
            if risk >= 0.5:
                box_color = _risk_color(risk)
                glow = True
            elif is_enrolled:
                box_color = Colors.IDENTIFIED
                glow = False
            else:
                box_color = Colors.UNKNOWN
                glow = False
            
            _draw_thick_box(frame, bbox, box_color, thickness=5, glow=glow)
            _draw_name_badge(frame, name, is_enrolled, bbox)
            _draw_status_bar(frame, bbox, behaviors, risk, is_enrolled)
        
        _draw_header(frame, frame_idx, len(cached_faces), enrolled_count, alert_count, args.session_name)
        
        out.write(frame)
        rendered += 1
        
        if rendered % 100 == 0:
            print(f"   {rendered} frames...")
        
        frame_idx += 1
    
    cap.release()
    out.release()

    print(f"\n   Done! {rendered} frames")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()

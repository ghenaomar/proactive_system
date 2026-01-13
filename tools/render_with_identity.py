#!/usr/bin/env python3
"""
Render Demo Video with Fixed Identity

Uses calibrated identity mapping to label each person with their name.
Identity is determined by position (from calibration), NOT by track_id.

This ensures stable identity throughout the video.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import numpy as np
import cv2


# ============================================================
# VISUAL STYLE CONSTANTS
# ============================================================

# Colors (BGR)
COLORS = {
    "green": (0, 200, 100),
    "yellow": (0, 220, 255),
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
    "blue": (255, 150, 50),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "dark_gray": (40, 40, 40),
}

# Risk level colors
RISK_COLORS = {
    "low": COLORS["green"],
    "medium": COLORS["yellow"],
    "high": COLORS["orange"],
    "critical": COLORS["red"],
}


def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.5:
        return "medium"
    elif score < 0.7:
        return "high"
    else:
        return "critical"


def draw_thick_box(img, bbox, color, thickness=3, glow=False):
    """Draw a thick bounding box with optional glow effect."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    if glow:
        # Draw glow layers
        for i in range(3, 0, -1):
            alpha = 0.3 / i
            overlay = img.copy()
            cv2.rectangle(overlay, (x1-i*2, y1-i*2), (x2+i*2, y2+i*2), color, thickness + i*2)
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_label(img, text, position, bg_color, text_color=COLORS["white"], font_scale=0.7):
    """Draw a label with background."""
    x, y = int(position[0]), int(position[1])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Background
    padding = 8
    cv2.rectangle(img, 
                  (x - padding, y - th - padding - 5),
                  (x + tw + padding, y + padding - 5),
                  bg_color, -1)
    
    # Text
    cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, thickness)
    
    return img


def draw_status_bar(img, bbox, behaviors: Dict, name: str):
    """Draw a status bar below the bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bar_height = 35
    bar_y = y2 + 5
    bar_width = x2 - x1
    
    # Background
    cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_height), COLORS["dark_gray"], -1)
    cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_height), COLORS["gray"], 1)
    
    # Name
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, name, (x1 + 5, bar_y + 22), font, 0.6, COLORS["white"], 2)
    
    # Behavior indicators
    indicators = []
    if behaviors.get("look_down", 0) > 0.5:
        indicators.append(("↓", COLORS["yellow"]))
    if behaviors.get("head_turn_left", 0) > 0.5 or behaviors.get("head_turn_right", 0) > 0.5:
        indicators.append(("↔", COLORS["orange"]))
    if behaviors.get("hand_to_face", 0) > 0.5:
        indicators.append(("✋", COLORS["red"]))
    
    # Draw indicators on the right side
    ix = x2 - 25
    for symbol, color in reversed(indicators):
        cv2.putText(img, symbol, (ix, bar_y + 22), font, 0.6, color, 2)
        ix -= 25
    
    return img


def match_bboxes_to_students(
    bboxes: List[Tuple[str, Tuple[float, float]]],  # [(sid, (cx, cy)), ...]
    student_positions: Dict[str, Tuple[float, float]],
    max_distance: float = 400
) -> Dict[str, str]:
    """
    One-to-one matching: each student assigned to exactly ONE bbox.
    
    Uses greedy matching by distance (closest pairs first).
    
    Returns:
        Dict mapping sid -> student_name (or "UNKNOWN")
    """
    if not bboxes or not student_positions:
        return {sid: "UNKNOWN" for sid, _ in bboxes}
    
    # Calculate all distances
    distances = []  # [(distance, sid, student_name), ...]
    
    for sid, (bx, by) in bboxes:
        for name, (sx, sy) in student_positions.items():
            dist = math.sqrt((bx - sx)**2 + (by - sy)**2)
            if dist <= max_distance:
                distances.append((dist, sid, name))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[0])
    
    # Greedy one-to-one assignment
    assigned_sids = set()
    assigned_students = set()
    result = {}
    
    for dist, sid, name in distances:
        if sid not in assigned_sids and name not in assigned_students:
            result[sid] = name
            assigned_sids.add(sid)
            assigned_students.add(name)
    
    # Mark unassigned as UNKNOWN
    for sid, _ in bboxes:
        if sid not in result:
            result[sid] = "UNKNOWN"
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Render video with fixed identity labels")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--features", required=True, help="MediaPipe features JSONL")
    parser.add_argument("--identity", required=True, help="Identity mapping JSON from calibration")
    parser.add_argument("--behaviors", help="Behaviors JSONL (optional)")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--max_frames", type=int, default=600, help="Max frames to render")
    parser.add_argument("--show_unknown", action="store_true", help="Show unknown persons too")
    args = parser.parse_args()

    print("=" * 60)
    print("RENDER VIDEO WITH IDENTITY")
    print("=" * 60)

    # 1. Load identity mapping
    print("\n1. Loading identity mapping...")
    with open(args.identity) as f:
        identity_data = json.load(f)
    
    student_positions = {}
    for name, data in identity_data.get("students", {}).items():
        pos = data.get("position", {})
        if "x" in pos and "y" in pos:
            student_positions[name] = (pos["x"], pos["y"])
            print(f"   ✓ {name} at ({pos['x']:.0f}, {pos['y']:.0f})")
    
    print(f"\n   Total: {len(student_positions)} students")

    # 2. Load features
    print("\n2. Loading features...")
    features_by_frame: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    
    with open(args.features) as f:
        for line in f:
            row = json.loads(line)
            frame_idx = row.get("frame_index")
            sid = row.get("student_id")
            if frame_idx is not None and sid:
                features_by_frame[frame_idx][sid] = row
    
    print(f"   ✓ {len(features_by_frame)} frames with features")

    # 3. Load behaviors (optional)
    behaviors_by_frame: Dict[int, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
    
    if args.behaviors and Path(args.behaviors).exists():
        print("\n3. Loading behaviors...")
        with open(args.behaviors) as f:
            for line in f:
                row = json.loads(line)
                frame_idx = row.get("frame_index")
                sid = row.get("student_id")
                if frame_idx is not None and sid:
                    for key in ["look_down", "head_turn_left", "head_turn_right", "hand_to_face"]:
                        if key in row:
                            behaviors_by_frame[frame_idx][sid][key] = row[key]
        print(f"   ✓ Loaded behaviors")
    else:
        print("\n3. No behaviors file (skipping)")

    # 4. Open video
    print("\n4. Opening video...")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"   ✗ Cannot open: {args.video}")
        sys.exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   ✓ {width}x{height} @ {fps:.1f} FPS")

    # 5. Setup output
    print("\n5. Setting up output...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    print(f"   ✓ Output: {args.out}")

    # 6. Render frames
    print("\n6. Rendering...")
    
    frame_idx = 0
    rendered = 0
    
    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get features for this frame
        frame_features = features_by_frame.get(frame_idx, {})
        frame_behaviors = behaviors_by_frame.get(frame_idx, {})
        
        # Collect all bboxes for this frame
        frame_bboxes = []  # [(sid, (cx, cy), bbox, feats), ...]
        
        for sid, feats in frame_features.items():
            bbox = feats.get("track_bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Use HEAD position (top 30% of bbox) instead of center!
            # This is more stable because heads don't overlap as much
            center_x = (x1 + x2) / 2
            head_y = y1 + (y2 - y1) * 0.15  # Top 15% = head area
            
            frame_bboxes.append((sid, (center_x, head_y), bbox, feats))
        
        # ONE-TO-ONE MATCHING: each student = one bbox only!
        # Use smaller max_distance for stricter matching
        sid_centers = [(sid, center) for sid, center, _, _ in frame_bboxes]
        sid_to_name = match_bboxes_to_students(sid_centers, student_positions, max_distance=250)
        
        # Now render each bbox with assigned name
        for sid, (center_x, center_y), bbox, feats in frame_bboxes:
            name = sid_to_name.get(sid, "UNKNOWN")
            
            # Skip unknown if not requested
            if name == "UNKNOWN" and not args.show_unknown:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Get behaviors
            behaviors = frame_behaviors.get(sid, {})
            
            # Determine risk and color
            risk_score = max(
                behaviors.get("look_down", 0),
                behaviors.get("head_turn_left", 0),
                behaviors.get("head_turn_right", 0),
                behaviors.get("hand_to_face", 0) * 0.5,
            )
            
            risk_level = get_risk_level(risk_score)
            color = RISK_COLORS[risk_level]
            
            # Draw bounding box
            glow = risk_level in ["high", "critical"]
            draw_thick_box(frame, bbox, color, thickness=4, glow=glow)
            
            # Draw name label
            label_text = name if name != "UNKNOWN" else f"Person {sid}"
            draw_label(frame, label_text, (x1, y1), color)
            
            # Draw status bar
            draw_status_bar(frame, bbox, behaviors, name)
        
        # Write frame
        out.write(frame)
        rendered += 1
        
        if rendered % 100 == 0:
            print(f"   Rendered {rendered} frames...")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\n   ✓ Rendered {rendered} frames")

    # Summary
    print("\n" + "=" * 60)
    print("RENDER COMPLETE")
    print("=" * 60)
    print(f"\n✅ Output: {args.out}")
    print(f"\nIdentity assignments:")
    for sid, name in sorted(track_to_name.items()):
        print(f"   {sid} → {name}")


if __name__ == "__main__":
    main()

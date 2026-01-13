"""Extract a frame from video with zones overlay.

Usage:
    python tools/extract_frame_with_zones.py \
        --video data/raw/SPU_DEMO.mp4 \
        --zones configs/zones/spu_demo.yaml \
        --frame 50 \
        --out frame_with_zones.jpg
"""

import argparse
import json
from pathlib import Path

import cv2
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--zones", required=True, help="Zones YAML file")
    ap.add_argument("--frame", type=int, default=50, help="Frame number to extract")
    ap.add_argument("--out", default="frame_with_zones.jpg", help="Output image path")
    args = ap.parse_args()
    
    # Load zones
    with open(args.zones, "r") as f:
        zones_data = yaml.safe_load(f)
    
    # Get zones list
    zones = zones_data.get("zones", {})
    if isinstance(zones, dict) and "zones" in zones:
        zones = zones["zones"]
    elif isinstance(zones, list):
        pass
    else:
        zones = []
    
    print(f"Loaded {len(zones)} zones")
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {W}x{H}, {total_frames} frames")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    
    if not ok:
        print(f"Error: Cannot read frame {args.frame}")
        return
    
    # Draw zones
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Orange
    ]
    
    for i, zone in enumerate(zones):
        zone_id = zone.get("id", f"zone_{i}")
        bbox = zone.get("bbox", [])
        
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = colors[i % len(colors)]
        
        # Draw thick rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        
        # Draw label with background
        label = f"{zone_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Background
        cv2.rectangle(frame, (x1, y1 - th - 20), (x1 + tw + 20, y1), color, -1)
        # Text
        cv2.putText(frame, label, (x1 + 10, y1 - 10), font, font_scale, (255, 255, 255), thickness)
        
        # Draw zone number in center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(frame, str(i + 1), (cx - 20, cy + 20), font, 2.0, color, 4)
    
    # Add info text
    info = f"Frame: {args.frame} | Zones: {len(zones)} | Size: {W}x{H}"
    cv2.putText(frame, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(args.out, frame)
    print(f"\n✅ Saved: {args.out}")
    print(f"\nZones:")
    for i, zone in enumerate(zones):
        zone_id = zone.get("id", f"zone_{i}")
        bbox = zone.get("bbox", [])
        print(f"  {i+1}. {zone_id}: {bbox}")


if __name__ == "__main__":
    main()

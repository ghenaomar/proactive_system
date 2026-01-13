#!/usr/bin/env python3
"""Calibrate zones based on actual YOLO detections.

This tool analyzes a video with YOLOv8 to find where people actually are,
then creates or adjusts zones to match those positions.

Usage:
    # Create new zones from scratch
    python tools/calibrate_zones_from_detections.py \
      --video data/raw/SPU_DEMO.mp4 \
      --out configs/zones/auto_zones.yaml \
      --frames 100

    # Adjust existing zones
    python tools/calibrate_zones_from_detections.py \
      --video data/raw/SPU_DEMO.mp4 \
      --current_zones configs/zones/spu_demo_30sec.yaml \
      --out configs/zones/adjusted_zones.yaml \
      --expand_factor 1.2

Author: Proctor AI Team
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def cluster_detections(
    detections: List[Tuple[int, int, int, int]],
    min_samples: int = 5,
    merge_distance: float = 100.0,
) -> List[Tuple[int, int, int, int]]:
    """Cluster detections into groups (potential seats)."""
    if not detections:
        return []
    
    # Convert to centers
    centers = []
    for x1, y1, x2, y2 in detections:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy, x1, y1, x2, y2))
    
    # Simple clustering by distance
    clusters = []
    used = set()
    
    for i, (cx, cy, x1, y1, x2, y2) in enumerate(centers):
        if i in used:
            continue
        
        cluster = [(cx, cy, x1, y1, x2, y2)]
        used.add(i)
        
        for j, (cx2, cy2, x12, y12, x22, y22) in enumerate(centers):
            if j in used:
                continue
            
            dist = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
            if dist < merge_distance:
                cluster.append((cx2, cy2, x12, y12, x22, y22))
                used.add(j)
        
        if len(cluster) >= min_samples:
            clusters.append(cluster)
    
    # Convert clusters to bboxes (union of all detections in cluster)
    result = []
    for cluster in clusters:
        x1_min = min(c[2] for c in cluster)
        y1_min = min(c[3] for c in cluster)
        x2_max = max(c[4] for c in cluster)
        y2_max = max(c[5] for c in cluster)
        result.append((int(x1_min), int(y1_min), int(x2_max), int(y2_max)))
    
    return result


def load_zones(path: Path) -> List[Dict[str, Any]]:
    """Load zones from YAML file."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    zones = []
    zones_data = data.get("zones", {})
    if isinstance(zones_data, dict):
        zones_list = zones_data.get("zones", [])
    else:
        zones_list = zones_data
    
    for z in zones_list:
        zone_id = z.get("id", "unknown")
        bbox = z.get("bbox", [])
        if len(bbox) == 4:
            zones.append({
                "id": zone_id,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            })
    
    return zones


def expand_bbox(
    bbox: List[int],
    factor: float,
    frame_size: Tuple[int, int],
) -> List[int]:
    """Expand bbox by factor while keeping within frame bounds."""
    x1, y1, x2, y2 = bbox
    W, H = frame_size
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * factor
    h = (y2 - y1) * factor
    
    nx1 = max(0, int(cx - w / 2))
    ny1 = max(0, int(cy - h / 2))
    nx2 = min(W, int(cx + w / 2))
    ny2 = min(H, int(cy + h / 2))
    
    return [nx1, ny1, nx2, ny2]


def find_best_zone_match(
    detection: Tuple[int, int, int, int],
    zones: List[Dict[str, Any]],
) -> Optional[str]:
    """Find which zone best matches a detection (by center distance)."""
    det_cx = (detection[0] + detection[2]) / 2
    det_cy = (detection[1] + detection[3]) / 2
    
    best_zone = None
    best_dist = float('inf')
    
    for z in zones:
        bbox = z["bbox"]
        zone_cx = (bbox[0] + bbox[2]) / 2
        zone_cy = (bbox[1] + bbox[3]) / 2
        
        dist = ((det_cx - zone_cx) ** 2 + (det_cy - zone_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_zone = z["id"]
    
    return best_zone if best_dist < 500 else None  # Max 500px distance


def main():
    parser = argparse.ArgumentParser(description="Calibrate zones from detections")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True, help="Output zones YAML path")
    parser.add_argument("--current_zones", type=str, help="Existing zones to adjust")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to analyze")
    parser.add_argument("--expand_factor", type=float, default=1.2, help="Expand zones by this factor")
    parser.add_argument("--min_detections", type=int, default=10, help="Min detections to create zone")
    parser.add_argument("--visualize", action="store_true", help="Create visualization")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Zone Calibration from Detections")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Output: {args.out}")
    print(f"Frames to analyze: {args.frames}")
    print(f"Expand factor: {args.expand_factor}")
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video size: {W}x{H}, {total_frames} frames")
    
    # Load YOLOv8
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return
    
    # Load existing zones if provided
    existing_zones = []
    if args.current_zones:
        existing_zones = load_zones(Path(args.current_zones))
        print(f"Loaded {len(existing_zones)} existing zones")
    
    # Collect detections
    print("\nAnalyzing video...")
    all_detections = []
    zone_detections: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    
    frame_indices = np.linspace(0, min(total_frames - 1, total_frames), args.frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run detection
        results = model.predict(frame, classes=[0], conf=0.25, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det = (int(x1), int(y1), int(x2), int(y2))
                all_detections.append(det)
                
                # Match to existing zone
                if existing_zones:
                    zone_id = find_best_zone_match(det, existing_zones)
                    if zone_id:
                        zone_detections[zone_id].append(det)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{args.frames} frames...")
    
    cap.release()
    print(f"Total detections collected: {len(all_detections)}")
    
    # Create or adjust zones
    new_zones = []
    
    if existing_zones:
        # Mode: Adjust existing zones based on actual detections
        print("\nAdjusting existing zones...")
        
        for zone in existing_zones:
            zone_id = zone["id"]
            dets = zone_detections.get(zone_id, [])
            
            if len(dets) >= args.min_detections:
                # Calculate new bbox from detections
                x1_min = min(d[0] for d in dets)
                y1_min = min(d[1] for d in dets)
                x2_max = max(d[2] for d in dets)
                y2_max = max(d[3] for d in dets)
                
                # Expand
                new_bbox = expand_bbox([x1_min, y1_min, x2_max, y2_max], args.expand_factor, (W, H))
                
                print(f"  {zone_id}: {len(dets)} detections, adjusted bbox")
            else:
                # Keep original but expand
                new_bbox = expand_bbox(zone["bbox"], args.expand_factor, (W, H))
                print(f"  {zone_id}: Only {len(dets)} detections, keeping original (expanded)")
            
            new_zones.append({
                "id": zone_id,
                "bbox": new_bbox,
            })
    else:
        # Mode: Create zones from scratch using clustering
        print("\nCreating zones from detections...")
        
        # Cluster detections
        clusters = cluster_detections(all_detections, min_samples=args.min_detections)
        print(f"Found {len(clusters)} clusters (potential seats)")
        
        for i, bbox in enumerate(clusters):
            zone_id = f"seat_{i + 1}"
            new_bbox = expand_bbox(list(bbox), args.expand_factor, (W, H))
            new_zones.append({
                "id": zone_id,
                "bbox": new_bbox,
            })
            print(f"  {zone_id}: {new_bbox}")
    
    # Save zones
    output_data = {
        "meta": {
            "source": args.video,
            "width": W,
            "height": H,
            "zones": len(new_zones),
            "method": "calibrated_from_detections",
            "expand_factor": args.expand_factor,
        },
        "zones": {
            "zones": new_zones,
        },
        "students": [
            {"id": f"s{i + 1}", "zone_id": z["id"]}
            for i, z in enumerate(new_zones)
        ],
    }
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n✅ Saved {len(new_zones)} zones to: {args.out}")
    
    # Create visualization
    if args.visualize:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            for z in new_zones:
                bbox = z["bbox"]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                cv2.putText(frame, z["id"], (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            vis_path = str(out_path).replace(".yaml", "_preview.jpg")
            cv2.imwrite(vis_path, frame)
            print(f"Visualization saved: {vis_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

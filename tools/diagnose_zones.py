#!/usr/bin/env python3
"""Diagnose Zone Tracking Issues.

This tool helps identify why bboxes or behaviors aren't showing correctly.
It analyzes:
1. Zone definitions vs actual detections
2. Track-to-zone assignment success rate
3. MediaPipe features availability
4. Behavior detection coverage

Usage:
    python tools/diagnose_zones.py --run_dir outputs/runs/xxx
    python tools/diagnose_zones.py --video data/raw/video.mp4 --zones configs/zones/xxx.yaml

Author: Proctor AI Team
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
    return rows


def load_zones_from_yaml(path: Path) -> List[Dict[str, Any]]:
    """Load zones from YAML file."""
    import yaml
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
                "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            })
    return zones


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
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
    
    return inter_area / union_area if union_area > 0 else 0.0


def center_in_bbox(point: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> bool:
    """Check if point is inside bbox."""
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def diagnose_from_run(run_dir: Path) -> Dict[str, Any]:
    """Diagnose issues from a completed run."""
    results = {
        "run_dir": str(run_dir),
        "issues": [],
        "stats": {},
    }
    
    # Check meta.json
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        results["issues"].append("ERROR: meta.json not found - run may have failed")
        return results
    
    with meta_path.open("r") as f:
        meta = json.load(f)
    
    # Check tracker report
    tracker_report = meta.get("tracker_report", {})
    results["stats"]["tracker_enabled"] = tracker_report.get("enabled", False)
    results["stats"]["total_tracks"] = tracker_report.get("total_tracks", 0)
    results["stats"]["unique_track_ids"] = tracker_report.get("unique_track_ids", 0)
    
    if not tracker_report.get("enabled"):
        results["issues"].append("WARNING: Tracker not enabled")
    
    if tracker_report.get("total_tracks", 0) == 0:
        results["issues"].append("ERROR: No tracks detected - detection or tracker failed")
    
    # Check detector report
    detector_report = meta.get("detector_report", {})
    results["stats"]["detector_enabled"] = detector_report.get("enabled", False)
    results["stats"]["total_detections"] = detector_report.get("total_detections", 0)
    
    if detector_report.get("total_detections", 0) == 0:
        results["issues"].append("ERROR: No detections - YOLOv8 failed or video empty")
    
    # Check MediaPipe report
    mp_report = meta.get("mediapipe_report", {})
    results["stats"]["mediapipe_enabled"] = mp_report.get("enabled", False)
    results["stats"]["mediapipe_rows"] = mp_report.get("total_rows", 0)
    
    if mp_report.get("enabled") and mp_report.get("total_rows", 0) == 0:
        results["issues"].append("ERROR: MediaPipe enabled but no features extracted")
    
    # Check assignment report
    assignment_report = meta.get("assignment_report", {})
    results["stats"]["zones_count"] = assignment_report.get("zones_count", 0)
    results["stats"]["assigned_tracks"] = assignment_report.get("assigned_tracks_total", 0)
    results["stats"]["unassigned_tracks"] = assignment_report.get("unassigned_tracks_total", 0)
    
    if assignment_report.get("zones_count", 0) > 0:
        assigned = assignment_report.get("assigned_tracks_total", 0)
        total = assigned + assignment_report.get("unassigned_tracks_total", 0)
        if total > 0:
            assign_rate = assigned / total
            results["stats"]["assignment_rate"] = f"{assign_rate:.1%}"
            if assign_rate < 0.5:
                results["issues"].append(f"WARNING: Low assignment rate ({assign_rate:.1%}) - zones may not match detections")
    
    # Analyze mediapipe.jsonl
    mp_path = run_dir / "features" / "mediapipe.jsonl"
    if mp_path.exists():
        mp_rows = read_jsonl(mp_path)
        
        # Check for track_bbox_xyxy
        with_bbox = sum(1 for r in mp_rows if r.get("track_bbox_xyxy"))
        without_bbox = len(mp_rows) - with_bbox
        
        results["stats"]["features_with_track_bbox"] = with_bbox
        results["stats"]["features_without_track_bbox"] = without_bbox
        
        if with_bbox == 0 and len(mp_rows) > 0:
            results["issues"].append("ERROR: No track_bbox_xyxy in features - dynamic ROI not working")
        
        # Check student_ids
        student_ids = set(str(r.get("student_id", "")) for r in mp_rows if r.get("student_id"))
        results["stats"]["unique_student_ids"] = list(student_ids)
        
        # Check roi_source
        roi_sources = defaultdict(int)
        for r in mp_rows:
            roi_sources[r.get("roi_source", "unknown")] += 1
        results["stats"]["roi_sources"] = dict(roi_sources)
    
    # Analyze behaviors.jsonl
    beh_path = run_dir / "events" / "behaviors.jsonl"
    if beh_path.exists():
        beh_rows = read_jsonl(beh_path)
        results["stats"]["total_behavior_events"] = len(beh_rows)
        
        # Count by behavior
        beh_counts = defaultdict(int)
        for r in beh_rows:
            beh_counts[r.get("behavior", "unknown")] += 1
        results["stats"]["behavior_counts"] = dict(beh_counts)
        
        # Check student coverage
        beh_students = set(str(r.get("student_id", "")) for r in beh_rows if r.get("student_id"))
        results["stats"]["students_with_behaviors"] = list(beh_students)
        
        if len(beh_rows) == 0:
            results["issues"].append("WARNING: No behavior events detected")
    else:
        results["issues"].append("INFO: behaviors.jsonl not found - run detect_behaviors.py")
    
    return results


def diagnose_zones_vs_detections(
    video_path: str,
    zones: List[Dict[str, Any]],
    sample_frames: int = 10,
) -> Dict[str, Any]:
    """Analyze if zones match actual detections."""
    results = {
        "video": video_path,
        "zones_count": len(zones),
        "issues": [],
        "zone_stats": {},
    }
    
    # Load YOLOv8
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
    except ImportError:
        results["issues"].append("ERROR: ultralytics not installed")
        return results
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        results["issues"].append(f"ERROR: Cannot open video: {video_path}")
        return results
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    # Per-zone stats
    for z in zones:
        zid = z["id"]
        results["zone_stats"][zid] = {
            "bbox": z["bbox"],
            "size": f"{z['bbox'][2]-z['bbox'][0]}x{z['bbox'][3]-z['bbox'][1]}",
            "detections_overlapping": 0,
            "detections_center_inside": 0,
            "best_iou": 0.0,
            "avg_iou": 0.0,
            "iou_list": [],
        }
    
    total_detections = 0
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run YOLOv8
        results_yolo = model.predict(frame, classes=[0], conf=0.25, verbose=False)
        
        for result in results_yolo:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det_bbox = (int(x1), int(y1), int(x2), int(y2))
                det_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                total_detections += 1
                
                # Check against each zone
                for z in zones:
                    zid = z["id"]
                    zone_bbox = z["bbox"]
                    
                    iou = calculate_iou(det_bbox, zone_bbox)
                    center_inside = center_in_bbox(det_center, zone_bbox)
                    
                    if iou > 0:
                        results["zone_stats"][zid]["detections_overlapping"] += 1
                        results["zone_stats"][zid]["iou_list"].append(iou)
                        if iou > results["zone_stats"][zid]["best_iou"]:
                            results["zone_stats"][zid]["best_iou"] = iou
                    
                    if center_inside:
                        results["zone_stats"][zid]["detections_center_inside"] += 1
    
    cap.release()
    
    # Calculate averages and identify issues
    results["total_detections_sampled"] = total_detections
    
    for zid, stats in results["zone_stats"].items():
        if stats["iou_list"]:
            stats["avg_iou"] = sum(stats["iou_list"]) / len(stats["iou_list"])
        del stats["iou_list"]  # Don't need in output
        
        # Check for issues
        if stats["detections_overlapping"] == 0:
            results["issues"].append(f"ERROR: Zone '{zid}' has NO overlapping detections - zone position wrong?")
        elif stats["best_iou"] < 0.3:
            results["issues"].append(f"WARNING: Zone '{zid}' best IoU is {stats['best_iou']:.2f} < 0.3 - zone too small?")
        
        if stats["detections_center_inside"] == 0:
            results["issues"].append(f"WARNING: Zone '{zid}' has no detection centers inside - zone misaligned?")
    
    return results


def visualize_zones_and_detections(
    video_path: str,
    zones: List[Dict[str, Any]],
    output_path: str,
    frame_idx: int = 100,
) -> None:
    """Create visualization of zones vs detections."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"ERROR: Cannot read frame {frame_idx}")
        return
    
    # Run YOLOv8
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        results = model.predict(frame, classes=[0], conf=0.25, verbose=False)
    except ImportError:
        print("ERROR: ultralytics not installed")
        return
    
    # Draw zones (red)
    for z in zones:
        bbox = z["bbox"]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        cv2.putText(frame, f"ZONE: {z['id']}", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw detections (green)
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"DET {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, frame)
    print(f"Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Zone Tracking Issues")
    parser.add_argument("--run_dir", type=str, help="Path to completed run directory")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--zones", type=str, help="Path to zones YAML file")
    parser.add_argument("--sample_frames", type=int, default=10, help="Number of frames to sample")
    parser.add_argument("--visualize", action="store_true", help="Create visualization image")
    parser.add_argument("--vis_frame", type=int, default=100, help="Frame index for visualization")
    parser.add_argument("--vis_out", type=str, default="diagnosis_zones.jpg", help="Visualization output path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Zone Tracking Diagnosis Tool")
    print("=" * 60)
    
    # Mode 1: Diagnose from completed run
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            sys.exit(1)
        
        print(f"\n[1] Analyzing run: {run_dir}")
        results = diagnose_from_run(run_dir)
        
        print("\n--- Run Statistics ---")
        for k, v in results["stats"].items():
            print(f"  {k}: {v}")
        
        print("\n--- Issues Found ---")
        if results["issues"]:
            for issue in results["issues"]:
                print(f"  {issue}")
        else:
            print("  No issues found!")
    
    # Mode 2: Analyze zones vs detections
    if args.video and args.zones:
        zones_path = Path(args.zones)
        if not zones_path.exists():
            print(f"ERROR: Zones file not found: {zones_path}")
            sys.exit(1)
        
        zones = load_zones_from_yaml(zones_path)
        print(f"\n[2] Analyzing zones vs detections...")
        print(f"    Video: {args.video}")
        print(f"    Zones: {args.zones} ({len(zones)} zones)")
        print(f"    Sampling: {args.sample_frames} frames")
        
        results = diagnose_zones_vs_detections(
            args.video, zones, args.sample_frames
        )
        
        print("\n--- Zone Statistics ---")
        for zid, stats in results["zone_stats"].items():
            print(f"\n  Zone '{zid}':")
            print(f"    Size: {stats['size']}")
            print(f"    Detections overlapping: {stats['detections_overlapping']}")
            print(f"    Detections center inside: {stats['detections_center_inside']}")
            print(f"    Best IoU: {stats['best_iou']:.3f}")
            print(f"    Avg IoU: {stats['avg_iou']:.3f}")
        
        print("\n--- Issues Found ---")
        if results["issues"]:
            for issue in results["issues"]:
                print(f"  {issue}")
        else:
            print("  No issues found!")
        
        # Create visualization
        if args.visualize:
            print(f"\n[3] Creating visualization...")
            visualize_zones_and_detections(
                args.video, zones, args.vis_out, args.vis_frame
            )
    
    # Show usage if no args
    if not args.run_dir and not (args.video and args.zones):
        print("\nUsage examples:")
        print("  # Diagnose from completed run:")
        print("  python tools/diagnose_zones.py --run_dir outputs/runs/xxx")
        print("")
        print("  # Analyze zones vs detections:")
        print("  python tools/diagnose_zones.py \\")
        print("    --video data/raw/SPU_DEMO.mp4 \\")
        print("    --zones configs/zones/spu_demo_30sec.yaml \\")
        print("    --sample_frames 20 \\")
        print("    --visualize")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

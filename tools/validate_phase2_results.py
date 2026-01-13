"""Validate Phase 2 outputs.

This script validates that all Phase 2 components are working correctly:
- Decision Engine outputs
- Alert Manager outputs
- Top Clips extraction
- Behavior aggregation

Usage:
    python tools/validate_phase2_results.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_file_exists(path: Path, name: str, errors: List[str], warnings: List[str]) -> bool:
    """Check if a file exists."""
    if not path.exists():
        errors.append(f"Missing: {name}")
        return False
    
    if path.stat().st_size == 0:
        warnings.append(f"Empty file: {name}")
        return False
    
    return True


def validate_jsonl(path: Path, name: str, errors: List[str]) -> List[Dict[str, Any]]:
    """Validate and load JSONL file."""
    if not path.exists():
        return []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data
    except Exception as e:
        errors.append(f"Failed to parse {name}: {e}")
        return []


def validate_json(path: Path, name: str, errors: List[str]) -> Dict[str, Any]:
    """Validate and load JSON file."""
    if not path.exists():
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse {name}: {e}")
        return {}


def validate_decisions(run_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate decision engine outputs."""
    errors = []
    warnings = []
    
    decisions_dir = run_dir / "decisions"
    
    # Check required files
    required_files = [
        ("risk_scores.jsonl", "Risk scores"),
        ("decisions.jsonl", "Decisions"),
        ("behavior_stats.json", "Behavior statistics"),
        ("summary.json", "Summary"),
    ]
    
    for filename, desc in required_files:
        path = decisions_dir / filename
        validate_file_exists(path, desc, errors, warnings)
    
    # Validate risk scores
    risk_scores = validate_jsonl(decisions_dir / "risk_scores.jsonl", "risk_scores.jsonl", errors)
    if risk_scores:
        print(f"✓ Risk scores: {len(risk_scores)} entries")
        
        # Check risk score range
        invalid_risks = [r for r in risk_scores if not (0 <= r.get('risk_score', -1) <= 1)]
        if invalid_risks:
            errors.append(f"Found {len(invalid_risks)} risk scores outside [0,1]")
    
    # Validate decisions
    decisions = validate_jsonl(decisions_dir / "decisions.jsonl", "decisions.jsonl", errors)
    if decisions:
        print(f"✓ Decisions: {len(decisions)} total")
        
        # Count by level
        levels = {}
        for d in decisions:
            level = d.get('decision_level', 'unknown')
            levels[level] = levels.get(level, 0) + 1
        
        print(f"  Decision levels:")
        for level, count in sorted(levels.items()):
            print(f"    - {level}: {count}")
    else:
        warnings.append("No decisions found - might be OK if video is clean")
    
    # Validate summary
    summary = validate_json(decisions_dir / "summary.json", "summary.json", errors)
    if summary:
        print(f"✓ Summary:")
        print(f"    Total decisions: {summary.get('total_decisions', 0)}")
        by_level = summary.get('by_level', {})
        if by_level:
            for level, count in sorted(by_level.items()):
                print(f"    {level}: {count}")
    
    return errors, warnings


def validate_clips(run_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate top clips extraction."""
    errors = []
    warnings = []
    
    decisions_dir = run_dir / "decisions"
    clips_path = decisions_dir / "top_clips.json"
    
    if not validate_file_exists(clips_path, "Top clips JSON", errors, warnings):
        return errors, warnings
    
    clips_data = validate_json(clips_path, "top_clips.json", errors)
    if not clips_data:
        return errors, warnings
    
    clips = clips_data if isinstance(clips_data, list) else []
    print(f"✓ Top clips: {len(clips)} found")
    
    # Validate each clip
    clips_dir = run_dir / "clips"
    for i, clip in enumerate(clips):
        # Check required fields
        required_fields = ['student_id', 'start_frame', 'end_frame', 'risk_score']
        missing = [f for f in required_fields if f not in clip]
        if missing:
            errors.append(f"Clip {i+1} missing fields: {missing}")
        
        # Check if video file exists (if clip_path is provided)
        if 'clip_path' in clip:
            clip_file = run_dir / clip['clip_path']
            if not clip_file.exists():
                errors.append(f"Missing clip file: {clip['clip_path']}")
        
        # Print clip info
        student = clip.get('student_id', 'unknown')
        risk = clip.get('risk_score', 0)
        reason = clip.get('reason', 'N/A')
        print(f"  Clip {i+1}: {student} (risk={risk:.2f}) - {reason}")
    
    return errors, warnings


def validate_behaviors(run_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate behavior detection outputs."""
    errors = []
    warnings = []
    
    events_dir = run_dir / "events"
    behaviors_path = events_dir / "behaviors.jsonl"
    
    if not validate_file_exists(behaviors_path, "Behaviors", errors, warnings):
        return errors, warnings
    
    behaviors = validate_jsonl(behaviors_path, "behaviors.jsonl", errors)
    if behaviors:
        print(f"✓ Behaviors: {len(behaviors)} events")
        
        # Count by type
        by_type = {}
        by_event = {}
        for b in behaviors:
            btype = b.get('behavior', 'unknown')
            etype = b.get('event_type', 'unknown')
            by_type[btype] = by_type.get(btype, 0) + 1
            by_event[etype] = by_event.get(etype, 0) + 1
        
        print(f"  By behavior:")
        for btype, count in sorted(by_type.items()):
            print(f"    - {btype}: {count}")
        
        print(f"  By event type:")
        for etype, count in sorted(by_event.items()):
            print(f"    - {etype}: {count}")
    
    return errors, warnings


def validate_render(run_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate rendered video outputs."""
    errors = []
    warnings = []
    
    demo_dir = run_dir / "demo"
    
    if not demo_dir.exists():
        warnings.append("No demo/ directory - render may not have been run")
        return errors, warnings
    
    # Look for any .mp4 files
    videos = list(demo_dir.glob("*.mp4"))
    if videos:
        print(f"✓ Rendered videos: {len(videos)}")
        for v in videos:
            size_mb = v.stat().st_size / (1024*1024)
            print(f"  - {v.name} ({size_mb:.1f} MB)")
    else:
        warnings.append("No rendered videos found in demo/")
    
    return errors, warnings


def validate(run_dir: str) -> int:
    """Main validation function."""
    rd = Path(run_dir)
    
    if not rd.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return 1
    
    print(f"=== Validating Phase 2 Results ===")
    print(f"Run directory: {run_dir}\n")
    
    all_errors = []
    all_warnings = []
    
    # Validate decisions
    print("--- Decision Engine ---")
    errs, warns = validate_decisions(rd)
    all_errors.extend(errs)
    all_warnings.extend(warns)
    print()
    
    # Validate behaviors
    print("--- Behaviors ---")
    errs, warns = validate_behaviors(rd)
    all_errors.extend(errs)
    all_warnings.extend(warns)
    print()
    
    # Validate clips
    print("--- Top Clips ---")
    errs, warns = validate_clips(rd)
    all_errors.extend(errs)
    all_warnings.extend(warns)
    print()
    
    # Validate render
    print("--- Render ---")
    errs, warns = validate_render(rd)
    all_errors.extend(errs)
    all_warnings.extend(warns)
    print()
    
    # Report
    if all_errors:
        print("❌ ERRORS:")
        for e in all_errors:
            print(f"  - {e}")
        print()
    
    if all_warnings:
        print("⚠️  WARNINGS:")
        for w in all_warnings:
            print(f"  - {w}")
        print()
    
    if all_errors:
        print("❌ Phase 2 validation FAILED!")
        return 1
    
    print("✅ Phase 2 validation PASSED!")
    if all_warnings:
        print("   (with warnings - review above)")
    
    return 0


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python tools/validate_phase2_results.py <run_dir>")
        print()
        print("Example:")
        print("  python tools/validate_phase2_results.py outputs/runs/20260113_075830_nogit_spu_demo")
        return 1
    
    return validate(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())

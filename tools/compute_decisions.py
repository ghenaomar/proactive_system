# tools/compute_decisions.py
"""
أداة حساب القرارات الشاملة
==========================
تحسب المخاطر + تصدر القرارات + تطلق التنبيهات

المخرجات:
- decisions/risk_scores.jsonl - نقاط المخاطر لكل فريم
- decisions/alerts.jsonl - التنبيهات من RiskScorer
- decisions/decisions.jsonl - القرارات من DecisionEngine
- decisions/behavior_stats.json - إحصائيات السلوكيات
- decisions/summary.json - ملخص شامل
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from proctor_ai.decisions.risk_scoring import RiskConfig, RiskScorer, iter_jsonl
from proctor_ai.decisions.decision_engine import (
    DecisionConfig,
    DecisionEngine,
    Decision,
    write_decisions_jsonl,
)
from proctor_ai.decisions.alert_manager import (
    AlertConfig,
    AlertManager,
    write_alerts_jsonl,
)
from proctor_ai.events.aggregator import (
    EventAggregator,
    BehaviorEvent,
    load_behaviors_jsonl,
    write_stats_json,
)


def _read_fps_from_meta(run_dir: str) -> Optional[float]:
    meta = os.path.join(run_dir, "meta.json")
    if not os.path.exists(meta):
        return None
    try:
        with open(meta, "r", encoding="utf-8") as f:
            m = json.load(f)
        for k in ("fps", "video_fps", "input_fps"):
            if k in m and m[k]:
                return float(m[k])
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute risk scores, decisions, and alerts")
    
    # المسارات
    ap.add_argument("--run_dir", required=True, help="مجلد الـ run")
    ap.add_argument("--features", default=None, help="default: <run_dir>/features/mediapipe.jsonl")
    ap.add_argument("--behaviors", default=None, help="default: <run_dir>/events/behaviors.jsonl")
    ap.add_argument("--out_dir", default=None, help="default: <run_dir>/decisions")

    # RiskScorer weights
    ap.add_argument("--w_look", type=float, default=0.45)
    ap.add_argument("--w_turn", type=float, default=0.45)
    ap.add_argument("--w_hand", type=float, default=0.10)

    # RiskScorer smoothing
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--ewma_alpha", type=float, default=0.85)

    # RiskScorer gating
    ap.add_argument("--min_face_px_for_turn", type=float, default=12.0)
    ap.add_argument("--require_presence", type=int, default=1)

    # RiskScorer alerts
    ap.add_argument("--alert_on", type=float, default=0.75)
    ap.add_argument("--alert_off", type=float, default=0.55)
    ap.add_argument("--hold_ms", type=int, default=1200)
    ap.add_argument("--release_ms", type=int, default=900)

    # DecisionEngine thresholds
    ap.add_argument("--risk_suspicious", type=float, default=0.35, help="عتبة المشتبه")
    ap.add_argument("--risk_high", type=float, default=0.60, help="عتبة الخطر العالي")
    ap.add_argument("--risk_confirmed", type=float, default=0.80, help="عتبة الغش المؤكد")
    ap.add_argument("--turn_count_suspicious", type=int, default=3, help="التفاتات مشتبهة/دقيقة")
    ap.add_argument("--turn_count_high", type=int, default=5, help="التفاتات عالية/دقيقة")
    ap.add_argument("--turn_count_confirmed", type=int, default=8, help="التفاتات مؤكدة/دقيقة")
    ap.add_argument("--decision_cooldown_sec", type=float, default=30.0, help="cooldown بين القرارات")

    # AlertManager
    ap.add_argument("--sound", type=int, default=0, help="تفعيل الصوت (1/0)")
    ap.add_argument("--sounds_dir", default="assets/sounds", help="مجلد الأصوات")

    ap.add_argument("--fps", type=float, default=0.0, help="override fps")
    
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    features_path = Path(args.features) if args.features else run_dir / "features" / "mediapipe.jsonl"
    behaviors_path = Path(args.behaviors) if args.behaviors else run_dir / "events" / "behaviors.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "decisions"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = args.fps or _read_fps_from_meta(str(run_dir)) or 25.0

    # ===== 1. RiskScorer =====
    print("=== Computing Risk Scores ===")
    
    risk_cfg = RiskConfig(
        w_look=args.w_look,
        w_turn=args.w_turn,
        w_hand=args.w_hand,
        window_sec=args.window_sec,
        ewma_alpha=args.ewma_alpha,
        min_face_px_for_turn=args.min_face_px_for_turn,
        require_presence=bool(args.require_presence),
        alert_on=args.alert_on,
        alert_off=args.alert_off,
        hold_ms=args.hold_ms,
        release_ms=args.release_ms,
    )

    scorer = RiskScorer(cfg=risk_cfg, fps=fps)
    risk_frames, risk_alerts = scorer.process_rows(iter_jsonl(str(features_path)))

    # كتابة risk scores
    risk_path = out_dir / "risk_scores.jsonl"
    with risk_path.open("w", encoding="utf-8") as f:
        for rf in risk_frames:
            f.write(json.dumps({
                "student_id": rf.student_id,
                "frame_index": rf.frame_index,
                "ts_ms": rf.ts_ms,
                "instant": rf.instant,
                "window_mean": rf.window_mean,
                "risk_score": rf.risk,
                "present": rf.present,
            }, ensure_ascii=False) + "\n")

    # كتابة risk alerts
    risk_alerts_path = out_dir / "risk_alerts.jsonl"
    with risk_alerts_path.open("w", encoding="utf-8") as f:
        for a in risk_alerts:
            f.write(json.dumps({
                "student_id": a.student_id,
                "type": "risk_alert",
                "start_frame": a.start_frame,
                "end_frame": a.end_frame,
                "start_ms": a.start_ms,
                "end_ms": a.end_ms,
                "peak_risk": a.peak_risk,
            }, ensure_ascii=False) + "\n")

    print(f"  Risk frames: {len(risk_frames)}")
    print(f"  Risk alerts: {len(risk_alerts)}")

    # ===== 2. Event Aggregator =====
    print("\n=== Aggregating Behaviors ===")
    
    aggregator = EventAggregator(window_sec=60.0)
    
    if behaviors_path.exists():
        behavior_count = 0
        for event in load_behaviors_jsonl(behaviors_path):
            aggregator.add_event(event)
            behavior_count += 1
        print(f"  Behavior events loaded: {behavior_count}")
    else:
        print(f"  Warning: behaviors file not found: {behaviors_path}")
        print("  Run detect_behaviors.py first to generate behaviors.jsonl")

    # كتابة إحصائيات السلوك
    stats = aggregator.get_all_stats()
    stats_path = out_dir / "behavior_stats.json"
    write_stats_json(stats, stats_path)
    print(f"  Students with stats: {len(stats)}")

    # ===== 3. Decision Engine =====
    print("\n=== Computing Decisions ===")
    
    decision_cfg = DecisionConfig(
        risk_threshold_suspicious=args.risk_suspicious,
        risk_threshold_high=args.risk_high,
        risk_threshold_confirmed=args.risk_confirmed,
        head_turn_count_suspicious=args.turn_count_suspicious,
        head_turn_count_high=args.turn_count_high,
        head_turn_count_confirmed=args.turn_count_confirmed,
        alert_cooldown_sec=args.decision_cooldown_sec,
    )

    decision_engine = DecisionEngine(decision_cfg)

    # إضافة السلوكيات للـ engine
    if behaviors_path.exists():
        for event in load_behaviors_jsonl(behaviors_path):
            decision_engine.add_behavior_event(
                student_id=event.student_id,
                behavior=event.behavior,
                event_type=event.event_type,
                ts_ms=event.ts_ms,
                score=event.score,
            )

    # تقييم كل نقطة risk
    decisions: list[Decision] = []
    for rf in risk_frames:
        dec = decision_engine.evaluate(
            student_id=rf.student_id,
            zone_id="",  # TODO: الحصول على zone_id من البيانات
            frame_index=rf.frame_index,
            ts_ms=rf.ts_ms if rf.ts_ms else 0,
            risk_score=rf.risk,
        )
        if dec:
            decisions.append(dec)

    # كتابة القرارات
    decisions_path = out_dir / "decisions.jsonl"
    write_decisions_jsonl(decisions, decisions_path)
    
    decision_summary = decision_engine.get_summary()
    print(f"  Total decisions: {decision_summary['total_decisions']}")
    print(f"  By level: {decision_summary['by_level']}")

    # ===== 4. Alert Manager =====
    print("\n=== Managing Alerts ===")
    
    alert_cfg = AlertConfig(
        sound_enabled=bool(args.sound),
        sounds_dir=args.sounds_dir,
    )
    
    alert_manager = AlertManager(alert_cfg)
    
    # تسجيل callback للطباعة
    def on_alert(record):
        severity_icon = {
            "info": "ℹ️",
            "low": "🟡",
            "medium": "🟠",
            "high": "🔴",
            "critical": "🚨",
        }.get(record.severity.value, "⚪")
        status = "🔇 suppressed" if record.suppressed else ("🔊" if record.sound_played else "")
        print(f"    {severity_icon} [{record.student_id}] {record.reason} {status}")
    
    alert_manager.on_alert(on_alert)
    
    # إطلاق التنبيهات للقرارات
    for dec in decisions:
        alert_manager.trigger_alert(dec, now_ms=dec.ts_ms)
    
    # كتابة التنبيهات
    alerts_path = out_dir / "alerts.jsonl"
    write_alerts_jsonl(alert_manager.get_alerts(), alerts_path)
    
    alert_summary = alert_manager.get_summary()
    print(f"  Total alerts: {alert_summary['total_alerts']}")
    print(f"  Suppressed: {alert_summary['suppressed']}")
    print(f"  Sound played: {alert_summary['sound_played']}")

    # ===== 5. Summary =====
    print("\n=== Writing Summary ===")
    
    per_student: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "risk_alerts": 0,
        "decisions": 0,
        "peak_risk": 0.0,
        "decision_levels": [],
    })
    
    for a in risk_alerts:
        per_student[a.student_id]["risk_alerts"] += 1
        per_student[a.student_id]["peak_risk"] = max(
            per_student[a.student_id]["peak_risk"],
            float(a.peak_risk)
        )
    
    for d in decisions:
        per_student[d.student_id]["decisions"] += 1
        per_student[d.student_id]["decision_levels"].append(d.decision_level.value)

    summary = {
        "fps": fps,
        "risk_config": risk_cfg.__dict__,
        "decision_config": decision_cfg.__dict__,
        "files": {
            "risk_scores": str(risk_path.relative_to(run_dir)),
            "risk_alerts": str(risk_alerts_path.relative_to(run_dir)),
            "decisions": str(decisions_path.relative_to(run_dir)),
            "alerts": str(alerts_path.relative_to(run_dir)),
            "behavior_stats": str(stats_path.relative_to(run_dir)),
        },
        "totals": {
            "risk_frames": len(risk_frames),
            "risk_alerts": len(risk_alerts),
            "decisions": len(decisions),
            "alerts_triggered": alert_summary["total_alerts"],
            "students": len(per_student),
        },
        "decision_summary": decision_summary,
        "alert_summary": alert_summary,
        "per_student": dict(per_student),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(f"Output directory: {out_dir}")
    print("Files written:")
    print(f"  - {risk_path}")
    print(f"  - {risk_alerts_path}")
    print(f"  - {decisions_path}")
    print(f"  - {alerts_path}")
    print(f"  - {stats_path}")
    print(f"  - {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

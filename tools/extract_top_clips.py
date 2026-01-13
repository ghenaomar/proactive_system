"""
Extract Top Clips - استخراج أهم المقاطع
========================================
يستخرج أهم المقاطع من الفيديو بناءً على:
1. نقاط المخاطر (risk_scores)
2. القرارات (decisions) - للحصول على سبب القرار

المخرجات:
- JSON مع معلومات المقاطع
- (اختياري) ملفات فيديو MP4 مستخرجة بـ ffmpeg
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_ffmpeg(
    *,
    video_path: str,
    start_ms: int,
    end_ms: int,
    out_path: str,
) -> bool:
    if end_ms <= start_ms:
        return False
    ss = max(0.0, float(start_ms) / 1000.0)
    to = max(ss, float(end_ms) / 1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{ss:.3f}",
        "-to",
        f"{to:.3f}",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        out_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def _load_decisions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """تحميل القرارات وتجميعها حسب الطالب والوقت"""
    if not path.exists():
        return {}
    
    decisions: Dict[str, List[Dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        sid = str(row.get("student_id", ""))
        if sid:
            decisions.setdefault(sid, []).append(row)
    
    # ترتيب حسب الوقت
    for sid in decisions:
        decisions[sid].sort(key=lambda x: int(x.get("ts_ms", 0)))
    
    return decisions


def _find_decision_for_segment(
    decisions: Dict[str, List[Dict[str, Any]]],
    student_id: str,
    start_ms: int,
    end_ms: int,
) -> Optional[Dict[str, Any]]:
    """البحث عن قرار ضمن فترة المقطع"""
    student_decisions = decisions.get(student_id, [])
    
    # البحث عن قرار في النطاق الزمني
    for dec in student_decisions:
        dec_ts = int(dec.get("ts_ms", 0))
        if start_ms <= dec_ts <= end_ms:
            return dec
    
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment risk timeline and optionally extract top clips.")
    ap.add_argument("--risk", required=True, help="decisions/risk_scores.jsonl")
    ap.add_argument("--out", required=True, help="decisions/top_clips.json")
    ap.add_argument("--decisions", default=None, help="decisions/decisions.jsonl (للحصول على سبب القرار)")
    ap.add_argument("--on", type=float, default=0.55, help="segment start threshold")
    ap.add_argument("--off", type=float, default=0.35, help="segment end threshold")
    ap.add_argument("--min_ms", type=int, default=1200, help="min segment duration")
    ap.add_argument("--max_clips", type=int, default=5)
    # Backwards/UX-friendly aliases
    ap.add_argument("--topk", type=int, default=None, help="alias for --max_clips")
    ap.add_argument("--pad_ms", type=int, default=400, help="context padding added before/after each segment when extracting")
    ap.add_argument("--video", default=None, help="optional input video for ffmpeg extraction")
    ap.add_argument("--clips_dir", default=None, help="output dir for extracted clips")
    # خيارات للتقارير
    ap.add_argument("--include_frames", action="store_true", help="extract key frames as images")
    ap.add_argument("--frames_dir", default=None, help="output dir for extracted frames")
    args = ap.parse_args()

    if args.topk is not None:
        args.max_clips = int(args.topk)

    risk_path = Path(args.risk)
    out_path = Path(args.out)
    rows = _read_jsonl(risk_path)
    
    # تحميل القرارات إن وجدت
    decisions_map: Dict[str, List[Dict[str, Any]]] = {}
    if args.decisions:
        decisions_path = Path(args.decisions)
        decisions_map = _load_decisions(decisions_path)
        print(f"Loaded decisions for {len(decisions_map)} students")

    # Group by student
    by_student: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = str(r.get("student_id", ""))
        if not sid:
            continue
        by_student.setdefault(sid, []).append(r)
    for sid, rr in by_student.items():
        rr.sort(key=lambda x: (int(x.get("ts_ms", 0)), int(x.get("frame_index", 0))))

    segments: List[Dict[str, Any]] = []

    for sid, rr in by_student.items():
        in_seg = False
        seg_start: Optional[Dict[str, Any]] = None
        peak_row: Optional[Dict[str, Any]] = None

        for r in rr:
            score = float(r.get("risk_score", 0.0) or 0.0)
            ts = int(r.get("ts_ms", 0) or 0)

            if not in_seg:
                if score >= float(args.on):
                    in_seg = True
                    seg_start = r
                    peak_row = r
                continue

            # in segment
            if peak_row is None or score > float(peak_row.get("risk_score", 0.0) or 0.0):
                peak_row = r

            if score <= float(args.off):
                # end segment here
                if seg_start is not None and peak_row is not None:
                    start_ts = int(seg_start.get("ts_ms", 0) or 0)
                    end_ts = ts
                    dur = end_ts - start_ts
                    if dur >= int(args.min_ms):
                        comp = (peak_row.get("components") or {})
                        reasons = []
                        for k in ("hand_to_face", "head_turn", "look_down", "combo"):
                            try:
                                v = float(comp.get(k, 0.0) or 0.0)
                            except Exception:
                                v = 0.0
                            if v >= 0.15:
                                reasons.append({"behavior": k, "evidence": round(v, 4)})

                        # البحث عن قرار مرتبط
                        related_decision = _find_decision_for_segment(
                            decisions_map, sid, start_ts, end_ts
                        )
                        
                        segment_data = {
                            "student_id": sid,
                            "zone_id": str(peak_row.get("zone_id", "")),
                            "start": {
                                "ts_ms": start_ts,
                                "frame_index": int(seg_start.get("frame_index", 0) or 0),
                            },
                            "end": {
                                "ts_ms": end_ts,
                                "frame_index": int(r.get("frame_index", 0) or 0),
                            },
                            "duration_ms": int(dur),
                            "peak": {
                                "ts_ms": int(peak_row.get("ts_ms", 0) or 0),
                                "frame_index": int(peak_row.get("frame_index", 0) or 0),
                                "risk_score": float(peak_row.get("risk_score", 0.0) or 0.0),
                                "components": comp,
                                "reasons": reasons,
                            },
                        }
                        
                        # إضافة معلومات القرار إن وجد
                        if related_decision:
                            segment_data["decision"] = {
                                "level": related_decision.get("decision_level", ""),
                                "severity": related_decision.get("alert_severity", ""),
                                "rule": related_decision.get("rule_name", ""),
                                "reason": related_decision.get("reason", ""),
                                "action": related_decision.get("action", ""),
                                "evidence": related_decision.get("evidence", {}),
                            }
                        
                        segments.append(segment_data)

                in_seg = False
                seg_start = None
                peak_row = None

        # If we ended while still in a segment, close it at last row
        if in_seg and seg_start is not None and peak_row is not None:
            last = rr[-1]
            end_ts = int(last.get("ts_ms", 0) or 0)
            start_ts = int(seg_start.get("ts_ms", 0) or 0)
            dur = end_ts - start_ts
            if dur >= int(args.min_ms):
                comp = (peak_row.get("components") or {})
                reasons = []
                for k in ("hand_to_face", "head_turn", "look_down", "combo"):
                    try:
                        v = float(comp.get(k, 0.0) or 0.0)
                    except Exception:
                        v = 0.0
                    if v >= 0.15:
                        reasons.append({"behavior": k, "evidence": round(v, 4)})
                
                # البحث عن قرار مرتبط
                related_decision = _find_decision_for_segment(
                    decisions_map, sid, start_ts, end_ts
                )
                
                segment_data = {
                    "student_id": sid,
                    "zone_id": str(peak_row.get("zone_id", "")),
                    "start": {
                        "ts_ms": start_ts,
                        "frame_index": int(seg_start.get("frame_index", 0) or 0),
                    },
                    "end": {
                        "ts_ms": end_ts,
                        "frame_index": int(last.get("frame_index", 0) or 0),
                    },
                    "duration_ms": int(dur),
                    "peak": {
                        "ts_ms": int(peak_row.get("ts_ms", 0) or 0),
                        "frame_index": int(peak_row.get("frame_index", 0) or 0),
                        "risk_score": float(peak_row.get("risk_score", 0.0) or 0.0),
                        "components": comp,
                        "reasons": reasons,
                    },
                }
                
                # إضافة معلومات القرار إن وجد
                if related_decision:
                    segment_data["decision"] = {
                        "level": related_decision.get("decision_level", ""),
                        "severity": related_decision.get("alert_severity", ""),
                        "rule": related_decision.get("rule_name", ""),
                        "reason": related_decision.get("reason", ""),
                        "action": related_decision.get("action", ""),
                        "evidence": related_decision.get("evidence", {}),
                    }
                
                segments.append(segment_data)

    # Rank segments and keep top-K
    segments.sort(key=lambda x: float(x.get("peak", {}).get("risk_score", 0.0) or 0.0), reverse=True)
    top = segments[: int(args.max_clips)]

    # Optionally extract actual video clips
    extracted: List[Dict[str, Any]] = []
    if args.video and args.clips_dir:
        clips_dir = Path(args.clips_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(top, start=1):
            pad = int(args.pad_ms)
            start_ms = int(s["start"]["ts_ms"]) - pad
            end_ms = int(s["end"]["ts_ms"]) + pad
            out_fn = f"clip_{i:02d}_sid_{s['student_id']}_peak_{float(s['peak']['risk_score']):.2f}.mp4"
            out_fp = str(clips_dir / out_fn)
            ok = _extract_ffmpeg(video_path=str(args.video), start_ms=start_ms, end_ms=end_ms, out_path=out_fp)
            extracted.append({"clip": out_fp, "ok": bool(ok), "segment": s})

    payload = {
        "params": {
            "on": float(args.on),
            "off": float(args.off),
            "min_ms": int(args.min_ms),
            "max_clips": int(args.max_clips),
            "pad_ms": int(args.pad_ms),
        },
        "segments_total": int(len(segments)),
        "top_segments": top,
        "extracted": extracted,
    }

    _write_json(out_path, payload)
    print(f"OK: wrote top clips -> {out_path}")
    if extracted:
        ok_n = sum(1 for e in extracted if e.get("ok"))
        print(f"OK: extracted clips -> {args.clips_dir} ({ok_n}/{len(extracted)} ok)")


if __name__ == "__main__":
    main()

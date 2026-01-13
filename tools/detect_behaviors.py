from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    import os

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _get_optional_float(d: Dict[str, Any], key: str) -> Optional[float]:
    """
    Return None if key absent or value None.
    Return float(value) if present.
    """
    if key not in d:
        return None
    v = d.get(key, None)
    if v is None:
        return None
    return _as_float(v, 0.0)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _yaw_to_lr_scores(yaw: Optional[float], yaw_on: float, yaw_full: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert yaw (signed) into left/right scores in [0..1].
    - Right score increases when yaw is positive.
    - Left score increases when yaw is negative.
    """
    if yaw is None:
        return None, None
    den = float(yaw_full - yaw_on)
    if den <= 1e-6:
        return None, None
    y = float(yaw)
    right = _clamp01((y - yaw_on) / den)
    left = _clamp01((-y - yaw_on) / den)
    return left, right


@dataclass
class Gate:
    on: float
    off: float


@dataclass
class TrackState:
    active: bool = False
    last_change_ms: int = 0
    last_valid_ms: int = 0


def _update_hysteresis(
    *,
    st: TrackState,
    value: Optional[float],         # None => no update (missing face/pose or missing key)
    ts_ms: int,
    gate: Gate,
    enter_ms: int,
    exit_ms: int,
    gap_ms: int,
) -> Tuple[Optional[str], float]:
    """
    Returns (event_type or None, used_score)
    """
    # if missing for long => end safely
    if value is None:
        if st.active and (ts_ms - st.last_valid_ms) >= gap_ms:
            st.active = False
            st.last_change_ms = ts_ms
            return "end", 0.0
        return None, 0.0

    st.last_valid_ms = ts_ms
    v = float(value)

    if not st.active:
        if v >= gate.on:
            if (ts_ms - st.last_change_ms) >= enter_ms:
                st.active = True
                st.last_change_ms = ts_ms
                return "start", v
        return None, v

    # active
    if v <= gate.off:
        if (ts_ms - st.last_change_ms) >= exit_ms:
            st.active = False
            st.last_change_ms = ts_ms
            return "end", v
    return None, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)

    ap.add_argument("--enter_ms", type=int, default=200)
    ap.add_argument("--exit_ms", type=int, default=400)
    ap.add_argument("--gap_ms", type=int, default=700)

    # Optional per-behavior timings (leave unset to use global enter/exit)
    ap.add_argument("--turn_enter_ms", type=int, default=350, help="head-turn min duration to start")
    ap.add_argument("--turn_exit_ms", type=int, default=500, help="head-turn min duration to end")

    ap.add_argument("--look_on", type=float, default=0.58)
    ap.add_argument("--look_off", type=float, default=0.52)

    # for left/right use same thresholds
    # NOTE: defaults are conservative to reduce false positives in far CCTV.
    ap.add_argument("--turn_on", type=float, default=0.55)
    ap.add_argument("--turn_off", type=float, default=0.45)

    # Face-size gate (reduces false head-turns when faces are tiny/partial in CCTV)
    ap.add_argument("--min_face_turn_px", type=float, default=22.0)
    ap.add_argument("--min_face_turn_px_pose", type=float, default=16.0, help="min face size (px) to allow pose-based turn proxy")
    ap.add_argument("--allow_pose_turn", action="store_true", help="allow pose-based yaw proxy for head-turn when FaceMesh is missing")

    ap.add_argument("--htf_on", type=float, default=0.55)
    ap.add_argument("--htf_off", type=float, default=0.45)

    # Optional EMA smoothing on scores to reduce jitter/false positives in CCTV.
    # 0 => disabled. Typical values: 0.25 - 0.45
    ap.add_argument("--ema_alpha", type=float, default=0.35)
    # Optional per-behavior EMA (useful to keep look_down stable but make head_turn more responsive)
    ap.add_argument("--ema_alpha_turn", type=float, default=-1.0, help="EMA alpha for head_turn (default: use --ema_alpha)")

    # Short hold to bridge FaceMesh dropouts during turning (ms). 0 disables.
    ap.add_argument("--turn_hold_ms", type=int, default=150, help="reuse last head_turn score for brief dropouts (ms)")

    # === NEW: Yaw calibration for angled cameras ===
    ap.add_argument("--yaw_calib", type=str, default="", help="JSON map {student_id: baseline_yaw_norm} to subtract")
    ap.add_argument("--use_yaw_norm_for_turn", action="store_true", help="compute head-turn from (yaw_norm - baseline)")
    ap.add_argument("--yaw_on_norm", type=float, default=0.22, help="turn starts when |yaw_corr| >= yaw_on_norm")
    ap.add_argument("--yaw_full_norm", type=float, default=0.42, help="turn score reaches 1 when |yaw_corr| >= yaw_full_norm")
    ap.add_argument("--turn_suppress_lookdown", type=float, default=0.65, help="if look_down >= this, suppress head-turn (reduce FP while writing)")

    args = ap.parse_args()

    alpha_global = float(args.ema_alpha)
    alpha_turn = float(alpha_global if float(args.ema_alpha_turn) < 0.0 else float(args.ema_alpha_turn))

    # Yaw calibration loading
    yaw_calib: Dict[str, float] = {}
    if args.yaw_calib:
        with open(args.yaw_calib, "r", encoding="utf-8") as f:
            yaw_calib = json.load(f) or {}
    use_yaw_for_turn = bool(args.use_yaw_norm_for_turn or bool(args.yaw_calib))

    # per-student EMA state (behavior -> value)
    # and last-valid raw scores for short dropout bridging (head_turn only)
    last_valid_turn: Dict[str, Dict[str, Tuple[int, float]]] = {}

    rows = _read_jsonl(args.inp)

    # per-student states
    states: Dict[str, Dict[str, TrackState]] = {}

    def get_state(sid: str, key: str) -> TrackState:
        if sid not in states:
            states[sid] = {}
        if key not in states[sid]:
            states[sid][key] = TrackState(active=False, last_change_ms=0, last_valid_ms=0)
        return states[sid][key]

    look_gate = Gate(on=args.look_on, off=args.look_off)
    turn_gate = Gate(on=args.turn_on, off=args.turn_off)
    htf_gate = Gate(on=args.htf_on, off=args.htf_off)

    out_events: List[Dict[str, Any]] = []

    max_scores: Dict[str, Dict[str, float]] = {}

    def bump_max(sid: str, name: str, v: float) -> None:
        if sid not in max_scores:
            max_scores[sid] = {"look_down": 0.0, "head_turn_left": 0.0, "head_turn_right": 0.0, "hand_to_face": 0.0}
        max_scores[sid][name] = max(max_scores[sid][name], float(v))

    counts: Dict[str, Dict[str, int]] = {}

    # EMA state per (student, behavior)
    ema: Dict[str, Dict[str, float]] = {}

    def smooth_score(sid: str, beh: str, v: Optional[float]) -> Optional[float]:
        """Optional EMA smoothing; returns None if input score missing."""
        if v is None:
            return None

        a = alpha_turn if beh.startswith("head_turn") else alpha_global
        if a <= 0.0:
            return float(v)

        if sid not in ema:
            ema[sid] = {}
        prev = ema[sid].get(beh)
        cur = float(v)
        if prev is None:
            ema[sid][beh] = cur
        else:
            ema[sid][beh] = (a * cur) + ((1.0 - a) * float(prev))
        return float(ema[sid][beh])

    for r in rows:
        sid = str(r.get("student_id"))
        zid = str(r.get("zone_id"))
        frame_index = int(r.get("frame_index", 0))
        ts_ms = int(r.get("ts_ms", 0))

        # Flags
        face_mesh_ok = bool(r.get("face_detected", False))
        face_proxy_ok = bool(r.get("face_proxy_detected", False))
        pose_ok = bool(r.get("pose_detected", False))

        face_mesh_usable = bool(face_mesh_ok and face_proxy_ok)

        # Face/Proxy size
        face_sz = _get_optional_float(r, "face_size_px") if face_proxy_ok else None

        # Look down:
        look_val: Optional[float] = None
        if face_mesh_usable:
            look_val = _get_optional_float(r, "look_down_score")
        elif pose_ok:
            look_val = _get_optional_float(r, "look_down_pose_score")

        # Head turns gate (face size + face/pose availability)
        can_turn = bool(face_mesh_usable and (face_sz is not None) and (float(face_sz) >= float(args.min_face_turn_px)))
        if (not can_turn) and bool(args.allow_pose_turn):
            can_turn = bool(
                pose_ok
                and face_proxy_ok
                and (face_sz is not None)
                and (float(face_sz) >= float(args.min_face_turn_px_pose))
            )

        # NEW: suppress turn while strongly looking down (normal writing posture)
        suppress_turn = (look_val is not None) and (float(look_val) >= float(args.turn_suppress_lookdown))

        left_val: Optional[float] = None
        right_val: Optional[float] = None

        if can_turn and (not suppress_turn):
            if use_yaw_for_turn:
                yaw = _get_optional_float(r, "yaw_norm")
                if yaw is not None:
                    yaw -= float(yaw_calib.get(sid, 0.0))
                    # CCTV note: when FaceMesh fails, yaw_norm may come from the Pose proxy.
                    # That proxy can still be jittery even after scaling; clamp it to avoid
                    # instant saturation and "everyone is turning" false positives.
                    if not face_mesh_usable:
                        if yaw > 0.8:
                            yaw = 0.8
                        elif yaw < -0.8:
                            yaw = -0.8
                left_val, right_val = _yaw_to_lr_scores(yaw, float(args.yaw_on_norm), float(args.yaw_full_norm))
            else:
                left_val = _get_optional_float(r, "head_turn_left_score")
                right_val = _get_optional_float(r, "head_turn_right_score")

        # Brief dropout bridging
        if int(args.turn_hold_ms) > 0:
            if sid not in last_valid_turn:
                last_valid_turn[sid] = {}

            if left_val is not None:
                last_valid_turn[sid]["head_turn_left"] = (ts_ms, float(left_val))
            else:
                prev = last_valid_turn[sid].get("head_turn_left")
                if prev and (ts_ms - int(prev[0])) <= int(args.turn_hold_ms):
                    left_val = float(prev[1])

            if right_val is not None:
                last_valid_turn[sid]["head_turn_right"] = (ts_ms, float(right_val))
            else:
                prev = last_valid_turn[sid].get("head_turn_right")
                if prev and (ts_ms - int(prev[0])) <= int(args.turn_hold_ms):
                    right_val = float(prev[1])

        # Hand-to-face (pose only)
        htf_val = _get_optional_float(r, "hand_to_face_score") if pose_ok else None

        # Max scores logging (only when valid)
        if look_val is not None:
            bump_max(sid, "look_down", look_val)
        if left_val is not None:
            bump_max(sid, "head_turn_left", left_val)
        if right_val is not None:
            bump_max(sid, "head_turn_right", right_val)
        if htf_val is not None:
            bump_max(sid, "hand_to_face", htf_val)

        # Mutual exclusion: avoid left+right flipping in noisy yaw.
        left_state = get_state(sid, "head_turn_left")
        right_state = get_state(sid, "head_turn_right")
        if left_state.active:
            right_val = None
        elif right_state.active:
            left_val = None

        updates = [
            ("look_down", look_val, look_gate, int(args.enter_ms), int(args.exit_ms)),
            ("head_turn_left", left_val, turn_gate, int(args.turn_enter_ms), int(args.turn_exit_ms)),
            ("head_turn_right", right_val, turn_gate, int(args.turn_enter_ms), int(args.turn_exit_ms)),
            ("hand_to_face", htf_val, htf_gate, int(args.enter_ms), int(args.exit_ms)),
        ]

        for beh, raw_val, gate, enter_ms, exit_ms in updates:
            val = smooth_score(sid, beh, raw_val)
            st = get_state(sid, beh)
            ev_type, used_score = _update_hysteresis(
                st=st,
                value=val,
                ts_ms=ts_ms,
                gate=gate,
                enter_ms=int(enter_ms),
                exit_ms=int(exit_ms),
                gap_ms=int(args.gap_ms),
            )
            if ev_type is None:
                continue

            out_events.append(
                {
                    "student_id": sid,
                    "zone_id": zid,
                    "frame_index": frame_index,
                    "ts_ms": ts_ms,
                    "behavior": beh,
                    "event_type": ev_type,
                    "score": float(used_score),
                }
            )

            if sid not in counts:
                counts[sid] = {}
            k = f"{beh}_{ev_type}"
            counts[sid][k] = int(counts[sid].get(k, 0)) + 1

    _write_jsonl(args.out, out_events)

    print(f"Wrote: {args.out}")
    print(f"Total behavior events: {len(out_events)}")
    if max_scores:
        print("Max scores seen (per student):")
        for sid in sorted(max_scores):
            print(" ", sid, max_scores[sid])
    if counts:
        print("Counts per student:")
        for sid in sorted(counts):
            print(" ", sid, counts[sid])


if __name__ == "__main__":
    main()

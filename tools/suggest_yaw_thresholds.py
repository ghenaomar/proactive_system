from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict, List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to features/mediapipe.jsonl")
    ap.add_argument("--calib", required=True, help="Path to calibration json {student_id: baseline_yaw}")
    ap.add_argument("--look_th", type=float, default=0.60)
    ap.add_argument("--margin", type=float, default=0.03, help="Added margin over p95")
    ap.add_argument("--span", type=float, default=0.16, help="yaw_full = yaw_on + span")
    ap.add_argument("--require_facemesh", action="store_true", help="Only use FaceMesh yaw (ignore Pose proxy yaw)")
    ap.add_argument("--min_face_px", type=float, default=0.0, help="Ignore rows with face_size_px < this (0 disables)")
    args = ap.parse_args()

    baseline: Dict[str, float] = json.load(open(args.calib, "r", encoding="utf-8"))

    vals: List[float] = []
    with open(args.features, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r: Dict[str, Any] = json.loads(s)

            if not r.get("face_proxy_detected"):
                continue

            if float(args.min_face_px) > 0.0:
                try:
                    if float(r.get("face_size_px", 0.0) or 0.0) < float(args.min_face_px):
                        continue
                except Exception:
                    continue

            if bool(args.require_facemesh) and not r.get("face_detected"):
                continue
            if not (r.get("pose_detected") or r.get("face_detected")):
                continue

            if r.get("face_detected") and r.get("face_proxy_detected"):
                look = float(r.get("look_down_score", 0.0) or 0.0)
            else:
                look = float(r.get("look_down_pose_score", 0.0) or 0.0)

            if look < float(args.look_th):
                continue

            yaw = r.get("yaw_norm", None)
            if yaw is None:
                continue

            sid = str(r.get("student_id"))
            yaw_corr = float(yaw) - float(baseline.get(sid, 0.0))
            vals.append(abs(yaw_corr))

    if len(vals) < 50:
        print("Too few samples:", len(vals))
        return

    vals.sort()
    p95 = vals[int(0.95 * len(vals))]
    p99 = vals[int(0.99 * len(vals))]

    yaw_on = round(float(p95) + float(args.margin), 3)
    yaw_full = round(float(yaw_on) + float(args.span), 3)

    print(f"Samples: {len(vals)}")
    print(f"abs(yaw_corr) p95= {round(p95,3)} p99= {round(p99,3)}")
    print(f"Suggested: --yaw_on_norm {yaw_on} --yaw_full_norm {yaw_full}")


if __name__ == "__main__":
    main()

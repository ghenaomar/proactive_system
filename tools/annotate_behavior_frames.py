import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def read_jsonl(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args=ap.parse_args()

    run_dir=Path(args.run_dir)
    frames_dir=Path(args.frames_dir)
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = read_jsonl(run_dir/"features"/"mediapipe.jsonl")
    behs  = read_jsonl(run_dir/"events"/"behaviors.jsonl")

    # (student_id, frame_index) -> roi_xyxy
    roi_map={}
    for r in feats:
        sid=r.get("student_id")
        fi=int(r.get("frame_index", -1))
        roi=r.get("roi_xyxy")
        if sid and fi >= 0 and isinstance(roi, list) and len(roi)==4:
            roi_map[(sid, fi)] = roi

    # حاول نجيب فونت افتراضي
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ev in behs:
        sid=ev["student_id"]
        fi=int(ev["frame_index"])
        img_path = frames_dir / f"frame_{fi}.png"
        if not img_path.exists():
            continue

        im=Image.open(img_path).convert("RGB")
        d=ImageDraw.Draw(im)

        roi = roi_map.get((sid, fi))
        if roi:
            x1,y1,x2,y2 = map(int, roi)
            d.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)

        label=f'{sid} {ev["behavior"]}:{ev["event_type"]} score={ev.get("score",0):.2f}'
        d.rectangle([5,5,5+len(label)*7,25], fill=(0,0,0))
        d.text((8,8), label, fill=(255,255,255), font=font)

        out_path = out_dir / f'{sid}_f{fi}_{ev["behavior"]}_{ev["event_type"]}.png'
        im.save(out_path)

    print("Wrote annotated frames to:", out_dir)

if __name__ == "__main__":
    main()

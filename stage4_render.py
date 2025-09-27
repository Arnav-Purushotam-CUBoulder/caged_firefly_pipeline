"""
Stage 4: Render final annotated videos and write final CSVs.

Reads Stage3 CSVs (frame,cx,cy,x1,y1,x2,y2,conf), draws rectangles on
the original video frames, and writes annotated videos per input video.
Also writes per-video final CSVs (copied from Stage3) in STAGE4_DIR.
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import cv2

import params


def _load_stage3(path: Path) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """Return mapping: frame -> list of (x1, y1, x2, y2, conf)."""
    mapping: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            x1, y1, x2, y2 = map(float, (row['x1'], row['y1'], row['x2'], row['y2']))
            conf = float(row['conf'])
            mapping.setdefault(fi, []).append((x1, y1, x2, y2, conf))
    return mapping


def run_stage4() -> Path:
    out_dir = params.STAGE4_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map video stem -> video path
    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    for csv_path in sorted(params.STAGE3_DIR.glob('*_merged.csv')):
        stem = csv_path.stem.replace('_merged', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"Warning: no matching video for {csv_path.name}")
            continue
        by_frame = _load_stage3(csv_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: cannot open video {vpath}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video = out_dir / f"{stem}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*params.FOURCC),
            fps,
            (w, h),
            True,
        )

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        for idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in by_frame:
                for x1, y1, x2, y2, conf in by_frame[idx]:
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    cv2.rectangle(frame, p1, p2, params.BOX_COLOR_BGR, params.BOX_THICKNESS)
            writer.write(frame)

        cap.release()
        writer.release()

        # Write final CSV (copy of Stage3 CSV) alongside annotated video
        out_csv = out_dir / f"{stem}_final.csv"
        out_csv.write_bytes(csv_path.read_bytes())

    return out_dir


__all__ = ['run_stage4']

"""
Renderer: draw boxes from latest stage CSVs and write annotated videos.

Reads Stage4 Gaussian CSVs (frame,x,y,w,h,conf,xy_semantics='center'),
or falls back to Stage3 merged CSVs (frame,cx,cy,x1,y1,x2,y2,conf),
draws rectangles on the original video frames, and writes annotated
videos per input video into STAGE_RENDERER_DIR. Also copies the CSV
used into the same folder for reference.
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional

import cv2

import params


def _ensure_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # handle 0 or NaN
        fps = 30.0
    return float(fps)


def _ensure_size(cap: cv2.VideoCapture) -> Tuple[int, int]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return int(w), int(h)


def _make_writer(out_dir: Path, stem: str, w: int, h: int, fps: float) -> Tuple[cv2.VideoWriter, Path, str]:
    """Create a VideoWriter with codec fallbacks. Returns (writer, path, fourcc)."""
    candidates = [
        ('mp4v', '.mp4'),
        ('avc1', '.mp4'),
        ('H264', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
    ]
    last_exc: Optional[Exception] = None
    for fourcc, ext in candidates:
        out_video = out_dir / f"{stem}_annotated{ext}"
        try:
            writer = cv2.VideoWriter(
                str(out_video),
                cv2.VideoWriter_fourcc(*fourcc),
                float(fps),
                (int(w), int(h)),
                True,
            )
            if writer.isOpened():
                return writer, out_video, fourcc
            writer.release()
        except Exception as e:  # pragma: no cover (safety)
            last_exc = e
    raise RuntimeError(f"Failed to create VideoWriter for {stem}; last error: {last_exc}")


def _load_rects_from_csv(path: Path) -> Dict[int, List[Tuple[float,float,float,float,float]]]:
    """Return mapping: frame -> list of rectangles (x1,y1,x2,y2,conf).
    Supports Stage4 (center semantics) and Stage3 (corner semantics).
    """
    mapping: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        use_stage4 = all(c in cols for c in ('x','y','w','h'))
        use_stage3 = all(c in cols for c in ('x1','y1','x2','y2'))
        for row in reader:
            fi = int(row['frame'])
            conf = float(row.get('conf', 'nan'))
            if use_stage4:
                w = int(float(row['w'])); h = int(float(row['h']))
                cx = float(row['x']); cy = float(row['y'])
                x1 = cx - w/2.0; y1 = cy - h/2.0
                x2 = x1 + w;     y2 = y1 + h
            elif use_stage3:
                x1 = float(row['x1']); y1 = float(row['y1'])
                x2 = float(row['x2']); y2 = float(row['y2'])
            else:
                continue
            mapping.setdefault(fi, []).append((x1, y1, x2, y2, conf))
    return mapping


def run_stage_renderer() -> Path:
    out_dir = params.STAGE_RENDERER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map video stem -> video path
    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    # Prefer Stage4 outputs; fall back to Stage3 if Stage4 missing
    stage4_csvs = list(sorted(params.STAGE4_DIR.glob('*_gauss.csv')))
    stage3_csvs = list(sorted(params.STAGE3_DIR.glob('*_merged.csv')))
    csv_paths = stage4_csvs if stage4_csvs else stage3_csvs

    for csv_path in csv_paths:
        stem = csv_path.stem.replace('_gauss', '').replace('_merged', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"Warning: no matching video for {csv_path.name}")
            continue
        by_frame = _load_rects_from_csv(csv_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: cannot open video {vpath}")
            continue
        fps = _ensure_fps(cap)
        w, h = _ensure_size(cap)

        writer, out_video, used_fourcc = _make_writer(out_dir, stem, w, h, fps)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        drawn = 0
        for idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in by_frame:
                for x1, y1, x2, y2, conf in by_frame[idx]:
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    cv2.rectangle(frame, p1, p2, params.BOX_COLOR_BGR, params.BOX_THICKNESS)
                    drawn += 1
            writer.write(frame)

        cap.release()
        writer.release()

        # Copy CSV used alongside annotated video
        out_csv = out_dir / f"{csv_path.name}"
        out_csv.write_bytes(csv_path.read_bytes())
        print(f"Renderer: {stem} frames={total} boxes_drawn={drawn} codec={used_fourcc} → {out_video}")

    return out_dir


__all__ = ['run_stage_renderer']

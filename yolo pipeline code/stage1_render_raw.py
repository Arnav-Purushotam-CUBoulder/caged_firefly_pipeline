#!/usr/bin/env python3
"""
Stage 1 (aux): render annotated videos from raw YOLO detections.

For each input video, reads the Stage 1 CSV:
  STAGE1_DIR/<video_stem>/<video_stem>.csv

and writes an annotated video with the raw YOLO boxes drawn under:
  STAGE1_DIR/<video_stem>/<video_stem>_yolo_raw.mp4

This lets you see exactly what YOLO outputs before any Stage 2/3 filtering.
"""
from __future__ import annotations

from pathlib import Path
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2

import params


def _open_video(path: Path):
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, fps, count


def _make_writer(path: Path, w: int, h: int, fps: float, codec: str = "mp4v"):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def run_for_video(video_path: Path) -> Path:
    """Render annotated video using boxes from Stage 1 YOLO detection CSV."""
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    s1_csv = (params.STAGE1_DIR / stem) / f"{stem}.csv"
    if not s1_csv.exists():
        raise FileNotFoundError(f"Missing Stage1 CSV for {stem}: {s1_csv}")

    out_root = params.STAGE1_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_video = out_root / f"{stem}_yolo_raw.mp4"

    # Load boxes by frame: frame -> list[(x,y,w,h)]
    boxes_by_t: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    with s1_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frame = int(row["frame"])
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                w = int(float(row["w"]))
                h = int(float(row["h"]))
            except Exception:
                continue
            boxes_by_t[frame].append((x, y, w, h))

    cap, W, H, fps_src, total = _open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (getattr(params, "MAX_FRAMES", None) is not None) else total
    fps = float(params.RENDER_FPS_HINT or fps_src)
    writer = _make_writer(out_video, W, H, fps, codec=params.RENDER_CODEC)

    t = 0
    boxes_drawn = 0
    try:
        while t < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if t in boxes_by_t:
                for (x, y, w, h) in boxes_by_t[t]:
                    x0 = int(x)
                    y0 = int(y)
                    x1 = x0 + int(w)
                    y1 = y0 + int(h)
                    cv2.rectangle(
                        frame,
                        (x0, y0),
                        (x1, y1),
                        params.BOX_COLOR_BGR,
                        int(params.BOX_THICKNESS),
                        cv2.LINE_AA,
                    )
                    boxes_drawn += 1
            writer.write(frame)
            t += 1
    finally:
        cap.release()
        writer.release()

    print(
        f"Stage1_raw  Rendered {t} frame(s) for {video_path.name}; "
        f"frames_with_boxes={len(boxes_by_t)}; boxes_drawn={boxes_drawn}"
    )
    print(f"Stage1_raw  Wrote raw YOLO annotated video → {out_video}")
    return out_video


__all__ = ["run_for_video"]


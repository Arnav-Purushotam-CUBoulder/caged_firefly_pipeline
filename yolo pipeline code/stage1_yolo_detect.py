#!/usr/bin/env python3
"""
Stage 1: YOLO detection on raw videos.

For each input video, runs a YOLO model frame-by-frame and writes a CSV
per video under:

  STAGE1_DIR/<video_stem>/<video_stem>.csv

CSV schema:
  x, y, w, h, frame, video_name, firefly_logit, background_logit

where (x,y,w,h) is a box in top-left + width/height pixel coordinates,
frame is 0-based frame index, and the 'logit' fields are populated from
the YOLO confidence (firefly_logit = conf, background_logit = 1 - conf).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import params

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Stage 1 YOLO detection requires ultralytics. Install with: pip install ultralytics\n"
        f"Import error: {e}"
    ) from e


def _progress(i: int, total: int, tag: str = "") -> None:
    """Simple CLI progress bar."""
    total = max(int(total or 1), 1)
    i = min(int(i), total)
    bar_w = 36
    frac = i / total
    fill = int(frac * bar_w)
    bar = "█" * fill + "·" * (bar_w - fill)
    sys.stdout.write(f"\r[{bar}] {i}/{total} {tag}")
    if i == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _format_eta(seconds: float) -> str:
    """Format ETA seconds as H:MM:SS or M:SS."""
    seconds = max(0, int(seconds + 0.5))
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _select_device(user_choice):
    import torch

    # Respect explicit non-auto choices first
    if isinstance(user_choice, int):
        return user_choice
    if isinstance(user_choice, str) and user_choice.lower() not in {"auto", "", "none"}:
        return user_choice

    # Auto detection
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return 0
    return "cpu"


def run_for_video(video_path: Path) -> Path:
    """Run YOLO on all frames of a video, write per-video CSV, and return its path."""
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    out_root = params.STAGE1_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{stem}.csv"

    weights = Path(params.YOLO_MODEL_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"YOLO_MODEL_WEIGHTS not found: {weights}")

    model = YOLO(str(weights))

    img_size_cfg = getattr(params, "YOLO_IMG_SIZE", None)
    imgsz = int(img_size_cfg) if img_size_cfg is not None else None
    device = _select_device(getattr(params, "YOLO_DEVICE", "auto"))
    # On Apple Silicon (MPS), torchvision NMS is not implemented. Enable CPU fallback.
    if isinstance(device, str) and device.lower() == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for unsupported MPS ops (e.g., NMS)")
    conf_thres = float(getattr(params, "YOLO_CONF_THRES", 0.1))
    iou_thres = float(getattr(params, "YOLO_IOU_THRES", 0.5))
    max_frames = getattr(params, "MAX_FRAMES", None)
    max_frames = int(max_frames) if max_frames is not None else None

    # Estimate total frames for progress reporting
    cap = cv2.VideoCapture(str(video_path))
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if max_frames is not None and max_frames > 0 and total_est > 0:
        total_target = min(total_est, max_frames)
    elif max_frames is not None and max_frames > 0:
        total_target = max_frames
    elif total_est > 0:
        total_target = total_est
    else:
        total_target = None

    rows: list[dict] = []
    total_boxes = 0
    frames_seen = 0

    predict_kwargs = dict(
        source=str(video_path),
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        verbose=False,
        stream=True,
    )
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    t0 = time.perf_counter()
    for frame_idx, r in enumerate(model.predict(**predict_kwargs)):
        if max_frames is not None and frame_idx >= max_frames:
            break
        if total_target:
            processed = frame_idx + 1
            elapsed = time.perf_counter() - t0
            if elapsed > 0 and processed > 0:
                fps = processed / elapsed
                remaining = max(0, total_target - processed)
                eta_sec = remaining / fps if fps > 0 else 0.0
                eta_str = _format_eta(eta_sec)
                tag = f"Stage1 YOLO ETA {eta_str}"
            else:
                tag = "Stage1 YOLO"
            _progress(processed, total_target, tag)
        frames_seen += 1
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        boxes = r.boxes
        if boxes.xyxy is None or boxes.conf is None:
            continue
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            conf = float(confs[i])
            rows.append(
                {
                    "x": int(round(x1)),
                    "y": int(round(y1)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                    "frame": int(frame_idx),
                    "video_name": video_path.name,
                    "firefly_logit": conf,
                    "background_logit": float(1.0 - conf),
                }
            )
            total_boxes += 1

    # Write CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x",
                "y",
                "w",
                "h",
                "frame",
                "video_name",
                "firefly_logit",
                "background_logit",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    print(
        f"Stage1  Video: {video_path.name}; frames_seen={frames_seen}; "
        f"detections={total_boxes}; conf>={conf_thres}"
    )
    if rows:
        frames_with = len({r["frame"] for r in rows})
        avg_boxes = total_boxes / max(1, frames_seen)
        print(
            f"Stage1  Frames with detections: {frames_with}; "
            f"avg_boxes/frame={avg_boxes:.2f}"
        )

    print(f"Stage1  Wrote detections CSV → {csv_path}")
    return csv_path


__all__ = ["run_for_video"]

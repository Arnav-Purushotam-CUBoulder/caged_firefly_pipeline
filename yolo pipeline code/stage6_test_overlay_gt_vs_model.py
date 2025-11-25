#!/usr/bin/env python3
"""
Stage 6 (test) — Overlay GT vs YOLO model predictions.

Uses:
  - Predictions: point CSV from Stage 3 boxes (x,y,t,firefly_logit,background_logit)
  - Normalized GT: gt_norm.csv produced by Stage 5 in the per-video folder

Writes a single overlay video (GT=green, Model=red, overlap=yellow).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import cv2
import numpy as np

import params


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _progress(i: int, total: int, tag: str = "") -> None:
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar = "=" * int(frac * 50) + " " * (50 - int(frac * 50))
    print(f"\r{tag} [{bar}] {int(frac*100):3d}%", end="" if i < total else "\n")


def _clamp_box(x0: int, y0: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    w = max(1, min(int(w), W))
    h = max(1, min(int(h), H))
    x0 = max(0, min(int(x0), W - w))
    y0 = max(0, min(int(y0), H - h))
    return x0, y0, w, h


def _read_preds_by_frame(csv_path: Path, max_frames: Optional[int] = None) -> Dict[int, List[Tuple[float, float]]]:
    by_frame: Dict[int, List[Tuple[float, float]]] = {}
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            t = int(float(r.get("t", r.get("frame", "0"))))
            if max_frames is not None and t >= max_frames:
                continue
            x = float(r["x"])
            y = float(r["y"])
        except Exception:
            continue
        by_frame.setdefault(t, []).append((x, y))
    return by_frame


def _read_gt_by_frame(gt_norm_csv: Path, max_frames: Optional[int] = None) -> Dict[int, List[Tuple[int, int]]]:
    by_frame: Dict[int, List[Tuple[int, int]]] = {}
    with gt_norm_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            t = int(r["t"])
            if max_frames is not None and t >= max_frames:
                continue
            x = int(round(float(r["x"])))
            y = int(round(float(r["y"])))
        except Exception:
            continue
        by_frame.setdefault(t, []).append((x, y))
    return by_frame


def stage6_test_overlay_gt_vs_model(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    post9_dir: Path,
    out_video_path: Path,
    gt_box_w: int,
    gt_box_h: int,
    thickness: int,
    max_frames: Optional[int] = None,
    render_threshold_overlays: bool = True,  # unused but kept for API parity
) -> None:
    """Render overlay video showing GT vs model predictions."""
    post9_dir = Path(post9_dir)
    gt_norm_csv = post9_dir / "gt_norm.csv"
    if not gt_norm_csv.exists():
        raise FileNotFoundError(f"[stage6_test] Missing normalized GT CSV: {gt_norm_csv}")

    preds_by_frame = _read_preds_by_frame(pred_csv_path, max_frames=max_frames)
    gt_by_frame = _read_gt_by_frame(gt_norm_csv, max_frames=max_frames)

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage6_test] Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _ensure_dir(out_video_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (W, H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"[stage6_test] Could not open VideoWriter: {out_video_path}")

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)

    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break

        red_layer = np.zeros_like(frame)
        green_layer = np.zeros_like(frame)

        # Model predictions (centers) in RED with fixed test box size
        for (x, y) in preds_by_frame.get(fr, []):
            x0 = int(round(x - gt_box_w / 2.0))
            y0 = int(round(y - gt_box_h / 2.0))
            x0, y0, w, h = _clamp_box(x0, y0, gt_box_w, gt_box_h, W, H)
            cv2.rectangle(red_layer, (x0, y0), (x0 + w, y0 + h), RED, thickness)

        # GT in GREEN
        for (gx, gy) in gt_by_frame.get(fr, []):
            x0 = int(round(gx - gt_box_w / 2.0))
            y0 = int(round(gy - gt_box_h / 2.0))
            x0, y0, gw, gh = _clamp_box(x0, y0, gt_box_w, gt_box_h, W, H)
            cv2.rectangle(green_layer, (x0, y0), (x0 + gw, y0 + gh), GREEN, thickness)

        red_mask = np.any(red_layer > 0, axis=2)
        green_mask = np.any(green_layer > 0, axis=2)
        overlap_mask = red_mask & green_mask
        only_red = red_mask & ~overlap_mask
        only_green = green_mask & ~overlap_mask

        frame[only_red] = RED
        frame[only_green] = GREEN
        frame[overlap_mask] = YELLOW

        out.write(frame)
        _progress(fr + 1, limit, "stage6-overlay")
        fr += 1

    cap.release()
    out.release()
    print(f"[stage6_test] Wrote overlay: {out_video_path}")


__all__ = ["stage6_test_overlay_gt_vs_model"]


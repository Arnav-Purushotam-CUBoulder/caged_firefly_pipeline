#!/usr/bin/env python3
"""
Stage 3: Gaussian intensity centroid refinement.

Reads Stage 2 filtered CSVs:
  STAGE2_DIR/<video_stem>/<video_stem>_bright.csv

For each detection (x,y,w,h,frame), refines the center inside the patch
using an intensity centroid (optionally Gaussian-weighted), recenters the
box around that center (keeping w,h), and writes per-video CSVs under:

  STAGE3_DIR/<video_stem>/<video_stem>_gauss.csv

with the same schema as Stage 2.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import csv

import cv2
import numpy as np

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


def _group_by_frame(rows: List[dict]) -> Dict[int, List[dict]]:
    by_t: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        try:
            t = int(row["frame"])
        except Exception:
            continue
        by_t[t].append(row)
    return by_t


def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        k = np.ones((h, w), dtype=np.float32)
        return k / float(h * w)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    s = float(g.sum())
    if s > 0:
        g /= s
    return g


def _intensity_centroid(img_gray: np.ndarray, gaussian_sigma: float = 0.0) -> tuple[float, float]:
    img = img_gray.astype(np.float32)
    if gaussian_sigma and gaussian_sigma > 0:
        gh, gw = img.shape[:2]
        G = _gaussian_kernel(gw, gh, gaussian_sigma)
        img = img * G
    total = float(img.sum())
    if total <= 1e-6:
        H, W = img.shape[:2]
        return (W / 2.0, H / 2.0)
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float32)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy


def run_for_video(video_path: Path) -> Path:
    """Refine boxes using Gaussian intensity centroid; returns Stage 3 CSV path."""
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    # Prefer Stage2.1 trajectory-intensity filtered output when enabled.
    s2_csv_default = (params.STAGE2_DIR / stem) / f"{stem}_bright.csv"
    s2_csv_traj = (params.STAGE2_DIR / stem) / f"{stem}_bright_traj.csv"
    if bool(getattr(params, "STAGE2_1_ENABLE", False)) and s2_csv_traj.exists():
        s2_csv = s2_csv_traj
    else:
        s2_csv = s2_csv_default
    if not s2_csv.exists():
        raise FileNotFoundError(f"Missing Stage2 input CSV for {stem}: {s2_csv}")

    rows: List[dict] = []
    with s2_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print(f"Stage3  NOTE: no Stage2 detections for {video_path.name}")
        out_root = params.STAGE3_DIR / stem
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / f"{stem}_gauss.csv"
        with out_csv.open("w", newline="") as f:
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
                    "gaussian_x_centroid",
                    "gaussian_y_centroid",
                ],
            )
            writer.writeheader()
        return out_csv

    by_t = _group_by_frame(rows)

    cap, W, H, fps, total = _open_video(video_path)
    max_frames = getattr(params, "MAX_FRAMES", None)
    if max_frames is not None:
        limit = min(total, int(max_frames))
    else:
        limit = total

    sigma = float(getattr(params, "GAUSS_SIGMA", 0.0))

    out_root = params.STAGE3_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_gauss.csv"

    out_rows: List[dict] = []
    refined = 0
    shifts: List[float] = []

    try:
        for t in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            dets = by_t.get(t, [])
            if not dets:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Hf, Wf = gray.shape[:2]
            for row in dets:
                try:
                    x = int(float(row["x"]))
                    y = int(float(row["y"]))
                    w = int(float(row["w"]))
                    h = int(float(row["h"]))
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                # Clip to frame bounds
                x0 = max(0, min(x, Wf - 1))
                y0 = max(0, min(y, Hf - 1))
                x1 = max(0, min(x0 + w, Wf))
                y1 = max(0, min(y0 + h, Hf))
                if x1 <= x0 or y1 <= y0:
                    continue
                patch = gray[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                cx_local, cy_local = _intensity_centroid(patch, sigma)
                new_cx = x0 + cx_local
                new_cy = y0 + cy_local
                # Recenter box around (new_cx, new_cy) while keeping size w,h
                new_x0 = int(round(new_cx - w / 2.0))
                new_y0 = int(round(new_cy - h / 2.0))
                new_x0 = max(0, min(new_x0, Wf - w))
                new_y0 = max(0, min(new_y0, Hf - h))

                new_row = dict(row)
                new_row["x"] = int(new_x0)
                new_row["y"] = int(new_y0)
                # Store global Gaussian centroid coordinates for this box
                new_row["gaussian_x_centroid"] = float(new_cx)
                new_row["gaussian_y_centroid"] = float(new_cy)
                out_rows.append(new_row)
                refined += 1

                # Track shift magnitude (using center positions)
                old_cx = x0 + w / 2.0
                old_cy = y0 + h / 2.0
                dx = float(new_cx - old_cx)
                dy = float(new_cy - old_cy)
                shifts.append((dx * dx + dy * dy) ** 0.5)
    finally:
        cap.release()

    with out_csv.open("w", newline="") as f:
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
                "gaussian_x_centroid",
                "gaussian_y_centroid",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    print(
        f"Stage3  Video: {video_path.name}; refined={refined}; "
        f"mean_center_shift_px={mean_shift:.2f}; sigma={sigma}"
    )
    return out_csv


__all__ = ["run_for_video"]

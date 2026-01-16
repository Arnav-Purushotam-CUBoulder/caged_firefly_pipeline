#!/usr/bin/env python3
"""
Stage 8 (test) — Analyze false positives (FPs) for the YOLO pipeline.

For each thr_* directory under the per-video Stage 5 output folder, this stage:
  - Computes nearest GT (TP or FN) for each FP and writes:
        thr_*/fp_nearest_tp.csv
  - Saves full-frame images with FP and nearest GT marked:
        thr_*/fp_pair_frames/
  - Saves full-frame overlays with GT (TP+FN) in green and FP in red:
        thr_*/fp_vs_gt_frames/
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math

import cv2
import numpy as np

import params


def _progress(i: int, total: int, tag: str = "") -> None:
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar = "=" * int(frac * 50) + " " * (50 - int(frac * 50))
    print(f"\r{tag} [{bar}] {int(frac*100):3d}%", end="" if i < total else "\n")


def _read_points_by_frame(csv_path: Path) -> Dict[int, List[Tuple[float, float]]]:
    by_t: Dict[int, List[Tuple[float, float]]] = {}
    if not csv_path.exists():
        return by_t
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = int(float(row.get("t", row.get("frame", "0"))))
                x = float(row["x"])
                y = float(row["y"])
            except Exception:
                continue
            by_t.setdefault(t, []).append((x, y))
    return by_t


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _draw_centered_box(
    img: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    H, W = img.shape[:2]
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = x0 + int(w) - 1
    y1 = y0 + int(h) - 1
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def _crop_brightness(
    frame: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> Tuple[int, int]:
    """Return (max_val, nbright) for a center crop using Stage2 area threshold."""
    H, W = frame.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    crop = frame[y0 : y0 + h, x0 : x0 + w].copy()
    if crop.size == 0:
        return 0, 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return 0, 0
    max_val = int(gray.max())
    area_thr = int(getattr(params, "STAGE2_AREA_INTENSITY_THR", 190))
    nbright = int((gray >= area_thr).sum())
    return max_val, nbright


def _analyze_threshold_dir(
    thr_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
) -> Tuple[int, int]:
    """Compute nearest GT (TP or FN) for each FP and render diagnostic frames."""
    fps_csv = thr_dir / "fps.csv"
    tps_csv = thr_dir / "tps.csv"
    fns_csv = thr_dir / "fns.csv"
    out_csv = thr_dir / "fp_nearest_tp.csv"

    out_pair_dir = thr_dir / "fp_pair_frames"
    out_pair_dir.mkdir(parents=True, exist_ok=True)
    out_vs_gt_dir = thr_dir / "fp_vs_gt_frames"
    out_vs_gt_dir.mkdir(parents=True, exist_ok=True)

    fps_by_t = _read_points_by_frame(fps_csv)
    tps_by_t = _read_points_by_frame(tps_csv)
    fns_by_t = _read_points_by_frame(fns_csv)

    frames = sorted(set(fps_by_t.keys()))
    total_fps = sum(len(v) for v in fps_by_t.values())
    no_gt_cnt = 0

    rows_out: List[Dict[str, object]] = []
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8_test] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for t in frames:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(t, total_frames - 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        fps = fps_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        fns = fns_by_t.get(t, [])
        gts = (tps or []) + (fns or [])

        for (fx, fy) in fps:
            fpx = int(round(fx))
            fpy = int(round(fy))
            fp_max, fp_bright = _crop_brightness(frame, fx, fy, box_w, box_h)
            if not gts:
                rows_out.append(
                    {
                        "t": t,
                        "fp_x": fx,
                        "fp_y": fy,
                        "gt_x": None,
                        "gt_y": None,
                        "dist": None,
                    }
                )
                # Render FP vs GT overlay even if no GT points (only FP highlighted)
                red_layer = np.zeros_like(frame)
                green_layer = np.zeros_like(frame)
                _draw_centered_box(
                    red_layer, fx, fy, box_w, box_h, (0, 0, 255), thickness
                )
                composed = frame.copy()
                red_mask = np.any(red_layer > 0, axis=2)
                composed[red_mask] = (0, 0, 255)
                img_name = (
                    f"t{t:06d}_FP({fpx},{fpy})_nearestGT(NA,NA)_"
                    f"dNA_FPmax{fp_max}_FPbrightpx{fp_bright}.png"
                )
                cv2.imwrite(str(out_vs_gt_dir / img_name), composed)
                no_gt_cnt += 1
                continue

            # nearest GT (TP or FN)
            min_d = None
            min_gt = None
            for (gx, gy) in gts:
                d = _euclid((fx, fy), (gx, gy))
                if min_d is None or d < min_d:
                    min_d = d
                    min_gt = (gx, gy)
            ngx = int(round(min_gt[0]))
            ngy = int(round(min_gt[1]))
            dstr = f"{float(min_d):.6f}"
            gt_max, gt_bright = _crop_brightness(frame, min_gt[0], min_gt[1], box_w, box_h)

            # pair frame: FP=RED, GT=GREEN
            canvas = frame.copy()
            _draw_centered_box(canvas, fx, fy, box_w, box_h, (0, 0, 255), thickness)
            _draw_centered_box(canvas, min_gt[0], min_gt[1], box_w, box_h, (0, 255, 0), thickness)
            pair_path = (
                out_pair_dir
                / (
                    f"t{t:06d}_FP({fpx},{fpy})_nearestGT({ngx},{ngy})_d{dstr}_"
                    f"FPmax{fp_max}_FPbrightpx{fp_bright}_"
                    f"GTmax{gt_max}_GTbrightpx{gt_bright}.png"
                )
            )
            cv2.imwrite(str(pair_path), canvas)

            # overlay: GT in GREEN (TP+FN), FP in RED
            red_layer = np.zeros_like(frame)
            green_layer = np.zeros_like(frame)
            _draw_centered_box(
                red_layer, fx, fy, box_w, box_h, (0, 0, 255), thickness
            )
            for (gx, gy) in gts:
                _draw_centered_box(
                    green_layer, gx, gy, box_w, box_h, (0, 255, 0), thickness
                )
            composed = frame.copy()
            red_mask = np.any(red_layer > 0, axis=2)
            green_mask = np.any(green_layer > 0, axis=2)
            overlap = red_mask & green_mask
            only_red = red_mask & ~overlap
            only_green = green_mask & ~overlap
            composed[only_red] = (0, 0, 255)
            composed[only_green] = (0, 255, 0)
            composed[overlap] = (0, 255, 255)
            img_name = (
                f"t{t:06d}_FP({fpx},{fpy})_nearestGT({ngx},{ngy})_"
                f"d{dstr}_FPmax{fp_max}_FPbrightpx{fp_bright}_"
                f"GTmax{gt_max}_GTbrightpx{gt_bright}.png"
            )
            cv2.imwrite(str(out_vs_gt_dir / img_name), composed)

            rows_out.append(
                {
                    "t": t,
                    "fp_x": fx,
                    "fp_y": fy,
                    "gt_x": float(min_gt[0]),
                    "gt_y": float(min_gt[1]),
                    "dist": float(min_d),
                }
            )

    cap.release()

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t", "fp_x", "fp_y", "gt_x", "gt_y", "dist"]
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fps, no_gt_cnt


def stage8_test_fp_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    verbose: bool = True,
) -> None:
    """Run FP nearest-GT analysis and render diagnostics for all thresholds for one video."""
    stage9_video_dir = Path(stage9_video_dir)
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage8_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return

    thr_dirs = sorted(
        [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")]
    )
    if not thr_dirs:
        if verbose:
            print(f"[stage8_test] No thr_* directories found in: {stage9_video_dir}")
        return

    for i, thr_dir in enumerate(thr_dirs, start=1):
        total_fps, no_gt = _analyze_threshold_dir(
            thr_dir, orig_video_path, box_w, box_h, thickness
        )
        if verbose:
            print(
                f"[stage8_test] {stage9_video_dir.name} / {thr_dir.name}: "
                f"FPs={total_fps}, FPs_with_no_GT={no_gt}"
            )
        _progress(i, len(thr_dirs), "stage8-test-fp")


__all__ = ["stage8_test_fp_nearest_tp_analysis"]

#!/usr/bin/env python3
"""
Stage 7 (test) — Analyze FNs: nearest TP distance and diagnostic frames per threshold.

For each thr_* directory under the per-video Stage 5 output folder, this stage:
  - Computes nearest TPs for each FN and writes:
        thr_*/fn_nearest_tp.csv
  - Saves full-frame images with FN and nearest TP marked:
        thr_*/fn_pair_frames/
  - Saves full-frame overlays with predictions (TP+FP) in red and FN in green:
        thr_*/fn_vs_pred_frames/
  - Saves raw (unannotated) full frames for each FN:
        thr_*/fn_raw_frames/
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import math

import cv2
import numpy as np


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


def _analyze_threshold_dir_and_render(
    *,
    thr_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    verbose: bool = True,
) -> Tuple[int, int]:
    """Compute nearest TP for each FN and render diagnostic frames."""
    fns_csv = thr_dir / "fns.csv"
    tps_csv = thr_dir / "tps.csv"
    fps_csv = thr_dir / "fps.csv"
    out_csv = thr_dir / "fn_nearest_tp.csv"

    out_pair_dir = thr_dir / "fn_pair_frames"
    out_pair_dir.mkdir(parents=True, exist_ok=True)
    out_vs_pred_dir = thr_dir / "fn_vs_pred_frames"
    out_vs_pred_dir.mkdir(parents=True, exist_ok=True)
    out_raw_dir = thr_dir / "fn_raw_frames"
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    fns_by_t = _read_points_by_frame(fns_csv)
    tps_by_t = _read_points_by_frame(tps_csv)
    fps_by_t = _read_points_by_frame(fps_csv) if fps_csv.exists() else {}

    preds_by_t: Dict[int, List[Tuple[float, float]]] = {}
    all_ts = set(tps_by_t.keys()) | set(fps_by_t.keys())
    for t in all_ts:
        preds_by_t[t] = (tps_by_t.get(t, []) or []) + (fps_by_t.get(t, []) or [])

    frames = sorted(set(fns_by_t.keys()))
    total_fns = sum(len(v) for v in fns_by_t.values())
    no_tp_cnt = 0

    rows_out: List[Dict[str, object]] = []
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage7_test] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for t in frames:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(t, total_frames - 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        fns = fns_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        preds = preds_by_t.get(t, [])
        for (fx, fy) in fns:
            fnx = int(round(fx))
            fny = int(round(fy))

            # Save raw (unannotated) frame for this FN
            raw_name = f"FN_raw_t{t:06d}_x{fnx}_y{fny}.png"
            cv2.imwrite(str(out_raw_dir / raw_name), frame)

            if not tps:
                rows_out.append(
                    {
                        "t": t,
                        "fn_x": fx,
                        "fn_y": fy,
                        "tp_x": None,
                        "tp_y": None,
                        "dist": None,
                        "image_path": "",
                    }
                )
                # Even if no TP, still render FN vs PRED overlay if any predictions exist
                if preds:
                    red_layer = np.zeros_like(frame)
                    green_layer = np.zeros_like(frame)
                    for (px, py) in preds:
                        _draw_centered_box(
                            red_layer, px, py, box_w, box_h, (0, 0, 255), thickness
                        )
                    _draw_centered_box(
                        green_layer, fx, fy, box_w, box_h, (0, 255, 0), thickness
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
                    img_name_pred = (
                        f"t{t:06d}_FN({fnx},{fny})_nearestPRED(NA,NA)_dNA.png"
                    )
                    cv2.imwrite(str(out_vs_pred_dir / img_name_pred), composed)
                continue

            # nearest TP
            min_d = None
            min_tp = None
            for (tx, ty) in tps:
                d = _euclid((fx, fy), (tx, ty))
                if min_d is None or d < min_d:
                    min_d = d
                    min_tp = (tx, ty)

            canvas = frame.copy()
            _draw_centered_box(canvas, fx, fy, box_w, box_h, (0, 255, 0), thickness)
            _draw_centered_box(
                canvas, min_tp[0], min_tp[1], box_w, box_h, (0, 0, 255), thickness
            )
            ntx = int(round(min_tp[0]))
            nty = int(round(min_tp[1]))
            dstr = f"{float(min_d):.6f}"
            out_path = (
                out_pair_dir
                / f"t{t:06d}_FN({fnx},{fny})_nearestTP({ntx},{nty})_d{dstr}.png"
            )
            cv2.imwrite(str(out_path), canvas)

            rows_out.append(
                {
                    "t": t,
                    "fn_x": fx,
                    "fn_y": fy,
                    "tp_x": float(min_tp[0]),
                    "tp_y": float(min_tp[1]),
                    "dist": float(min_d),
                    "image_path": str(out_path),
                }
            )

            # overlay: preds (TP+FP) in RED and FN in GREEN
            red_layer = np.zeros_like(frame)
            green_layer = np.zeros_like(frame)
            for (px, py) in preds:
                _draw_centered_box(
                    red_layer, px, py, box_w, box_h, (0, 0, 255), thickness
                )
            _draw_centered_box(
                green_layer, fx, fy, box_w, box_h, (0, 255, 0), thickness
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

            if preds:
                min_d_pred = None
                min_pred = None
                for (px, py) in preds:
                    d = _euclid((fx, fy), (px, py))
                    if min_d_pred is None or d < min_d_pred:
                        min_d_pred = d
                        min_pred = (px, py)
                npx = int(round(min_pred[0]))
                npy = int(round(min_pred[1]))
                dstr_pred = f"{min_d_pred:.6f}"
                img_name_pred = (
                    f"t{t:06d}_FN({fnx},{fny})_nearestPRED({npx},{npy})_d{dstr_pred}.png"
                )
            else:
                img_name_pred = (
                    f"t{t:06d}_FN({fnx},{fny})_nearestPRED(NA,NA)_dNA.png"
                )
            cv2.imwrite(str(out_vs_pred_dir / img_name_pred), composed)

        _progress(frames.index(t) + 1, len(frames), "stage7-test-fn")

    cap.release()

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t", "fn_x", "fn_y", "tp_x", "tp_y", "dist", "image_path"]
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fns, no_tp_cnt


def stage7_test_fn_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    verbose: bool = True,
) -> None:
    """Run FN nearest-TP analysis and render diagnostics for all thresholds for one video."""
    stage9_video_dir = Path(stage9_video_dir)
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage7_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return

    thr_dirs = sorted(
        [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")]
    )
    if not thr_dirs:
        if verbose:
            print(f"[stage7_test] No thr_* directories found in: {stage9_video_dir}")
        return

    total_fns_all = 0
    total_no_tp = 0
    for i, thr_dir in enumerate(thr_dirs, start=1):
        total_fns, no_tp = _analyze_threshold_dir_and_render(
            thr_dir=thr_dir,
            orig_video_path=orig_video_path,
            box_w=box_w,
            box_h=box_h,
            thickness=thickness,
            verbose=verbose,
        )
        total_fns_all += total_fns
        total_no_tp += no_tp
        _progress(i, len(thr_dirs), "stage7-test-fn")

    if verbose:
        print(
            f"[stage7_test] Done. Total FNs analyzed: {total_fns_all}. "
            f"FNs without any TP: {total_no_tp}."
        )


__all__ = ["stage7_test_fn_nearest_tp_analysis"]


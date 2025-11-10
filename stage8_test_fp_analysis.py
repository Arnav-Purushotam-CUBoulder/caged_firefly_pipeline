#!/usr/bin/env python3
"""
Stage 8 (test) — Analyze FPs: nearest TP distance and diagnostic frames per threshold.

Inputs (per video): thr_*/(tps.csv, fps.csv, fns.csv)
Outputs:
  - thr_*/fp_nearest_tp.csv
  - thr_*/fp_pair_frames/ full-frame images with FP and nearest TP marked
  - thr_*/fp_vs_gt_frames/ overlays with GT (TP+FN) in green and FP in red
"""
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math

import cv2
import numpy as np


def _progress(i: int, total: int, tag: str = ''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * 50) + ' ' * (50 - int(frac * 50))
    print(f'\r{tag} [{bar}] {int(frac*100):3d}%', end='' if i < total else '\n')

def _read_points_by_frame(csv_path: Path) -> Dict[int, List[Tuple[float, float]]]:
    by_t: Dict[int, List[Tuple[float, float]]] = {}
    if not csv_path.exists():
        return by_t
    with csv_path.open('r', newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = int(float(row.get('t', row.get('frame', '0'))))
                x = float(row['x']); y = float(row['y'])
            except Exception:
                continue
            by_t.setdefault(t, []).append((x, y))
    return by_t

def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def _draw_centered_box(img: np.ndarray, cx: float, cy: float, w: int, h: int,
                       color: Tuple[int,int,int], thickness: int = 1):
    H, W = img.shape[:2]
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x1 = x0 + int(w) - 1;        y1 = y0 + int(h) - 1
    x0 = max(0, min(x0, W-1)); y0 = max(0, min(y0, H-1))
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)


def _analyze_threshold_dir_and_render(
    *,
    thr_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    color: Tuple[int,int,int],
    thickness: int,
    verbose: bool = True
) -> Tuple[int, int]:
    fps_csv = thr_dir / 'fps.csv'
    tps_csv = thr_dir / 'tps.csv'
    fns_csv = thr_dir / 'fns.csv'
    out_csv = thr_dir / 'fp_nearest_tp.csv'
    out_img_dir = thr_dir / 'fp_pair_frames'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir_fp_vs_gt = thr_dir / 'fp_vs_gt_frames'
    out_img_dir_fp_vs_gt.mkdir(parents=True, exist_ok=True)

    fps_by_t = _read_points_by_frame(fps_csv)
    tps_by_t = _read_points_by_frame(tps_csv)
    fns_by_t = _read_points_by_frame(fns_csv)

    frames = sorted(set(fps_by_t.keys()))
    total_fps = sum(len(v) for v in fps_by_t.values())
    no_tp_cnt = 0

    rows_out: List[Dict[str, object]] = []
    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8_test] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    for t in frames:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(t, total_frames-1)))
        ok, frame = cap.read()
        if not ok:
            continue
        fps = fps_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        gts = (tps_by_t.get(t, []) or []) + (fns_by_t.get(t, []) or [])
        for (fx, fy) in fps:
            if not tps:
                rows_out.append({'t': t, 'fp_x': fx, 'fp_y': fy, 'tp_x': None, 'tp_y': None, 'dist': None, 'image_path': ''})
                no_tp_cnt += 1
                continue
            # nearest TP
            min_d = None; min_tp = None
            for (tx, ty) in tps:
                d = _euclid((fx, fy), (tx, ty))
                if (min_d is None) or (d < min_d):
                    min_d = d; min_tp = (tx, ty)
            # render FP + nearest TP
            canvas = frame.copy()
            _draw_centered_box(canvas, fx, fy, box_w, box_h, color, thickness)
            _draw_centered_box(canvas, min_tp[0], min_tp[1], box_w, box_h, color, thickness)
            out_path = out_img_dir / f"t{t:06d}_fp_{int(round(fx))}_{int(round(fy))}_nearesttp_{int(round(min_tp[0]))}_{int(round(min_tp[1]))}.png"
            cv2.imwrite(str(out_path), canvas)
            rows_out.append({'t': t, 'fp_x': fx, 'fp_y': fy, 'tp_x': float(min_tp[0]), 'tp_y': float(min_tp[1]), 'dist': float(min_d), 'image_path': str(out_path)})
            # overlay: GT (TP+FN) in GREEN, FP in RED
            red = frame.copy(); green = frame.copy()
            _draw_centered_box(red, fx, fy, box_w, box_h, (0,0,255), thickness)
            for (gx, gy) in gts:
                _draw_centered_box(green, gx, gy, box_w, box_h, (0,255,0), thickness)
            red_mask = np.any(red > 0, axis=2); green_mask = np.any(green > 0, axis=2)
            overlap = red_mask & green_mask
            fused = frame.copy()
            fused[red_mask & ~overlap] = (0,0,255)
            fused[green_mask & ~overlap] = (0,255,0)
            fused[overlap] = (0,255,255)
            out_path2 = out_img_dir_fp_vs_gt / f"t{t:06d}_fp_vs_gt.png"
            cv2.imwrite(str(out_path2), fused)
        _progress(frames.index(t)+1, len(frames), 'stage8-test-fp')

    cap.release()
    # write CSV
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t','fp_x','fp_y','tp_x','tp_y','dist','image_path'])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fps, no_tp_cnt


def stage8_test_fp_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int = 10,
    box_h: int = 10,
    thickness: int = 1,
    verbose: bool = True,
) -> None:
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage8_test] Stage-5 test dir not found: {stage9_video_dir}")
        return
    thr_dirs = sorted([p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith('thr_')])
    if not thr_dirs:
        if verbose:
            print(f"[stage8_test] No thr_* folders in: {stage9_video_dir}")
        return
    total_fps_all = 0; total_no_tp = 0
    for td in thr_dirs:
        fps_csv = td / 'fps.csv'; tps_csv = td / 'tps.csv'
        if not (fps_csv.exists() and tps_csv.exists()):
            if verbose:
                print(f"[stage8_test] Skipping {td.name} (missing fps.csv or tps.csv)")
            continue
        total_fps, no_tp = _analyze_threshold_dir_and_render(
            thr_dir=td,
            orig_video_path=orig_video_path,
            box_w=box_w,
            box_h=box_h,
            color=(0,255,255),  # yellow
            thickness=thickness,
            verbose=verbose,
        )
        total_fps_all += total_fps; total_no_tp += no_tp
    if verbose:
        print(f"[stage8_test] Done. Total FPs analyzed: {total_fps_all}. FPs without any TP: {total_no_tp}.")


__all__ = ['stage8_test_fp_nearest_tp_analysis']

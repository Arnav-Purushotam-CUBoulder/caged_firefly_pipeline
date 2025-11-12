#!/usr/bin/env python3
"""
Stage 7 (test) — Analyze FNs: nearest TP distance and diagnostic frames per threshold.

Inputs (per video):
  - Stage9 folder: thr_*/(tps.csv, fps.csv, fns.csv)
Outputs:
  - thr_*/fn_nearest_tp.csv
  - thr_*/fn_pair_frames/ full-frame images with FN and nearest TP marked
  - thr_*/fn_vs_pred_frames/ overlays with predictions (TP+FP) in red and FN in green
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
    fns_csv = thr_dir / 'fns.csv'
    tps_csv = thr_dir / 'tps.csv'
    fps_csv = thr_dir / 'fps.csv'
    out_csv = thr_dir / 'fn_nearest_tp.csv'
    out_img_dir = thr_dir / 'fn_pair_frames'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir_fn_vs_pred = thr_dir / 'fn_vs_pred_frames'
    out_img_dir_fn_vs_pred.mkdir(parents=True, exist_ok=True)

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for t in frames:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(t, total_frames-1)))
        ok, frame = cap.read()
        if not ok:
            continue
        fns = fns_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        preds = preds_by_t.get(t, [])
        for (fx, fy) in fns:
            if not tps:
                rows_out.append({'t': t, 'fn_x': fx, 'fn_y': fy, 'tp_x': None, 'tp_y': None, 'dist': None, 'image_path': ''})
                no_tp_cnt += 1
                # Even if no TP, still render FN vs PRED overlay if any predictions exist
                # Create overlay layers (do not draw directly on frame)
                red_layer   = np.zeros_like(frame)
                green_layer = np.zeros_like(frame)
                # Draw all predictions (TP + FP) in RED
                for (px, py) in preds:
                    _draw_centered_box(red_layer, px, py, box_w, box_h, (0,0,255), thickness)
                # Draw this FN in GREEN
                _draw_centered_box(green_layer, fx, fy, box_w, box_h, (0,255,0), thickness)
                # Compose outlines → red/green/yellow
                composed = frame.copy()
                red_mask   = np.any(red_layer > 0, axis=2)
                green_mask = np.any(green_layer > 0, axis=2)
                overlap    = red_mask & green_mask
                only_red   = red_mask & ~overlap
                only_green = green_mask & ~overlap
                composed[only_red]   = (0,0,255)
                composed[only_green] = (0,255,0)
                composed[overlap]    = (0,255,255)
                fnx = int(round(fx)); fny = int(round(fy))
                # Determine nearest prediction if available
                if preds:
                    min_d_pred = None; min_pred = None
                    for (px, py) in preds:
                        d = _euclid((fx, fy), (px, py))
                        if (min_d_pred is None) or (d < min_d_pred):
                            min_d_pred = d; min_pred = (px, py)
                    npx, npy = int(round(min_pred[0])), int(round(min_pred[1]))
                    dstr_pred = f"{min_d_pred:.6f}"
                    legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                    img_name_pred = (
                        f"t{t:06d}_{legend}_FN({fnx},{fny})_nearestPRED({npx},{npy})_d{dstr_pred}.png"
                    )
                else:
                    legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                    img_name_pred = (
                        f"t{t:06d}_{legend}_FN({fnx},{fny})_nearestPRED(NA,NA)_dNA.png"
                    )
                out_path2 = out_img_dir_fn_vs_pred / img_name_pred
                cv2.imwrite(str(out_path2), composed)
                continue
            # nearest TP
            min_d = None; min_tp = None
            for (tx, ty) in tps:
                d = _euclid((fx, fy), (tx, ty))
                if (min_d is None) or (d < min_d):
                    min_d = d; min_tp = (tx, ty)
            # render FN + nearest TP
            canvas = frame.copy()
            _draw_centered_box(canvas, fx, fy, box_w, box_h, color, thickness)
            _draw_centered_box(canvas, min_tp[0], min_tp[1], box_w, box_h, color, thickness)
            out_path = out_img_dir / f"t{t:06d}_fn_{int(round(fx))}_{int(round(fy))}_nearesttp_{int(round(min_tp[0]))}_{int(round(min_tp[1]))}.png"
            cv2.imwrite(str(out_path), canvas)
            rows_out.append({'t': t, 'fn_x': fx, 'fn_y': fy, 'tp_x': float(min_tp[0]), 'tp_y': float(min_tp[1]), 'dist': float(min_d), 'image_path': str(out_path)})
            # overlay: preds (TP+FP) in RED and FN in GREEN with nearest-PRED info and unique filenames
            red_layer   = np.zeros_like(frame)
            green_layer = np.zeros_like(frame)
            # Draw all predictions in RED
            for (px, py) in preds:
                _draw_centered_box(red_layer, px, py, box_w, box_h, (0,0,255), thickness)
            # Draw this FN in GREEN
            _draw_centered_box(green_layer, fx, fy, box_w, box_h, (0,255,0), thickness)
            # Compose outlines → red/green/yellow
            composed = frame.copy()
            red_mask   = np.any(red_layer > 0, axis=2)
            green_mask = np.any(green_layer > 0, axis=2)
            overlap    = red_mask & green_mask
            only_red   = red_mask & ~overlap
            only_green = green_mask & ~overlap
            composed[only_red]   = (0,0,255)
            composed[only_green] = (0,255,0)
            composed[overlap]    = (0,255,255)

            fnx = int(round(fx)); fny = int(round(fy))
            if preds:
                min_d_pred = None; min_pred = None
                for (px, py) in preds:
                    d = _euclid((fx, fy), (px, py))
                    if (min_d_pred is None) or (d < min_d_pred):
                        min_d_pred = d; min_pred = (px, py)
                npx, npy = int(round(min_pred[0])), int(round(min_pred[1]))
                dstr_pred = f"{min_d_pred:.6f}"
                legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                img_name_pred = (
                    f"t{t:06d}_{legend}_FN({fnx},{fny})_nearestPRED({npx},{npy})_d{dstr_pred}.png"
                )
            else:
                legend = "LEGEND_PRED=RED_FN=GREEN_OVERLAP=YELLOW"
                img_name_pred = (
                    f"t{t:06d}_{legend}_FN({fnx},{fny})_nearestPRED(NA,NA)_dNA.png"
                )

            out_path2 = out_img_dir_fn_vs_pred / img_name_pred
            cv2.imwrite(str(out_path2), composed)
        _progress(frames.index(t)+1, len(frames), 'stage7-test-fn')

    cap.release()
    # write CSV
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t','fn_x','fn_y','tp_x','tp_y','dist','image_path'])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fns, no_tp_cnt


def stage7_test_fn_nearest_tp_analysis(
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
            print(f"[stage7_test] Stage-5 test dir not found: {stage9_video_dir}")
        return
    thr_dirs = sorted([p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith('thr_')])
    if not thr_dirs:
        if verbose:
            print(f"[stage7_test] No thr_* folders in: {stage9_video_dir}")
        return
    total_fns_all = 0; total_no_tp = 0
    for td in thr_dirs:
        fns_csv = td / 'fns.csv'; tps_csv = td / 'tps.csv'
        if not (fns_csv.exists() and tps_csv.exists()):
            if verbose:
                print(f"[stage7_test] Skipping {td.name} (missing fns.csv or tps.csv)")
            continue
        total_fns, no_tp = _analyze_threshold_dir_and_render(
            thr_dir=td,
            orig_video_path=orig_video_path,
            box_w=box_w,
            box_h=box_h,
            color=(0,255,255),  # yellow
            thickness=thickness,
            verbose=verbose,
        )
        total_fns_all += total_fns; total_no_tp += no_tp
    if verbose:
        print(f"[stage7_test] Done. Total FNs analyzed: {total_fns_all}. FNs without any TP: {total_no_tp}.")


__all__ = ['stage7_test_fn_nearest_tp_analysis']

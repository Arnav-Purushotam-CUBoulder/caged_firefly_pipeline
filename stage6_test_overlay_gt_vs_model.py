#!/usr/bin/env python3
"""
Stage 6 (test) — Overlay GT vs Model and per-threshold TP/FP/FN videos for caged_fireflies.

Uses:
  - Predictions: Stage4 '*_gauss.csv' (x,y,t,firefly_logit,background_logit)
  - Normalized GT: the *_norm_offset*.csv produced by stage5_test_validate in the per-video folder
  - Per-threshold partitions: thr_*/(tps.csv, fps.csv, fns.csv) produced by stage5_test_validate

Writes:
  - A single overlay video (GT=green, Model=red, overlap=yellow) under params.POST_STAGE10_DIR
  - Per-threshold TP/FP/FN videos under <out>/{stem}_by_threshold/
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import re

import cv2
import numpy as np

import params


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    frac  = min(1.0, max(0.0, i / float(total)))
    bar   = '=' * int(frac * 50) + ' ' * (50 - int(frac * 50))
    print(f"\r{tag} [{bar}] {int(frac*100):3d}%", end='' if i < total else '\n')

def _clamp_box(x0:int, y0:int, w:int, h:int, W:int, H:int) -> Tuple[int,int,int,int]:
    w = max(1, min(int(w), W))
    h = max(1, min(int(h), H))
    x0 = max(0, min(int(x0), W - w))
    y0 = max(0, min(int(y0), H - h))
    return x0, y0, w, h

def _read_preds_by_frame(csv_path: Path, max_frames: Optional[int]=None) -> Dict[int, List[Tuple[float,float,int,int]]]:
    rows = list(csv.DictReader(csv_path.open()))
    by_frame: Dict[int, List[Tuple[float,float,int,int]]] = {}
    for r in rows:
        try:
            f = int(r.get('t', r.get('frame', 0)))
            if max_frames is not None and f >= max_frames:
                continue
            x = float(r['x']); y = float(r['y'])
            # Stage4 does not carry w/h; use configured BOX_SIZE_PX
            w = int(getattr(params, 'BOX_SIZE_PX', 40))
            h = int(getattr(params, 'BOX_SIZE_PX', 40))
            by_frame.setdefault(f, []).append((x,y,w,h))
        except Exception:
            continue
    return by_frame

def _read_gt_by_frame(gt_norm_csv: Path, max_frames: Optional[int]=None) -> Dict[int, List[Tuple[int,int]]]:
    rows = list(csv.DictReader(gt_norm_csv.open()))
    by_frame: Dict[int, List[Tuple[int,int]]] = {}
    for r in rows:
        try:
            f = int(r['t'])
            if max_frames is not None and f >= max_frames:
                continue
            x = int(round(float(r['x']))); y = int(round(float(r['y'])))
            by_frame.setdefault(f, []).append((x,y))
        except Exception:
            continue
    return by_frame

def _find_norm_gt_in(post9_dir: Path) -> Optional[Path]:
    try:
        cands = sorted(post9_dir.glob('*_norm_offset*.csv'))
        return cands[0] if cands else None
    except Exception:
        return None

def _read_thr_partitions(thr_dir: Path, max_frames: Optional[int]):
    def _read_xy(path: Path) -> Dict[int, List[Tuple[int,int]]]:
        by_t: Dict[int, List[Tuple[int,int]]] = {}
        if not path.exists():
            return by_t
        rows = list(csv.DictReader(path.open()))
        for r in rows:
            try:
                t = int(r.get('t'))
                if max_frames is not None and t >= max_frames:
                    continue
                x = int(round(float(r['x']))); y = int(round(float(r['y'])))
                by_t.setdefault(t, []).append((x,y))
            except Exception:
                continue
        return by_t
    fps_by_t = _read_xy(thr_dir / 'fps.csv')
    tps_by_t = _read_xy(thr_dir / 'tps.csv')
    fns_by_t = _read_xy(thr_dir / 'fns.csv')
    return fps_by_t, tps_by_t, fns_by_t

def _list_thr_dirs(stage9_dir: Path) -> List[Tuple[str, Path]]:
    dirs = []
    for p in sorted(stage9_dir.iterdir()):
        if p.is_dir() and p.name.startswith('thr_') and p.name.endswith('px'):
            dirs.append((p.name, p))
    return dirs


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
    render_threshold_overlays: bool = True,
    thr_box_w: Optional[int] = None,
    thr_box_h: Optional[int] = None,
) -> None:
    # 1) Original overlay (GT vs Model)
    gt_norm_csv_path = _find_norm_gt_in(post9_dir)
    if gt_norm_csv_path is None:
        raise FileNotFoundError(f"[stage6_test] Missing normalized GT CSV in {post9_dir}. Run stage5_test_validate first.")

    preds_by_frame = _read_preds_by_frame(pred_csv_path, max_frames=max_frames)
    gt_by_frame    = _read_gt_by_frame(gt_norm_csv_path, max_frames=max_frames)

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage6_test] Could not open video: {orig_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _ensure_dir(out_video_path.parent)
    out  = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (W,H))

    RED     = (0,0,255)
    GREEN   = (0,255,0)
    YELLOW  = (0,255,255)

    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        red_layer   = np.zeros_like(frame)
        green_layer = np.zeros_like(frame)

        # Model (preds) in RED
        for (x,y,w,h) in preds_by_frame.get(fr, []):
            x0 = int(round(float(x) - w/2.0)); y0 = int(round(float(y) - h/2.0))
            x0,y0,w,h = _clamp_box(x0,y0,w,h,W,H)
            cv2.rectangle(red_layer, (x0,y0), (x0+w, y0+h), RED, thickness)
        # GT in GREEN
        for (gx,gy) in gt_by_frame.get(fr, []):
            x0 = int(round(gx - gt_box_w/2.0)); y0 = int(round(gy - gt_box_h/2.0))
            x0,y0,gw,gh = _clamp_box(x0,y0,gt_box_w,gt_box_h,W,H)
            cv2.rectangle(green_layer, (x0,y0), (x0+gw, y0+gh), GREEN, thickness)
        red_mask   = np.any(red_layer > 0, axis=2)
        green_mask = np.any(green_layer > 0, axis=2)
        overlap_mask = red_mask & green_mask
        only_red   = red_mask & ~overlap_mask
        frame[only_red] = RED
        only_green = green_mask & ~overlap_mask
        frame[only_green] = GREEN
        frame[overlap_mask] = YELLOW

        out.write(frame)
        _progress(fr+1, limit, 'post10-overlay'); fr += 1
    cap.release(); out.release()
    print(f"[stage6_test] Wrote overlay: {out_video_path}")

    # 2) Per-threshold TP/FP/FN overlays
    if render_threshold_overlays:
        thr_dirs = _list_thr_dirs(post9_dir)
        if not thr_dirs:
            print(f"[stage6_test] No thr_* folders found in: {post9_dir} — skipping TP/FP/FN videos.")
            return
        per_thr_out_dir = out_video_path.parent / f"{orig_video_path.stem}_by_threshold"
        _ensure_dir(per_thr_out_dir)
        bw = int(thr_box_w if thr_box_w is not None else gt_box_w)
        bh = int(thr_box_h if thr_box_h is not None else gt_box_h)
        for thr_name, thr_path in thr_dirs:
            cap = cv2.VideoCapture(str(orig_video_path))
            if not cap.isOpened():
                raise RuntimeError(f"[stage6_test] Could not open video: {orig_video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25
            W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            safe_thr = re.sub(r'[^0-9A-Za-z_.-]+', '', thr_name.replace('thr_', 'thr'))
            out_path = per_thr_out_dir / f"{orig_video_path.stem}_tp_fp_fn_{safe_thr}.mp4"
            out  = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W,H))
            RED=(0,0,255); GREEN=(0,255,0); YELLOW=(0,255,255)
            fps_by_t, tps_by_t, fns_by_t = _read_thr_partitions(thr_path, max_frames)
            limit = total_frames if max_frames is None else min(total_frames, max_frames)
            fr = 0
            while True:
                if fr >= limit:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                for (x,y) in fps_by_t.get(fr, []):
                    x0 = int(round(x - bw/2.0)); y0 = int(round(y - bh/2.0))
                    x0,y0,w,h = _clamp_box(x0,y0,bw,bh,W,H)
                    cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), RED, params.OVERLAY_BOX_THICKNESS)
                for (x,y) in fns_by_t.get(fr, []):
                    x0 = int(round(x - bw/2.0)); y0 = int(round(y - bh/2.0))
                    x0,y0,w,h = _clamp_box(x0,y0,bw,bh,W,H)
                    cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), GREEN, params.OVERLAY_BOX_THICKNESS)
                for (x,y) in tps_by_t.get(fr, []):
                    x0 = int(round(x - bw/2.0)); y0 = int(round(y - bh/2.0))
                    x0,y0,w,h = _clamp_box(x0,y0,bw,bh,W,H)
                    cv2.rectangle(frame, (x0,y0), (x0+w, y0+h), YELLOW, params.OVERLAY_BOX_THICKNESS)
                out.write(frame)
                _progress(fr+1, limit, f"post10-threshold:{thr_name}"); fr += 1
            cap.release(); out.release()
            print(f"[stage6_test] Wrote TP/FP/FN video: {out_path}")


__all__ = ['stage6_test_overlay_gt_vs_model']

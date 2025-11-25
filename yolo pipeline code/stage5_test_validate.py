#!/usr/bin/env python3
"""
Stage 5 (test) — Validate YOLO Stage 3 predictions against ground truth.

For each video, this stage:
  - Reads GT CSV (x,y,t) or (x,y,w,h,frame) and normalizes to x,y,t.
  - Reads prediction points CSV produced from Stage 3 boxes:
        x, y, t, firefly_logit, background_logit
  - For each distance threshold in DIST_THRESHOLDS_PX, runs greedy per-frame
    matching to compute TP/FP/FN and mean error.
  - Writes per-threshold CSVs under:
        out_dir/thr_<thr>px/{tps.csv,fps.csv,fns.csv}
  - Prints summary metrics per threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import math
import re

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


def _softmax_conf_firefly(b: float, f: float) -> float:
    m = max(b, f)
    eb = math.exp(b - m)
    ef = math.exp(f - m)
    denom = eb + ef
    return ef / denom if denom > 0 else 0.5


def _pairwise_dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _greedy_match_full(
    frame_gts: List[Tuple[float, float]],
    frame_preds_xy: List[Tuple[float, float]],
    max_dist_px: float,
):
    """Greedy 1-1 frame matching. Returns (matches, unmatched_pred_idxs, unmatched_gt_idxs)."""
    nG = len(frame_gts)
    nP = len(frame_preds_xy)
    if nG == 0 and nP == 0:
        return [], [], []
    max_d2 = max_dist_px * max_dist_px
    used_g = [False] * nG
    used_p = [False] * nP
    pairs = []
    for gi, g in enumerate(frame_gts):
        for pi, p in enumerate(frame_preds_xy):
            d2 = _pairwise_dist2(g, p)
            if d2 <= max_d2:
                pairs.append((d2, gi, pi))
    pairs.sort(key=lambda x: x[0])
    matches = []
    for d2, gi, pi in pairs:
        if not used_g[gi] and not used_p[pi]:
            used_g[gi] = True
            used_p[pi] = True
            matches.append((gi, pi, math.sqrt(d2)))
    unmatched_pred = [i for i, u in enumerate(used_p) if not u]
    unmatched_gt = [i for i, u in enumerate(used_g) if not u]
    return matches, unmatched_pred, unmatched_gt


def _parse_frame_index(s: str) -> Optional[int]:
    """Extract last integer from a frame string; returns None if not found."""
    ms = re.findall(r"\d+", str(s))
    if not ms:
        return None
    try:
        return int(ms[-1])
    except Exception:
        return None


def _read_and_normalize_gt(
    gt_csv: Path,
    out_dir: Path,
    gt_t_offset: int,
    max_frames: Optional[int],
) -> Tuple[Dict[int, List[Tuple[float, float]]], Path]:
    """Load GT and normalize time to zero-based t. Returns (gt_by_t, norm_csv_path)."""
    _ensure_dir(out_dir)
    norm_path = out_dir / "gt_norm.csv"
    gt_by_t: Dict[int, List[Tuple[float, float]]] = {}
    rows_norm: List[Dict[str, int]] = []

    with gt_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        if {"x", "y", "t"} <= cols:
            for row in r:
                try:
                    x = int(round(float(row["x"])))
                    y = int(round(float(row["y"])))
                    t_raw = int(round(float(row["t"])))
                    t = t_raw - int(gt_t_offset)
                    if t < 0:
                        continue
                    if max_frames is not None and t >= max_frames:
                        continue
                except Exception:
                    continue
                gt_by_t.setdefault(t, []).append((x, y))
                rows_norm.append({"x": x, "y": y, "t": t})
        elif {"x", "y", "frame"} <= cols:
            for row in r:
                try:
                    x = int(round(float(row["x"])))
                    y = int(round(float(row["y"])))
                    raw_t = _parse_frame_index(row.get("frame", ""))
                    if raw_t is None:
                        continue
                    t = int(raw_t) - int(gt_t_offset)
                    if t < 0:
                        continue
                    if max_frames is not None and t >= max_frames:
                        continue
                except Exception:
                    continue
                gt_by_t.setdefault(t, []).append((x, y))
                rows_norm.append({"x": x, "y": y, "t": t})
        else:
            raise ValueError(
                f"[stage5_test] Unsupported GT CSV schema. Expected x,y,t or x,y,frame; found: {sorted(cols)}"
            )

    with norm_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for r in rows_norm:
            w.writerow(r)
    return gt_by_t, norm_path


def _dedupe_gt(
    gt_by_t: Dict[int, List[Tuple[float, float]]],
    thr_px: float,
) -> Dict[int, List[Tuple[float, float]]]:
    """Simple distance-based GT dedupe per frame: keep first in each cluster."""
    if thr_px <= 0:
        return gt_by_t
    thr2 = thr_px * thr_px
    out: Dict[int, List[Tuple[float, float]]] = {}
    for t, pts in gt_by_t.items():
        kept: List[Tuple[float, float]] = []
        for p in pts:
            if not kept:
                kept.append(p)
                continue
            if any(_pairwise_dist2(p, q) <= thr2 for q in kept):
                continue
            kept.append(p)
        out[t] = kept
    return out


def _read_predictions(
    pred_csv: Path,
    max_frames: Optional[int],
) -> Dict[int, List[Dict[str, float]]]:
    """Read prediction points CSV: x,y,t,firefly_logit,background_logit."""
    preds_by_t: Dict[int, List[Dict[str, float]]] = {}
    with pred_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(float(row.get("t", row.get("frame", 0))))
                if max_frames is not None and t >= max_frames:
                    continue
                x = float(row["x"])
                y = float(row["y"])
                ffl = float(row.get("firefly_logit", "nan"))
                bkg = float(row.get("background_logit", "nan"))
                conf = _softmax_conf_firefly(bkg, ffl)
            except Exception:
                continue
            preds_by_t.setdefault(t, []).append(
                {"x": x, "y": y, "firefly_logit": ffl, "background_logit": bkg, "conf": conf}
            )
    return preds_by_t


def _evaluate_frames(
    gt_by_t: Dict[int, List[Tuple[float, float]]],
    preds_by_t: Dict[int, List[Dict[str, float]]],
    thr_px: float,
):
    """Return (fps_by_t, tps_by_t, fns_by_t, (TP,FP,FN,mean_err))."""
    from collections import defaultdict

    fps_by_t: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    tps_by_t: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    fns_by_t: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    all_dists: List[float] = []

    frames = sorted(set(gt_by_t.keys()) | set(preds_by_t.keys()))
    for t in frames:
        gts = gt_by_t.get(t, [])
        preds = preds_by_t.get(t, [])
        preds_xy = [(p["x"], p["y"]) for p in preds]
        matches, unmatched_pred_idxs, unmatched_gt_idxs = _greedy_match_full(
            gts, preds_xy, thr_px
        )
        for gi, pi, d in matches:
            tps_by_t[t].append(preds[pi])
            all_dists.append(d)
        for pi in unmatched_pred_idxs:
            fps_by_t[t].append(preds[pi])
        for gi in unmatched_gt_idxs:
            gx, gy = gts[gi]
            fns_by_t[t].append((gx, gy))

    TP = sum(len(v) for v in tps_by_t.values())
    FP = sum(len(v) for v in fps_by_t.values())
    FN = sum(len(v) for v in fns_by_t.values())
    mean_err = (sum(all_dists) / len(all_dists)) if all_dists else 0.0
    return fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err)


def _write_thr_csvs(
    out_root: Path,
    thr: float,
    fps_by_t,
    tps_by_t,
    fns_by_t,
) -> None:
    thr_dir = out_root / f"thr_{float(thr):.1f}px"
    _ensure_dir(thr_dir)
    csv_fp = thr_dir / "fps.csv"
    csv_tp = thr_dir / "tps.csv"
    csv_fn = thr_dir / "fns.csv"

    with csv_fp.open("w", newline="") as f_fp, csv_tp.open(
        "w", newline=""
    ) as f_tp, csv_fn.open("w", newline="") as f_fn:
        w_fp = csv.DictWriter(f_fp, fieldnames=["x", "y", "t", "confidence"])
        w_tp = csv.DictWriter(f_tp, fieldnames=["x", "y", "t", "confidence"])
        w_fn = csv.DictWriter(f_fn, fieldnames=["x", "y", "t"])
        w_fp.writeheader()
        w_tp.writeheader()
        w_fn.writeheader()
        for t in sorted(set(list(fps_by_t.keys()) + list(tps_by_t.keys()) + list(fns_by_t.keys()))):
            for p in fps_by_t.get(t, []):
                w_fp.writerow(
                    {
                        "x": float(p["x"]),
                        "y": float(p["y"]),
                        "t": int(t),
                        "confidence": float(p.get("conf", float("nan"))),
                    }
                )
            for p in tps_by_t.get(t, []):
                w_tp.writerow(
                    {
                        "x": float(p["x"]),
                        "y": float(p["y"]),
                        "t": int(t),
                        "confidence": float(p.get("conf", float("nan"))),
                    }
                )
            for (gx, gy) in fns_by_t.get(t, []):
                w_fn.writerow({"x": float(gx), "y": float(gy), "t": int(t)})


def stage5_test_validate_against_gt(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    out_dir: Path,
    dist_thresholds: List[float],
    crop_w: int,
    crop_h: int,
    gt_t_offset: int,
    max_frames: Optional[int],
    only_firefly_rows: bool,
    gt_dedupe_dist_threshold_px: float,
) -> None:
    """Run validation for a single video/prediction CSV and write per-threshold outputs."""
    _ensure_dir(out_dir)
    # 1) Load and normalize GT
    gt_by_t, norm_gt_csv = _read_and_normalize_gt(
        gt_csv_path, out_dir, gt_t_offset, max_frames
    )
    # 2) Dedupe GT (simple distance-based)
    gt_by_t = _dedupe_gt(gt_by_t, float(gt_dedupe_dist_threshold_px))
    # Overwrite normalized GT with deduped version
    with norm_gt_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for t in sorted(gt_by_t.keys()):
            for (x, y) in gt_by_t[t]:
                w.writerow({"x": int(x), "y": int(y), "t": int(t)})

    # 3) Load predictions
    preds_by_t = _read_predictions(pred_csv_path, max_frames)

    print("\n=== Detection Metrics (point, same-frame, dist sweep) ===")
    print(f"Video: {orig_video_path}")
    print(f"Pred CSV: {pred_csv_path}")
    print(f"GT CSV (raw): {gt_csv_path}")
    print(f"GT offset: {gt_t_offset} (t' = t - offset)")
    if max_frames is not None:
        print(f"Max frames considered: {max_frames}")
    print(f"Distance thresholds (px): {', '.join(str(t) for t in dist_thresholds)}")
    print()

    for thr in dist_thresholds:
        fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err) = _evaluate_frames(
            gt_by_t, preds_by_t, float(thr)
        )
        _write_thr_csvs(out_dir, float(thr), fps_by_t, tps_by_t, fns_by_t)
        prec = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"Threshold: {float(thr):.1f}px")
        print(f"  TP: {TP}   FP: {FP}   FN: {FN}")
        print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
        print(f"  Mean error (px): {mean_err:.3f}\n")


__all__ = ["stage5_test_validate_against_gt"]


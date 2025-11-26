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


def _center_crop_clamped_with_origin(
    img: np.ndarray, cx: float, cy: float, w: int, h: int
):
    """Return (crop, x0, y0) for a crop centered at (cx,cy) clipped to image."""
    H, W = img.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    return img[y0 : y0 + h, x0 : x0 + w].copy(), x0, y0


def _center_crop_clamped(
    img: np.ndarray, cx: float, cy: float, w: int, h: int
) -> np.ndarray:
    """Simple center crop (no origin returned)."""
    H, W = img.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    return img[y0 : y0 + h, x0 : x0 + w].copy()


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


def _filter_gt_by_brightness_and_area(
    video_path: Path,
    gt_by_t: Dict[int, List[Tuple[float, float]]],
    *,
    crop_w: int,
    crop_h: int,
    bright_max_threshold: int,
    area_threshold_px: int,
    max_frames: Optional[int],
    area_min_pixel_brightness: int,
) -> Tuple[Dict[int, List[Tuple[float, float]]], int, int]:
    """Keep GT points whose crop meets brightness and area criteria."""
    from collections import defaultdict

    total = sum(len(v) for v in gt_by_t.values())
    if total == 0:
        return gt_by_t, 0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage5_test] Could not open video for GT filtering: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = total_frames if max_frames is None else min(total_frames, max_frames)

    filtered: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    kept = 0
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        points = gt_by_t.get(fr, [])
        if points:
            for (x, y) in points:
                crop = _center_crop_clamped(frame, float(x), float(y), crop_w, crop_h)
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                max_val = int(gray.max()) if gray.size else 0
                if max_val < int(bright_max_threshold):
                    continue
                _, bin_img = cv2.threshold(
                    gray, int(area_min_pixel_brightness), 255, cv2.THRESH_BINARY
                )
                num, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    bin_img, connectivity=8
                )
                area = (
                    int(stats[1:, cv2.CC_STAT_AREA].max())
                    if (num > 1 and stats.shape[0] > 1)
                    else 0
                )
                if area >= int(area_threshold_px):
                    filtered[fr].append((int(x), int(y)))
                    kept += 1
        if getattr(params, "STAGE5_SHOW_PROGRESS", False):
            _progress(fr + 1, limit, "stage5-test-gt-filter")
        fr += 1
    cap.release()
    return filtered, kept, total


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


def _write_crops_and_csvs_for_threshold(
    video_path: Path,
    out_root: Path,
    thr: float,
    fps_by_t,
    tps_by_t,
    fns_by_t,
    crop_w: int,
    crop_h: int,
    max_frames: Optional[int],
) -> None:
    """Write per-threshold TP/FP/FN CSVs and crops, similar to original pipeline."""
    thr_dir = out_root / f"thr_{float(thr):.1f}px"
    crops_dir_fp = thr_dir / "fp_crops"
    crops_dir_tp = thr_dir / "tp_crops"
    crops_dir_fn = thr_dir / "fn_crops"
    for d in (thr_dir, crops_dir_fp, crops_dir_tp, crops_dir_fn):
        _ensure_dir(d)

    csv_fp = thr_dir / "fps.csv"
    csv_tp = thr_dir / "tps.csv"
    csv_fn = thr_dir / "fns.csv"

    with csv_fp.open("w", newline="") as f_fp, csv_tp.open(
        "w", newline=""
    ) as f_tp, csv_fn.open("w", newline="") as f_fn:
        w_fp = csv.DictWriter(
            f_fp, fieldnames=["x", "y", "t", "filepath", "confidence"]
        )
        w_tp = csv.DictWriter(
            f_tp, fieldnames=["x", "y", "t", "filepath", "confidence"]
        )
        w_fn = csv.DictWriter(
            f_fn, fieldnames=["x", "y", "t", "filepath", "confidence"]
        )
        w_fp.writeheader()
        w_tp.writeheader()
        w_fn.writeheader()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"[stage5_test] Could not open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit = total if max_frames is None else min(total, max_frames)

        fr = 0
        while True:
            if fr >= limit:
                break
            ok, frame = cap.read()
            if not ok:
                break

            # False positives
            for p in fps_by_t.get(fr, []):
                x = float(p["x"])
                y = float(p["y"])
                conf = float(p.get("conf", float("nan")))
                crop, x0c, y0c = _center_crop_clamped_with_origin(
                    frame, x, y, crop_w, crop_h
                )
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                fname = (
                    f"FP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_"
                    f"max{max_val}.png"
                )
                outp = crops_dir_fp / fname
                cv2.imwrite(str(outp), crop)
                w_fp.writerow(
                    {
                        "x": float(x),
                        "y": float(y),
                        "t": int(fr),
                        "filepath": str(outp),
                        "confidence": conf,
                    }
                )

            # True positives
            for p in tps_by_t.get(fr, []):
                x = float(p["x"])
                y = float(p["y"])
                conf = float(p.get("conf", float("nan")))
                crop, x0c, y0c = _center_crop_clamped_with_origin(
                    frame, x, y, crop_w, crop_h
                )
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                fname = (
                    f"TP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_"
                    f"max{max_val}.png"
                )
                outp = crops_dir_tp / fname
                cv2.imwrite(str(outp), crop)
                w_tp.writerow(
                    {
                        "x": float(x),
                        "y": float(y),
                        "t": int(fr),
                        "filepath": str(outp),
                        "confidence": conf,
                    }
                )

            # False negatives (GT points with no match)
            for (gx, gy) in fns_by_t.get(fr, []):
                crop, x0c, y0c = _center_crop_clamped_with_origin(
                    frame, gx, gy, crop_w, crop_h
                )
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                fname = f"FN_t{fr:06d}_x{int(round(gx))}_y{int(round(gy))}_max{max_val}.png"
                outp = crops_dir_fn / fname
                cv2.imwrite(str(outp), crop)
                w_fn.writerow(
                    {
                        "x": float(gx),
                        "y": float(gy),
                        "t": int(fr),
                        "filepath": str(outp),
                        "confidence": "",
                    }
                )

            fr += 1
            if getattr(params, "STAGE5_SHOW_PROGRESS", False):
                _progress(fr, limit, f"stage5-test-crops@thr={thr:.1f}px")

        cap.release()


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
    # 2) Apply brightness + area filter to GT (mirror Stage 2 params)
    bright_thr = int(getattr(params, "STAGE2_BRIGHT_MAX_THRESHOLD", 190))
    area_thr = int(getattr(params, "STAGE2_AREA_INTENSITY_THR", 190))
    min_pixels = int(getattr(params, "STAGE2_AREA_MIN_BRIGHT_PIXELS", 0))
    if min_pixels > 0:
        gt_by_t, kept_gt, total_gt = _filter_gt_by_brightness_and_area(
            orig_video_path,
            gt_by_t,
            crop_w=crop_w,
            crop_h=crop_h,
            bright_max_threshold=bright_thr,
            area_threshold_px=min_pixels,
            max_frames=max_frames,
            area_min_pixel_brightness=area_thr,
        )
        print(
            f"[stage5_test] GT brightness+area filter: kept={kept_gt} of {total_gt} "
            f"(bright_thr={bright_thr}, area_thr={area_thr}, min_pixels={min_pixels})"
        )
    # 3) Dedupe GT (simple distance-based)
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
        _write_crops_and_csvs_for_threshold(
            orig_video_path,
            out_dir,
            float(thr),
            fps_by_t,
            tps_by_t,
            fns_by_t,
            crop_w,
            crop_h,
            max_frames,
        )
        prec = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"Threshold: {float(thr):.1f}px")
        print(f"  TP: {TP}   FP: {FP}   FN: {FN}")
        print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
        print(f"  Mean error (px): {mean_err:.3f}\n")


__all__ = ["stage5_test_validate_against_gt"]

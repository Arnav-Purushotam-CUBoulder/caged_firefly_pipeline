#!/usr/bin/env python3
"""
Stage 5 (test) — Validate Stage4 predictions against ground truth for caged_fireflies.

This mirrors the night_time Stage 9 validator, adapted to the caged_fireflies
pipeline and params. It:
  - Normalizes GT time (subtracts offset) and writes a normalized GT CSV.
  - Deduplicates GT by centroid distance.
  - Reads predictions from Stage4 '*_gauss.csv' (expects x,y,t,firefly_logit,background_logit).
  - Performs per-frame greedy matching under a distance threshold sweep.
  - Saves FP/TP/FN crops and per-threshold CSVs under a per-video output folder.

No audit trail; all outputs are contained under params.DIR_STAGE5_TEST_OUT/<video_stem>/.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import math
import sys
import re
import os

import cv2
import numpy as np

import params


# ───────────────────────── helpers ─────────────────────────
_BAR = 50

def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _softmax_conf_firefly(b: float, f: float) -> float:
    m = max(b, f)
    eb = math.exp(b - m)
    ef = math.exp(f - m)
    denom = eb + ef
    return ef / denom if denom > 0 else 0.5

def _pairwise_dist2(a, b):
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return dx*dx + dy*dy

def _greedy_match_full(frame_gts, frame_preds_xy, max_dist_px):
    """Greedy 1-1 frame matching. Returns (matches, unmatched_pred_idxs, unmatched_gt_idxs)."""
    nG = len(frame_gts); nP = len(frame_preds_xy)
    if nG == 0 and nP == 0:
        return [], [], []
    max_d2 = max_dist_px * max_dist_px
    used_g = [False]*nG; used_p = [False]*nP
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
            used_g[gi] = True; used_p[pi] = True
            matches.append((gi, pi, math.sqrt(d2)))
    unmatched_pred = [i for i, u in enumerate(used_p) if not u]
    unmatched_gt   = [i for i, u in enumerate(used_g) if not u]
    return matches, unmatched_pred, unmatched_gt

def _center_crop_clamped(img: np.ndarray, cx: float, cy: float, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w].copy()

def sub_abs(n: int) -> str:
    return (f"minus{abs(n)}" if n < 0 else f"plus{abs(n)}")

_NAMING_BIN_THR = 50  # internal threshold used only for crop filename stats


# ────────────────────── GT & predictions I/O ──────────────────────
def _read_and_normalize_gt(gt_csv: Path, gt_t_offset: int, out_dir: Path, max_frames: Optional[int]):
    """Load GT and normalize time to zero-based t, handling multiple schemas.

    Supported input schemas:
      - x,y,t                        (direct, preferred)
      - x,y,w,h,frame                (from caged_firefly_flash_annotation_tool)

    For the annotation-tool schema, t is parsed from the digits in the 'frame'
    filename (last integer group). We then subtract gt_t_offset from t to
    align with the video index. Writes a normalized copy with columns x,y,t.

    Returns: (gt_by_t dict, normalized_csv_path, gt_wh_by_t or None)
    """
    _ensure_dir(out_dir)
    norm_path = out_dir / f"{gt_csv.stem}_norm_offset{sub_abs(gt_t_offset)}.csv"
    from collections import defaultdict
    gt_by_t: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    gt_wh_by_t: Optional[Dict[int, List[Tuple[int,int]]]] = None
    rows_norm = []
    with gt_csv.open('r', newline='') as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])

        def _parse_frame_index(frame_value: str) -> Optional[int]:
            try:
                base = os.path.basename(str(frame_value))
                digs = re.findall(r"\d+", base)
                return int(digs[-1]) if digs else None
            except Exception:
                return None

        if {'x','y','t'} <= cols:
            # Direct x,y,t schema
            for row in r:
                try:
                    x = int(round(float(row['x'])))
                    y = int(round(float(row['y'])))
                    t = int(round(float(row['t']))) - int(gt_t_offset)
                    if t < 0:
                        continue
                    if max_frames is not None and t >= max_frames:
                        continue
                except Exception:
                    continue
                gt_by_t[t].append((x,y))
                rows_norm.append({'x':x,'y':y,'t':t})
        elif {'x','y','frame'} <= cols:
            # Annotation-tool schema: x,y,w,h,frame (w,h ignored)
            gt_wh_by_t = defaultdict(list)
            for row in r:
                try:
                    x = int(round(float(row['x'])))
                    y = int(round(float(row['y'])))
                    # Try read per-point w,h; fall back to BOX_SIZE_PX if missing
                    try:
                        w_ = int(round(float(row.get('w', str(getattr(params, 'BOX_SIZE_PX', 40))))))
                        h_ = int(round(float(row.get('h', str(getattr(params, 'BOX_SIZE_PX', 40))))))
                    except Exception:
                        w_ = int(getattr(params, 'BOX_SIZE_PX', 40))
                        h_ = int(getattr(params, 'BOX_SIZE_PX', 40))
                    raw_t = _parse_frame_index(row.get('frame',''))
                    if raw_t is None:
                        continue
                    t = int(raw_t) - int(gt_t_offset)
                    if t < 0:
                        continue
                    if max_frames is not None and t >= max_frames:
                        continue
                except Exception:
                    continue
                gt_by_t[t].append((x,y))
                gt_wh_by_t[t].append((w_, h_))
                rows_norm.append({'x':x,'y':y,'t':t})
        else:
            raise ValueError(
                f"[stage5_test] Unsupported GT CSV schema. Expected x,y,t or x,y,w,h,frame; found: {sorted(cols)}"
            )
    with norm_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['x','y','t'])
        w.writeheader()
        for r in rows_norm:
            w.writerow(r)
    return gt_by_t, norm_path, gt_wh_by_t

def _read_predictions(pred_csv: Path, only_firefly_rows: bool, max_frames: Optional[int]):
    """Accepts Stage4 gauss CSV (x,y,t,firefly_logit,background_logit). Returns dict t -> list of preds."""
    from collections import defaultdict
    preds_by_t: Dict[int, List[Dict]] = defaultdict(list)
    with pred_csv.open('r', newline='') as f:
        r = csv.DictReader(f)
        cols = set(r.fieldnames or [])
        has_t = 't' in cols or 'frame' in cols
        need_logits = {'background_logit','firefly_logit'}.issubset(cols)
        if not need_logits:
            raise ValueError(f"[stage5_test] Predictions CSV requires logits columns; found: {r.fieldnames}")
        for row in r:
            try:
                # Stage4 has no class column; treat all rows as firefly
                t = int(row.get('t', row.get('frame', 0))) if has_t else int(row['frame'])
                if max_frames is not None and t >= max_frames:
                    continue
                x = float(row['x']); y = float(row['y'])
                b = float(row['background_logit']); ffl = float(row['firefly_logit'])
                conf = _softmax_conf_firefly(b, ffl)
            except Exception:
                continue
            preds_by_t[t].append({'x':x,'y':y,'b':b,'f':ffl,'conf':conf})
    return preds_by_t


# ────────────────────── GT filtering and dedupe ──────────────────────
def _write_norm_gt_from_map(out_path: Path, gt_by_t: Dict[int, List[Tuple[int,int]]]):
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['x','y','t'])
        w.writeheader()
        for t in sorted(gt_by_t.keys()):
            for (x,y) in gt_by_t[t]:
                w.writerow({'x': int(x), 'y': int(y), 't': int(t)})

def _gaussian_kernel(w: int, h: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.full((h, w), 1.0/(h*w), dtype=np.float32)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((x - xc)**2 + (y - yc)**2) / (2.0 * sigma**2)).astype(np.float32)
    s = float(g.sum())
    if s > 0:
        g /= s
    return g

def _intensity_centroid(img_gray: np.ndarray, gaussian_sigma: float = 0.0) -> Tuple[float, float]:
    if img_gray.size == 0:
        return 0.0, 0.0
    img = img_gray.astype(np.float32)
    if gaussian_sigma and gaussian_sigma > 0:
        gh, gw = img.shape[:2]
        G = _gaussian_kernel(gw, gh, gaussian_sigma)
        img = img * G
    total = float(img.sum())
    if total <= 1e-6:
        H, W = img.shape[:2]
        return (W/2.0, H/2.0)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    cx = float((xs * img).sum() / total)
    cy = float((ys * img).sum() / total)
    return cx, cy

def _refine_gt_by_gaussian_centroid(
    video_path: Path,
    gt_by_t: Dict[int, List[Tuple[int,int]]],
    *,
    patch_w: int,
    patch_h: int,
    per_point_wh: Optional[Dict[int, List[Tuple[int,int]]]] = None,
    sigma: float,
    max_frames: Optional[int],
    save_dir: Optional[Path] = None,
) -> Dict[int, List[Tuple[int,int]]]:
    from collections import defaultdict
    if not gt_by_t:
        return gt_by_t
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage5_test] Could not open video for GT gaussian refine: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    refined: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    if save_dir is not None:
        _ensure_dir(save_dir)
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        dets = gt_by_t.get(fr, [])
        if dets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sizes = (per_point_wh.get(fr) if (per_point_wh is not None) else None)
            for idx_pt, (x, y) in enumerate(dets):
                # Compute crop bounds like Stage 4
                H, W = gray.shape[:2]
                if sizes is not None and idx_pt < len(sizes):
                    w = max(1, int(round(sizes[idx_pt][0])))
                    h = max(1, int(round(sizes[idx_pt][1])))
                else:
                    w = max(1, int(round(patch_w)))
                    h = max(1, int(round(patch_h)))
                x0 = int(round(float(x) - w/2.0)); y0 = int(round(float(y) - h/2.0))
                x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
                crop = gray[y0:y0+h, x0:x0+w]
                ccx, ccy = _intensity_centroid(crop, float(sigma))
                new_cx = x0 + ccx
                new_cy = y0 + ccy
                rx, ry = int(round(new_cx)), int(round(new_cy))
                refined[fr].append((rx, ry))
                if save_dir is not None:
                    color_crop = frame[y0:y0+h, x0:x0+w].copy()
                    px = int(round(new_cx - x0)); py = int(round(new_cy - y0))
                    if 0 <= py < color_crop.shape[0] and 0 <= px < color_crop.shape[1]:
                        color_crop[py, px] = (0, 0, 255)
                    out_name = f"gt_refined_t{fr:06d}_x{rx}_y{ry}_{w}x{h}.png"
                    cv2.imwrite(str(save_dir / out_name), color_crop)
        fr += 1
    cap.release()
    return refined

def _filter_gt_by_brightness_and_area(
    video_path: Path,
    gt_by_t: Dict[int, List[Tuple[int,int]]],
    *,
    crop_w: int,
    crop_h: int,
    bright_max_threshold: int,
    area_threshold_px: int,
    max_frames: Optional[int],
    area_min_pixel_brightness: int,
) -> Tuple[Dict[int, List[Tuple[int,int]]], int, int]:
    """Keep GT points whose crop meets brightness and area criteria."""
    from collections import defaultdict
    total = sum(len(v) for v in gt_by_t.values())
    if total == 0:
        return gt_by_t, 0, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage5_test] Could not open video for GT filtering: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    filtered: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
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
                _, bin_img = cv2.threshold(gray, int(area_min_pixel_brightness), 255, cv2.THRESH_BINARY)
                num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
                area = int(stats[1:, cv2.CC_STAT_AREA].max()) if (num > 1 and stats.shape[0] > 1) else 0
                if area >= int(area_threshold_px):
                    filtered[fr].append((int(x), int(y)))
                    kept += 1
        _progress(fr+1, limit, 'stage5-test-gt-filter'); fr += 1
    cap.release()
    return filtered, kept, total

def _dedupe_gt_via_distance_and_weight(
    video_path: Path,
    gt_by_t: Dict[int, List[Tuple[int,int]]],
    *,
    crop_w: int,
    crop_h: int,
    dist_threshold_px: float,
    max_frames: Optional[int],
) -> Tuple[Dict[int, List[Tuple[int,int]]], int]:
    """Deduplicate GT per frame by centroid distance; keep point with max BGR-sum crop weight."""
    from collections import defaultdict
    if not gt_by_t:
        return gt_by_t, 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage5_test] Could not open video for GT dedupe: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = total_frames if max_frames is None else min(total_frames, max_frames)
    def crop_weight(frame: np.ndarray, x: int, y: int) -> float:
        crop = _center_crop_clamped(frame, float(x), float(y), crop_w, crop_h)
        if crop.size == 0:
            return 0.0
        return float(crop.sum())
    def _dist2(a: Tuple[int,int], b: Tuple[int,int]) -> float:
        dx = float(a[0]) - float(b[0]); dy = float(a[1]) - float(b[1])
        return dx*dx + dy*dy
    thr2 = float(dist_threshold_px) * float(dist_threshold_px)
    deduped: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    removed = 0
    fr = 0
    while True:
        if fr >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        pts = gt_by_t.get(fr, [])
        n = len(pts)
        if n <= 1:
            if n == 1:
                x, y = pts[0]
                deduped[fr].append((int(x), int(y)))
            fr += 1
            continue
        weights = [crop_weight(frame, int(p[0]), int(p[1])) for p in pts]
        parent = list(range(n)); rank = [0]*n
        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
        for i in range(n):
            ai = pts[i]
            for j in range(i+1, n):
                if _dist2(ai, pts[j]) <= thr2:
                    union(i, j)
        comps: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            comps.setdefault(r, []).append(i)
        kept = 0
        for comp_idxs in comps.values():
            best_idx = max(comp_idxs, key=lambda k: weights[k])
            bx, by = pts[best_idx]
            deduped[fr].append((int(bx), int(by)))
            kept += 1
        removed += (n - kept)
        fr += 1
    cap.release()
    return deduped, removed


# ────────────────────── evaluation ──────────────────────
def _evaluate_frames(gt_by_t, preds_by_t, thr_px: float):
    from collections import defaultdict
    fps_by_t: Dict[int, List[Dict]] = defaultdict(list)
    tps_by_t: Dict[int, List[Dict]] = defaultdict(list)
    fns_by_t: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    all_dists: List[float] = []
    frames = sorted(set(gt_by_t.keys()) | set(preds_by_t.keys()))
    for t in frames:
        gts = gt_by_t.get(t, [])
        preds = preds_by_t.get(t, [])
        preds_xy = [(p['x'], p['y']) for p in preds]
        matches, unmatched_pred_idxs, unmatched_gt_idxs = _greedy_match_full(gts, preds_xy, thr_px)
        for gi, pi, d in matches:
            tps_by_t[t].append(preds[pi])
            all_dists.append(d)
        for pi in unmatched_pred_idxs:
            fps_by_t[t].append(preds[pi])
        for gi in unmatched_gt_idxs:
            gx, gy = gts[gi]
            fns_by_t[t].append((int(round(gx)), int(round(gy))))
    TP = sum(len(v) for v in tps_by_t.values())
    FP = sum(len(v) for v in fps_by_t.values())
    FN = sum(len(v) for v in fns_by_t.values())
    mean_err = (sum(all_dists)/len(all_dists)) if all_dists else 0.0
    return fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err)

def _thr_folder_name(thr: float) -> str:
    return f"thr_{float(thr):.1f}px"


# ────────────────────── output writers ──────────────────────
def _write_crops_and_csvs_for_threshold(
    video_path: Path,
    out_root: Path,
    thr: float,
    fps_by_t, tps_by_t, fns_by_t,
    crop_w: int, crop_h: int,
    max_frames: Optional[int],
):
    thr_dir = out_root / _thr_folder_name(thr)
    crops_dir_fp = thr_dir / "crops"
    crops_dir_tp = thr_dir / "tp_crops"
    crops_dir_fn = thr_dir / "fn_crops"
    for d in (thr_dir, crops_dir_fp, crops_dir_tp, crops_dir_fn):
        _ensure_dir(d)
    csv_fp = thr_dir / "fps.csv"
    csv_tp = thr_dir / "tps.csv"
    csv_fn = thr_dir / "fns.csv"
    with csv_fp.open('w', newline='') as f_fp, \
         csv_tp.open('w', newline='') as f_tp, \
         csv_fn.open('w', newline='') as f_fn:
        w_fp = csv.writer(f_fp); w_fp.writerow(['x','y','t','filepath','confidence'])
        w_tp = csv.writer(f_tp); w_tp.writerow(['x','y','t','filepath','confidence'])
        w_fn = csv.writer(f_fn); w_fn.writerow(['x','y','t','filepath','confidence'])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"[stage5_test] Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit = min(max_frames, total_frames) if max_frames is not None else total_frames
        fr = 0
        while True:
            if limit is not None and fr >= limit:
                break
            ok, frame = cap.read()
            if not ok:
                break
            for p in fps_by_t.get(fr, []):
                x = float(p['x']); y = float(p['y']); conf = float(p['conf'])
                crop = _center_crop_clamped(frame, x, y, crop_w, crop_h)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                if gray is not None and gray.size:
                    _, bin_img = cv2.threshold(gray, int(_NAMING_BIN_THR) - 1, 255, cv2.THRESH_BINARY)
                    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
                    area = int(stats[1:, cv2.CC_STAT_AREA].max()) if (num > 1 and stats.shape[0] > 1) else 0
                else:
                    area = 0
                fname = f"FP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_conf{conf:.4f}_max{max_val}_area{area}.png"
                outp = crops_dir_fp / fname
                cv2.imwrite(str(outp), crop)
                w_fp.writerow([int(round(x)), int(round(y)), fr, str(outp), f"{conf:.6f}"])
            for p in tps_by_t.get(fr, []):
                x = float(p['x']); y = float(p['y']); conf = float(p['conf'])
                crop = _center_crop_clamped(frame, x, y, crop_w, crop_h)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                if gray is not None and gray.size:
                    _, bin_img = cv2.threshold(gray, int(_NAMING_BIN_THR) - 1, 255, cv2.THRESH_BINARY)
                    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
                    area = int(stats[1:, cv2.CC_STAT_AREA].max()) if (num > 1 and stats.shape[0] > 1) else 0
                else:
                    area = 0
                fname = f"TP_t{fr:06d}_x{int(round(x))}_y{int(round(y))}_conf{conf:.4f}_max{max_val}_area{area}.png"
                outp = crops_dir_tp / fname
                cv2.imwrite(str(outp), crop)
                w_tp.writerow([int(round(x)), int(round(y)), fr, str(outp), f"{conf:.6f}"])
            for (gx, gy) in fns_by_t.get(fr, []):
                conf = float('nan')  # FN confidence optional; omitted here
                crop = _center_crop_clamped(frame, gx, gy, crop_w, crop_h)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                max_val = int(gray.max()) if (gray is not None and gray.size) else 0
                if gray is not None and gray.size:
                    _, bin_img = cv2.threshold(gray, int(_NAMING_BIN_THR) - 1, 255, cv2.THRESH_BINARY)
                    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
                    area = int(stats[1:, cv2.CC_STAT_AREA].max()) if (num > 1 and stats.shape[0] > 1) else 0
                else:
                    area = 0
                conf_for_name = 0.0 if conf != conf else conf
                fname = f"FN_t{fr:06d}_x{gx}_y{gy}_conf{conf_for_name:.4f}_max{max_val}_area{area}.png"
                outp = crops_dir_fn / fname
                cv2.imwrite(str(outp), crop)
                w_fn.writerow([gx, gy, fr, str(outp), ("" if conf!=conf else f"{conf:.6f}")])
            _progress(fr+1, limit, f"stage5-test-crops@{_thr_folder_name(thr)}"); fr += 1
        cap.release()


# ────────────────────── public API ──────────────────────
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
    gt_by_t, norm_gt_csv, gt_wh_by_t = _read_and_normalize_gt(gt_csv_path, gt_t_offset, out_dir, max_frames)
    # Refine GT centers using Stage-4 Gaussian centroid logic to match pipeline semantics
    pw = int(getattr(params, 'BOX_SIZE_PX', 40))
    ph = int(getattr(params, 'BOX_SIZE_PX', 40))
    sigma = float(getattr(params, 'GAUSS_SIGMA', 0.0))
    gt_by_t_refined = _refine_gt_by_gaussian_centroid(
        orig_video_path,
        gt_by_t,
        patch_w=pw,
        patch_h=ph,
        per_point_wh=gt_wh_by_t,
        sigma=sigma,
        max_frames=max_frames,
        save_dir=(out_dir / "gt_refined_crops"),
    )
    gt_by_t_dedup, removed_dups = _dedupe_gt_via_distance_and_weight(
        orig_video_path,
        gt_by_t_refined,
        crop_w=crop_w,
        crop_h=crop_h,
        dist_threshold_px=float(gt_dedupe_dist_threshold_px),
        max_frames=max_frames,
    )
    _write_norm_gt_from_map(norm_gt_csv, gt_by_t_dedup)
    final_kept = sum(len(v) for v in gt_by_t_dedup.values())
    print(f"[stage5_test] Normalized + refined (Gaussian) + deduped GT saved to: {norm_gt_csv}  "
          f"(removed duplicates: {removed_dups}; final: {final_kept})")

    preds_by_t = _read_predictions(pred_csv_path, only_firefly_rows, max_frames)
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
        fps_by_t, tps_by_t, fns_by_t, (TP, FP, FN, mean_err) = _evaluate_frames(gt_by_t_dedup, preds_by_t, float(thr))
        _write_crops_and_csvs_for_threshold(
            orig_video_path, out_dir, float(thr),
            fps_by_t, tps_by_t, fns_by_t,
            crop_w, crop_h, max_frames,
        )
        prec = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec  = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"Threshold: {float(thr):.1f}px")
        print(f"  TP: {TP}   FP: {FP}   FN: {FN}")
        print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
        print(f"  Mean error (px): {mean_err:.3f}\n")


__all__ = ['stage5_test_validate_against_gt']

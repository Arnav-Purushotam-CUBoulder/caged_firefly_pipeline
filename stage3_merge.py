"""
Stage 3: Merge detections by centroid distance and keep heaviest RGB sum.

Reads Stage2 CSVs (frame,cx,cy,size,pred,conf,firefly_logit,background_logit), keeps positives with
conf >= CONFIDENCE_MIN, groups detections per frame whose centroid
distance <= MERGE_DISTANCE_PX via union-find, then for each group keeps
the single detection with the highest RGB-sum weight computed on the
original frame inside a BOX_SIZE_PX square centered at the centroid.

Outputs Stage3 CSVs with: frame,cx,cy,x1,y1,x2,y2,conf,firefly_logit,background_logit
where (cx,cy) is the kept centroid; (x1,y1,x2,y2) is the clamped box; logits are
propagated from the corresponding Stage2 detection that wins the merge.
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import cv2
import numpy as np

import params

def _clamp_box_center(cx: float, cy: float, box_w: int, box_h: int, W: int, H: int) -> Tuple[int,int,int,int,int,int]:
    """Return clamped integer box corners and size (x1,y1,x2,y2,w,h) centered at (cx,cy)."""
    w = max(1, int(round(box_w)))
    h = max(1, int(round(box_h)))
    x0 = int(round(cx - w/2.0))
    y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    return x0, y0, x0 + w, y0 + h, w, h


def _rgb_weight(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return -1.0
    return float(crop.astype(np.float64).sum())


def _euclid(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx*dx + dy*dy) ** 0.5)


def _load_stage2(path: Path) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """Return mapping: frame -> list of (cx, cy, conf, firefly_logit, background_logit)
    for positive detections meeting the confidence threshold.
    """
    mapping: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        has_logits = ('firefly_logit' in cols and 'background_logit' in cols)
        for row in reader:
            pred = int(row['pred'])
            conf = float(row['conf'])
            if pred != 1 or conf < params.CONFIDENCE_MIN:
                continue
            fi = int(row['frame'])
            cx = float(row['cx']); cy = float(row['cy'])
            if has_logits:
                lf = float(row.get('firefly_logit', 'nan'))
                lb = float(row.get('background_logit', 'nan'))
            else:
                lf = float('nan'); lb = float('nan')
            mapping.setdefault(fi, []).append((cx, cy, conf, lf, lb))
    return mapping


def _merge_frame_by_distance(frame_bgr: np.ndarray, dets: List[Tuple[float,float,float,float,float]]):
    """Given a frame and list of (cx,cy,conf), return kept rows as
    (cx, cy, x1, y1, x2, y2, conf, firefly_logit, background_logit) using union-find by centroid distance
    and keeping max RGB-sum per group.
    """
    n = len(dets)
    if n == 0:
        return []
    H, W = frame_bgr.shape[:2]
    # Precompute boxes and weights
    boxes = []  # (x1,y1,x2,y2)
    cents = []  # (cx,cy)
    confs = []
    lfs = []
    lbs = []
    for cx, cy, conf, lf, lb in dets:
        x1, y1, x2, y2, w, h = _clamp_box_center(cx, cy, params.BOX_SIZE_PX, params.BOX_SIZE_PX, W, H)
        boxes.append((x1, y1, x2, y2))
        cents.append((cx, cy))
        confs.append(conf)
        lfs.append(lf)
        lbs.append(lb)
    # Union-find by centroid distance
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    th = float(params.MERGE_DISTANCE_PX)
    for i in range(n):
        for j in range(i+1, n):
            if _euclid(cents[i], cents[j]) <= th:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    results = []
    # Select max-weight within each group
    for root, idxs in groups.items():
        max_w = -1.0
        keep_i = idxs[0]
        for i in idxs:
            x1,y1,x2,y2 = boxes[i]
            wt = _rgb_weight(frame_bgr, x1, y1, x2, y2)
            if wt > max_w:
                max_w = wt
                keep_i = i
        cx, cy = cents[keep_i]
        x1,y1,x2,y2 = boxes[keep_i]
        conf = confs[keep_i]
        lf = lfs[keep_i]
        lb = lbs[keep_i]
        results.append((float(cx), float(cy), float(x1), float(y1), float(x2), float(y2), float(conf), float(lf), float(lb)))
    return results


def run_stage3() -> Path:
    out_dir = params.STAGE3_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}
    for csv_path in sorted(params.STAGE2_DIR.glob('*_classified.csv')):
        data = _load_stage2(csv_path)
        stem = csv_path.stem.replace('_classified', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"Warning: Stage3: no matching video for {csv_path.name}")
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: Stage3: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        out_csv = out_dir / f"{stem}_merged.csv"
        before = sum(len(v) for v in data.values())
        after = 0
        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'conf', 'firefly_logit', 'background_logit'])
            for idx in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                dets = data.get(idx, [])
                merged = _merge_frame_by_distance(frame, dets)
                after += len(merged)
                for cx, cy, x1, y1, x2, y2, conf, lf, lb in merged:
                    writer.writerow([
                        int(idx), float(cx), float(cy),
                        float(x1), float(y1), float(x2), float(y2), float(conf), float(lf), float(lb)
                    ])
        cap.release()
        print(f"Stage3: {stem} before={before} after={after} merged={before-after} dist_px={params.MERGE_DISTANCE_PX}")

    return out_dir


__all__ = ['run_stage3']

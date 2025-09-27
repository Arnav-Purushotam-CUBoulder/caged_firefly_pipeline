"""
Stage 3: Merge detections into boxes using IoU.

Reads Stage2 CSVs (frame,cx,cy,size,pred,conf), keeps positives with
conf >= CONFIDENCE_MIN, converts centroids to fixed-size boxes and
iteratively merges overlapping boxes per frame (IoU >= OVERLAP_IOU).

Outputs Stage3 CSVs with: frame,cx,cy,x1,y1,x2,y2,conf
where cx,cy are the merged centroid (average) and conf is max group conf.
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import numpy as np

import params


def _box_from_centroid(cx: float, cy: float) -> np.ndarray:
    half = params.BOX_SIZE_PX / 2.0
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=float)


def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return float(inter / (a1 + a2 - inter))


def _load_stage2(path: Path) -> Dict[int, List[Tuple[float, float, float]]]:
    """Return mapping: frame -> list of (cx, cy, conf) for positive detections."""
    mapping: Dict[int, List[Tuple[float, float, float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = int(row['pred'])
            conf = float(row['conf'])
            if pred != 1 or conf < params.CONFIDENCE_MIN:
                continue
            fi = int(row['frame'])
            cx = float(row['cx']); cy = float(row['cy'])
            mapping.setdefault(fi, []).append((cx, cy, conf))
    return mapping


def _merge_centroids(cents: List[Tuple[float, float, float]]):
    """
    Merge centroids by converting to fixed-size boxes and iteratively merging
    overlapping ones (IoU >= threshold). Returns list of tuples:
    (cx, cy, x1, y1, x2, y2, conf)
    """
    if not cents:
        return []
    boxes = [(_box_from_centroid(cx, cy), cx, cy, conf) for cx, cy, conf in cents]
    changed = True
    while changed:
        changed = False
        new_boxes = []
        i = 0
        while i < len(boxes):
            base_b, base_cx, base_cy, base_conf = boxes[i]
            group = [boxes[i]]
            j = i + 1
            while j < len(boxes):
                if _iou(base_b, boxes[j][0]) >= params.OVERLAP_IOU:
                    group.append(boxes.pop(j))
                    changed = True
                else:
                    j += 1
            xs = [c[1] for c in group]
            ys = [c[2] for c in group]
            confs = [c[3] for c in group]
            mcx = float(sum(xs) / len(xs))
            mcy = float(sum(ys) / len(ys))
            mb = _box_from_centroid(mcx, mcy)
            mconf = float(max(confs))
            new_boxes.append((mb, mcx, mcy, mconf))
            i += 1
        boxes = new_boxes
    # format output
    results = []
    for b, cx, cy, conf in boxes:
        x1, y1, x2, y2 = b.tolist()
        results.append((cx, cy, x1, y1, x2, y2, conf))
    return results


def run_stage3() -> Path:
    out_dir = params.STAGE3_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(params.STAGE2_DIR.glob('*_classified.csv')):
        data = _load_stage2(csv_path)
        stem = csv_path.stem.replace('_classified', '')
        out_csv = out_dir / f"{stem}_merged.csv"
        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'conf'])
            for frame_idx in sorted(data.keys()):
                merged = _merge_centroids(data[frame_idx])
                for cx, cy, x1, y1, x2, y2, conf in merged:
                    writer.writerow([
                        int(frame_idx), float(cx), float(cy),
                        float(x1), float(y1), float(x2), float(y2), float(conf)
                    ])

    return out_dir


__all__ = ['run_stage3']

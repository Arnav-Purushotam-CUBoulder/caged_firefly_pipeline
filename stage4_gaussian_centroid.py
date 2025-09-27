"""
Stage 4: Gaussian intensity centroid refinement with crops.

Reads Stage3 merged CSVs (frame,cx,cy,x1,y1,x2,y2,conf), refines the
centroid within a small patch using an optional Gaussian-weighted
intensity moment, and writes per-video CSVs with center semantics and
fixed patch size, plus saves crops for visual inspection.

Outputs (per video in STAGE4_DIR):
  - <stem>_gauss.csv  with columns: frame,x,y,w,h,conf,xy_semantics='center'
  - crops/ directory containing small crops with center marked
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import cv2
import numpy as np

import params


def _center_crop_clamped(img, cx: float, cy: float, w: int, h: int):
    H, W = img.shape[:2]
    w = max(1, int(round(w)))
    h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0))
    y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w))
    y0 = max(0, min(y0, H - h))
    return img[y0:y0+h, x0:x0+w], x0, y0


def _gaussian_kernel(w: int, h: int, sigma: float):
    if sigma <= 0:
        k = np.ones((h, w), dtype=np.float32)
        return k / float(h * w)
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((x - xc)**2 + (y - yc)**2) / (2.0 * sigma**2)).astype(np.float32)
    s = float(g.sum())
    if s > 0:
        g /= s
    return g


def _intensity_centroid(img_gray: np.ndarray, gaussian_sigma: float = 0.0):
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


def _load_stage3(path: Path) -> Dict[int, List[Tuple[float,float,float,float,float]]]:
    """Return mapping frame -> list of (cx,cy,x1,y1,x2,y2,conf), but we keep
    only (cx,cy,conf) and compute refined center later."""
    mapping: Dict[int, List[Tuple[float,float,float,float,float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            cx = float(row['cx']); cy = float(row['cy'])
            conf = float(row.get('conf', 'nan'))
            mapping.setdefault(fi, []).append((cx, cy, conf, 0.0, 0.0))
    return mapping


def run_stage4() -> Path:
    out_dir = params.STAGE4_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    # Use unified BOX_SIZE_PX for both dimensions
    pw = int(getattr(params, 'BOX_SIZE_PX', 40))
    ph = int(getattr(params, 'BOX_SIZE_PX', 40))
    sigma = float(getattr(params, 'GAUSS_SIGMA', 0.0))

    for csv_path in sorted(params.STAGE3_DIR.glob('*_merged.csv')):
        stem = csv_path.stem.replace('_merged', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"Warning: Stage4: no matching video for {csv_path.name}")
            continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: Stage4: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total

        data = _load_stage3(csv_path)
        out_csv = out_dir / f"{stem}_gauss.csv"
        crops_dir = (out_dir / f"{stem}_crops").resolve()
        crops_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        shifts = []

        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'x', 'y', 'w', 'h', 'conf', 'xy_semantics'])
            for idx in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                dets = data.get(idx, [])
                if not dets:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for cx, cy, conf, _, _ in dets:
                    crop, x0, y0 = _center_crop_clamped(gray, cx, cy, pw, ph)
                    if crop.size == 0:
                        continue
                    ccx, ccy = _intensity_centroid(crop, sigma)
                    new_cx = x0 + ccx
                    new_cy = y0 + ccy
                    writer.writerow([int(idx), float(new_cx), float(new_cy), int(pw), int(ph), float(conf), 'center'])
                    processed += 1
                    # shift magnitude
                    dx = float(new_cx - cx)
                    dy = float(new_cy - cy)
                    shifts.append((dx*dx + dy*dy) ** 0.5)
                    # save color crop with center marked
                    crop_bgr, x0c, y0c = _center_crop_clamped(frame, new_cx, new_cy, pw, ph)
                    px = int(round(new_cx - x0c)); py = int(round(new_cy - y0c))
                    if 0 <= py < crop_bgr.shape[0] and 0 <= px < crop_bgr.shape[1]:
                        crop_bgr[py, px] = (0, 0, 255)
                    out_name = f"stage4_frame_{idx:06d}_x{int(round(new_cx))}_y{int(round(new_cy))}_{pw}x{ph}_conf{conf:.4f}.png"
                    cv2.imwrite(str(crops_dir / out_name), crop_bgr)
        cap.release()
        mean_shift = float(np.mean(shifts)) if shifts else 0.0
        print(f"Stage4: {stem} refined={processed} mean_shift_px={mean_shift:.2f} patch={pw}x{ph} sigma={sigma}")

    return out_dir


__all__ = ['run_stage4']

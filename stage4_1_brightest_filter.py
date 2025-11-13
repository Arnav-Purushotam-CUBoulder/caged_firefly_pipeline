"""
Stage 4.1: Brightest-pixel filter on Stage 4 outputs.

Reads Stage4 Gaussian CSVs (x,y,t,firefly_logit,background_logit) and drops
any candidate whose centered patch has max luminance below a threshold.

Luminance formula matches the annotation tool's hover readout:
  intensity = round(0.299*R + 0.587*G + 0.114*B)

Outputs per-video CSV into STAGE4_1_DIR with the same schema as Stage4.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import csv

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
    return img[y0:y0+h, x0:x0+w]


def _read_stage4(path: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """Return mapping frame -> list of (x,y,firefly_logit,background_logit)."""
    by_t: Dict[int, List[Tuple[float, float, float, float]]] = {}
    with path.open('r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(row.get('t', row.get('frame', 0)))
                x = float(row['x']); y = float(row['y'])
                lf = float(row.get('firefly_logit', 'nan'))
                lb = float(row.get('background_logit', 'nan'))
                by_t.setdefault(t, []).append((x, y, lf, lb))
            except Exception:
                continue
    return by_t


def run_stage4_1() -> Path:
    out_root = params.STAGE4_1_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    # Map video stems to paths
    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    # Iterate Stage4 outputs
    stage4_csvs = sorted(params.STAGE4_DIR.glob('*_gauss.csv'))
    if not stage4_csvs:
        print('[stage4.1] No Stage4 CSVs found; nothing to filter.')
        return out_root

    thr = float(getattr(params, 'STAGE4_1_BRIGHT_MAX_THRESHOLD', 50))
    box = int(getattr(params, 'BOX_SIZE_PX', 40))

    for csv_path in stage4_csvs:
        stem = csv_path.stem.replace('_gauss', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"[stage4.1] Warning: no matching video for {csv_path.name}")
            continue
        by_t = _read_stage4(csv_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[stage4.1] Warning: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit = min(total, params.N) if params.N else total

        kept = 0
        dropped = 0
        # Per-video output structure
        vid_dir = out_root / stem
        kept_dir = vid_dir / "kept_crops"
        drop_dir = vid_dir / "dropped_crops"
        for d in (vid_dir, kept_dir, drop_dir):
            d.mkdir(parents=True, exist_ok=True)

        out_csv = vid_dir / f"{stem}_gauss_bright_filtered.csv"
        with out_csv.open('w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['x', 'y', 't', 'firefly_logit', 'background_logit'])
            for t in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                # Convert to RGB for luminance calculation matching the tool
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets = by_t.get(t, [])
                for (x, y, lf, lb) in dets:
                    # Compute clamped crop bounds once
                    H, W = frame_rgb.shape[:2]
                    wbox = box; hbox = box
                    x0 = int(round(x - wbox/2.0))
                    y0 = int(round(y - hbox/2.0))
                    x0 = max(0, min(x0, W - wbox))
                    y0 = max(0, min(y0, H - hbox))
                    x1 = x0 + wbox; y1 = y0 + hbox
                    crop_rgb = frame_rgb[y0:y1, x0:x1]
                    if crop_rgb.size == 0:
                        dropped += 1
                        continue
                    # Luminance = 0.299 R + 0.587 G + 0.114 B
                    lumin = (0.299 * crop_rgb[:, :, 0] + 0.587 * crop_rgb[:, :, 1] + 0.114 * crop_rgb[:, :, 2])
                    max_val = float(lumin.max()) if lumin.size else 0.0
                    # For naming: count bright pixels using Stage 4.2 threshold
                    crop_bgr = frame[y0:y1, x0:x1]
                    thr_bright = int(getattr(params, 'STAGE4_2_INTENSITY_THR', 200))
                    nbright = int((lumin >= thr_bright).sum()) if lumin.size else 0
                    # Save BGR crops for visualization (already computed as crop_bgr)
                    # add softmax confidence from logits (if present)
                    conf_error = False
                    try:
                        m = max(lf, lb)
                        conf = float(np.exp(lf - m) / (np.exp(lf - m) + np.exp(lb - m)))
                        if not np.isfinite(conf):
                            conf_error = True
                    except Exception:
                        conf_error = True
                        conf = float('nan')
                    conf_str = ("error" if conf_error else f"{conf:.4f}")
                    fname = (
                        f"t{t:06d}_x{int(round(x))}_y{int(round(y))}_"
                        f"{wbox}x{hbox}_conf{conf_str}_max{int(round(max_val))}_brightpx{nbright}.png"
                    )
                    if max_val < thr:
                        dropped += 1
                        cv2.imwrite(str(drop_dir / fname), crop_bgr)
                        continue
                    kept += 1
                    cv2.imwrite(str(kept_dir / fname), crop_bgr)
                    w.writerow([float(x), float(y), int(t), float(lf), float(lb)])
        cap.release()
        print(f"[stage4.1] {stem}: kept={kept} dropped={dropped} thr={thr}")

    return out_root


__all__ = ['run_stage4_1']

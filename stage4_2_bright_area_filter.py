"""
Stage 4.2: Bright-area pixels filter on Stage 4/4.1 outputs.

Counts the number of pixels above a luminance threshold inside the centered
patch for each candidate. If the count is below a minimum, the candidate is
discarded. Luminance uses the same formula as the annotator and Stage 4.1:
  L = 0.299*R + 0.587*G + 0.114*B

Inputs: prefers Stage 4.1 filtered CSVs when present, otherwise Stage 4 CSVs.
Outputs: per-video CSV and crops for kept and dropped candidates.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import cv2
import numpy as np

import params


def _read_stage_csv(path: Path) -> Dict[int, List[Tuple[float, float, float, float]]]:
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


def _candidate_csv_for_stem(stem: str) -> Path | None:
    # Prefer Stage 4.1 filtered CSV (inside per-video folder) when present
    cand_41 = params.STAGE4_1_DIR / stem / f"{stem}_gauss_bright_filtered.csv"
    if cand_41.exists():
        return cand_41
    # Fall back to Stage 4 CSV
    cand_4 = params.STAGE4_DIR / f"{stem}_gauss.csv"
    return cand_4 if cand_4.exists() else None


def run_stage4_2() -> Path:
    out_root = params.STAGE4_2_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    box = int(getattr(params, 'BOX_SIZE_PX', 40))
    thr = float(getattr(params, 'STAGE4_2_INTENSITY_THR', 50))
    min_pix = int(getattr(params, 'STAGE4_2_MIN_BRIGHT_PIXELS', 6))

    # Iterate candidates per video
    for vstem, vpath in vid_map.items():
        cand_csv = _candidate_csv_for_stem(vstem)
        if cand_csv is None:
            continue
        by_t = _read_stage_csv(cand_csv)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[stage4.2] Warning: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        limit = min(total, params.N) if params.N else total

        kept = 0
        dropped = 0
        vid_dir = out_root / vstem
        kept_dir = vid_dir / "kept_crops"
        drop_dir = vid_dir / "dropped_crops"
        for d in (vid_dir, kept_dir, drop_dir):
            d.mkdir(parents=True, exist_ok=True)
        out_csv = vid_dir / f"{vstem}_gauss_brightarea_filtered.csv"
        with out_csv.open('w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['x', 'y', 't', 'firefly_logit', 'background_logit'])
            for t in range(limit):
                ok, frame = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets = by_t.get(t, [])
                for (x, y, lf, lb) in dets:
                    H, W = frame_rgb.shape[:2]
                    wbox = box; hbox = box
                    x0 = int(round(x - wbox/2.0))
                    y0 = int(round(y - hbox/2.0))
                    x0 = max(0, min(x0, W - wbox))
                    y0 = max(0, min(y0, H - hbox))
                    x1 = x0 + wbox; y1 = y0 + hbox
                    crop_rgb = frame_rgb[y0:y1, x0:x1]
                    crop_bgr = frame[y0:y1, x0:x1]
                    if crop_rgb.size == 0:
                        dropped += 1
                        continue
                    # Luminance and bright-pixel count
                    lumin = (0.299 * crop_rgb[:, :, 0] + 0.587 * crop_rgb[:, :, 1] + 0.114 * crop_rgb[:, :, 2])
                    max_val = float(lumin.max()) if lumin.size else 0.0
                    nbright = int((lumin >= thr).sum())
                    # also compute softmax confidence from logits
                    try:
                        m = max(lf, lb)
                        conf = float(np.exp(lf - m) / (np.exp(lf - m) + np.exp(lb - m)))
                    except Exception:
                        conf = float('nan')
                    fname = (f"t{t:06d}_x{int(round(x))}_y{int(round(y))}_"
                             f"{wbox}x{hbox}_conf{0.0 if conf!=conf else conf:.4f}_max{int(round(max_val))}_brightpx{nbright}.png")
                    if nbright < min_pix:
                        dropped += 1
                        cv2.imwrite(str(drop_dir / fname), crop_bgr)
                        continue
                    kept += 1
                    cv2.imwrite(str(kept_dir / fname), crop_bgr)
                    w.writerow([float(x), float(y), int(t), float(lf), float(lb)])
        cap.release()
        print(f"[stage4.2] {vstem}: kept={kept} dropped={dropped} thr={thr} min_pix={min_pix}")

    return out_root


__all__ = ['run_stage4_2']


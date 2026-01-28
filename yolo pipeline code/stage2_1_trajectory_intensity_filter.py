#!/usr/bin/env python3
"""
Stage 2.1: trajectory + intensity "hill" filter on Stage 2 detections.

Goal: reduce noisy per-frame detections by keeping only those that form a
short spatiotemporal trajectory whose intensity over time looks like a
single flash (rise then fall).

Input:
  STAGE2_DIR/<video_stem>/<video_stem>_bright.csv

Outputs:
  STAGE2_DIR/<video_stem>/<video_stem>_bright_traj.csv       (selected only; Stage2 schema)
  STAGE2_DIR/<video_stem>/<video_stem>_bright_traj_all.csv   (all dets + trajectory metadata)

Notes:
  - This stage does not depend on any code from other pipelines; it is
    implemented entirely within the caged_fireflies YOLO pipeline.
  - Stage 3 can be configured to read the *_bright_traj.csv file when
    STAGE2_1_ENABLE=True in params.py.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import params

import cv2
import numpy as np


@dataclass(frozen=True)
class _Det:
    row_idx: int
    frame: int
    cx: float
    cy: float


@dataclass
class _Track:
    id: int
    dets: List[_Det]

    @property
    def last_frame(self) -> int:
        return int(self.dets[-1].frame)

    @property
    def last_cx(self) -> float:
        return float(self.dets[-1].cx)

    @property
    def last_cy(self) -> float:
        return float(self.dets[-1].cy)


def _read_stage2_rows(stem: str) -> Tuple[Path, List[dict]]:
    s2_csv = (params.STAGE2_DIR / stem) / f"{stem}_bright.csv"
    if not s2_csv.exists():
        raise FileNotFoundError(f"Missing Stage2 CSV for {stem}: {s2_csv}")
    rows: List[dict] = []
    with s2_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return s2_csv, rows


def _group_dets_by_frame(dets: List[_Det]) -> Dict[int, List[_Det]]:
    by_t: Dict[int, List[_Det]] = defaultdict(list)
    for d in dets:
        by_t[int(d.frame)].append(d)
    return by_t


def _link_tracks(
    by_t: Dict[int, List[_Det]],
    *,
    link_radius_px: float,
    max_frame_gap: int,
    time_scale: float,
) -> List[_Track]:
    active: Dict[int, _Track] = {}
    finished: List[_Track] = []
    next_id = 0

    def finalize_stale(cur_t: int) -> None:
        stale = [tid for tid, tr in active.items() if (cur_t - tr.last_frame) > max_frame_gap]
        for tid in stale:
            finished.append(active.pop(tid))

    for t in sorted(by_t.keys()):
        t_int = int(t)
        finalize_stale(t_int)

        dets = by_t.get(t_int, [])
        if not dets:
            continue

        # Greedy assignment between active tracks and current-frame detections.
        candidates: List[Tuple[float, int, int]] = []
        active_items = list(active.items())
        for det_i, d in enumerate(dets):
            for tid, tr in active_items:
                gap = int(d.frame - tr.last_frame)
                if gap <= 0 or gap > max_frame_gap:
                    continue
                dx = float(d.cx - tr.last_cx)
                dy = float(d.cy - tr.last_cy)
                dist_xy = float(math.hypot(dx, dy))
                max_dist = float(link_radius_px) * max(1.0, float(gap))
                if dist_xy > max_dist:
                    continue
                # Prefer small spatial distance, with a light gap penalty.
                candidates.append((dist_xy + float(time_scale) * float(gap), tid, det_i))

        candidates.sort(key=lambda x: x[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for _, tid, det_i in candidates:
            if tid in used_tracks or det_i in used_dets:
                continue
            active[tid].dets.append(dets[det_i])
            used_tracks.add(tid)
            used_dets.add(det_i)

        for det_i, d in enumerate(dets):
            if det_i in used_dets:
                continue
            active[next_id] = _Track(id=next_id, dets=[d])
            next_id += 1

    finished.extend(active.values())
    return finished


def _track_motion_xy(dets: List[_Det]) -> float:
    if len(dets) < 2:
        return 0.0
    motion = 0.0
    prev = dets[0]
    for cur in dets[1:]:
        motion += float(math.hypot(float(cur.cx - prev.cx), float(cur.cy - prev.cy)))
        prev = cur
    return float(motion)


def _moving_average(vals: List[float], win: int) -> List[float]:
    if win <= 1 or len(vals) <= 2:
        return [float(v) for v in vals]
    w = max(1, int(win))
    half = w // 2
    out: List[float] = []
    n = len(vals)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        seg = vals[a:b]
        out.append(float(sum(seg)) / float(len(seg)))
    return out


def _is_hill_like_curve(ys: List[float]) -> bool:
    """Heuristic: rises then falls with a single main peak."""
    if len(ys) < 3:
        return False

    min_range = float(getattr(params, "STAGE2_1_INTENSITY_MIN_RANGE", 0.0))
    if (max(ys) - min(ys)) < min_range:
        return False

    win = int(getattr(params, "STAGE2_1_HILL_SMOOTH_WINDOW", 3))
    y = _moving_average(ys, win)

    peak_i = int(max(range(len(y)), key=lambda i: y[i]))
    n = len(y)
    pmin = float(getattr(params, "STAGE2_1_HILL_PEAK_POS_MIN_FRAC", 0.10))
    pmax = float(getattr(params, "STAGE2_1_HILL_PEAK_POS_MAX_FRAC", 0.90))
    if peak_i <= int(math.floor(pmin * (n - 1))) or peak_i >= int(math.ceil(pmax * (n - 1))):
        return False

    diffs = [y[i + 1] - y[i] for i in range(n - 1)]
    eps = 1e-6
    signs: List[int] = []
    for d in diffs:
        if abs(float(d)) <= eps:
            signs.append(0)
        elif d > 0:
            signs.append(1)
        else:
            signs.append(-1)

    before = [s for s in signs[:peak_i] if s != 0]
    after = [s for s in signs[peak_i:] if s != 0]
    if not before or not after:
        return False

    min_up = int(getattr(params, "STAGE2_1_HILL_MIN_UP_STEPS", 2))
    min_down = int(getattr(params, "STAGE2_1_HILL_MIN_DOWN_STEPS", 2))
    up_steps = sum(1 for s in before if s > 0)
    down_steps = sum(1 for s in after if s < 0)
    if up_steps < min_up or down_steps < min_down:
        return False

    frac = float(getattr(params, "STAGE2_1_HILL_MIN_MONOTONIC_FRAC", 0.60))
    if (up_steps / max(1, len(before))) < frac:
        return False
    if (down_steps / max(1, len(after))) < frac:
        return False

    return True


def _patch_intensity(gray: np.ndarray, cx: float, cy: float, patch_size: int, metric: str) -> float:
    H, W = gray.shape[:2]
    ps = max(1, int(patch_size))
    half = ps // 2

    xc = int(round(float(cx)))
    yc = int(round(float(cy)))
    x0 = xc - half
    y0 = yc - half
    x1 = x0 + ps
    y1 = y0 + ps

    x0 = max(0, min(int(x0), W - 1))
    y0 = max(0, min(int(y0), H - 1))
    x1 = max(0, min(int(x1), W))
    y1 = max(0, min(int(y1), H))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0

    m = str(metric or "sum").strip().lower()
    if m == "max":
        return float(patch.max())
    if m == "mean":
        return float(patch.mean())
    # default: sum
    return float(patch.sum())


def _compute_intensities_for_dets(
    video_path: Path,
    dets_by_t: Dict[int, List[_Det]],
) -> Dict[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = getattr(params, "MAX_FRAMES", None)
    if max_frames is not None:
        limit = min(total, int(max_frames))
    else:
        limit = total

    ps = int(getattr(params, "STAGE2_1_INTENSITY_PATCH_SIZE", 10))
    metric = str(getattr(params, "STAGE2_1_INTENSITY_METRIC", "sum"))

    intensity_by_row: Dict[int, float] = {}
    t = 0
    try:
        while t < limit:
            ok, frame = cap.read()
            if not ok:
                break
            dets = dets_by_t.get(int(t))
            if not dets:
                t += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for d in dets:
                intensity_by_row[int(d.row_idx)] = _patch_intensity(
                    gray, float(d.cx), float(d.cy), ps, metric
                )
            t += 1
    finally:
        cap.release()
    return intensity_by_row


def run_for_video(video_path: Path) -> Path:
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    in_csv, rows = _read_stage2_rows(stem)
    out_root = in_csv.parent
    out_selected = out_root / f"{stem}_bright_traj.csv"
    out_all = out_root / f"{stem}_bright_traj_all.csv"

    base_fields = list(rows[0].keys()) if rows else [
        "x",
        "y",
        "w",
        "h",
        "frame",
        "video_name",
        "firefly_logit",
        "background_logit",
    ]

    if not rows:
        for p in [out_selected, out_all]:
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=base_fields)
                w.writeheader()
        print(f"Stage2.1 NOTE: no Stage2 detections for {video_path.name}; wrote empty → {out_selected.name}")
        return out_selected

    dets: List[_Det] = []
    for i, row in enumerate(rows):
        try:
            t = int(float(row.get("frame", 0)))
            x = float(row["x"])
            y = float(row["y"])
            w_box = float(row["w"])
            h_box = float(row["h"])
        except Exception:
            continue
        cx = float(x + 0.5 * w_box)
        cy = float(y + 0.5 * h_box)
        dets.append(_Det(row_idx=int(i), frame=int(t), cx=cx, cy=cy))

    if not dets:
        for p in [out_selected, out_all]:
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=base_fields)
                w.writeheader()
        print(f"Stage2.1 NOTE: Stage2 CSV had no parsable rows for {video_path.name}; wrote empty → {out_selected.name}")
        return out_selected

    by_t = _group_dets_by_frame(dets)
    intensity_by_row = _compute_intensities_for_dets(video_path, by_t)

    link_radius_px = float(getattr(params, "STAGE2_1_LINK_RADIUS_PX", 12.0))
    max_gap = int(getattr(params, "STAGE2_1_MAX_FRAME_GAP", 3))
    time_scale = float(getattr(params, "STAGE2_1_TIME_SCALE", 1.0))
    min_pts = int(getattr(params, "STAGE2_1_MIN_TRACK_POINTS", 3))
    require_hill = bool(getattr(params, "STAGE2_1_REQUIRE_HILL_SHAPE", True))
    min_range = float(getattr(params, "STAGE2_1_INTENSITY_MIN_RANGE", 0.0))

    tracks = _link_tracks(by_t, link_radius_px=link_radius_px, max_frame_gap=max_gap, time_scale=time_scale)

    det_meta: Dict[int, Tuple[int, int, float, float, int, float]] = {}
    selected_tracks = 0
    selected_dets = 0

    for tr in tracks:
        dets_sorted = sorted(tr.dets, key=lambda d: (int(d.frame), int(d.row_idx)))
        ys: List[float] = []
        for d in dets_sorted:
            v = intensity_by_row.get(int(d.row_idx))
            if v is None:
                continue
            ys.append(float(v))

        intensity_range = float(max(ys) - min(ys)) if ys else 0.0
        hill_ok = (not require_hill) or _is_hill_like_curve(ys)
        is_selected = int((len(ys) >= min_pts) and (intensity_range >= min_range) and hill_ok)

        if is_selected:
            selected_tracks += 1
            selected_dets += len(dets_sorted)

        motion_xy = float(_track_motion_xy(dets_sorted))
        for d in dets_sorted:
            det_meta[int(d.row_idx)] = (
                int(tr.id),
                int(len(dets_sorted)),
                float(motion_xy),
                float(intensity_range),
                int(is_selected),
                float(intensity_by_row.get(int(d.row_idx), 0.0)),
            )

    # Write all CSV (includes metadata and per-detection intensity)
    all_fields = list(base_fields)
    extras = [
        "traj_id",
        "traj_size",
        "traj_motion_xy",
        "traj_intensity_range",
        "traj_is_selected",
        "traj_intensity_value",
    ]
    for e in extras:
        if e not in all_fields:
            all_fields.append(e)

    with out_all.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            meta = det_meta.get(int(i))
            if meta is None:
                continue
            traj_id, traj_size, motion_xy, intensity_range, is_selected, intensity_val = meta
            out_row = dict(row)
            out_row["traj_id"] = int(traj_id)
            out_row["traj_size"] = int(traj_size)
            out_row["traj_motion_xy"] = float(motion_xy)
            out_row["traj_intensity_range"] = float(intensity_range)
            out_row["traj_is_selected"] = int(is_selected)
            out_row["traj_intensity_value"] = float(intensity_val)
            w.writerow(out_row)

    # Write selected-only CSV (strict Stage2 schema to keep Stage3 compatible)
    with out_selected.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            meta = det_meta.get(int(i))
            if meta is None:
                continue
            if int(meta[4]) != 1:
                continue
            out_row = {k: row.get(k, "") for k in base_fields}
            w.writerow(out_row)

    print(
        "Stage2.1 Trajectory intensity filter:",
        f"dets_in={len(dets)}",
        f"tracks={len(tracks)}",
        f"selected_tracks={selected_tracks}",
        f"selected_dets={selected_dets}",
        f"(min_pts={min_pts}, min_range={min_range}, require_hill={require_hill}, link_r={link_radius_px}, gap={max_gap}, t_scale={time_scale})",
    )
    print(f"Stage2.1 Wrote selected CSV → {out_selected}")
    print(f"Stage2.1 Wrote all CSV → {out_all}")

    return out_selected


__all__ = ["run_for_video"]

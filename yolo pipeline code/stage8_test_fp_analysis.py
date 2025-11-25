#!/usr/bin/env python3
"""
Stage 8 (test) — Analyze false positives (FPs) for the YOLO pipeline.

For each threshold directory thr_* under a per-video Stage 5 output folder,
compute the nearest GT (TP or FN) to each FP and write a CSV:
  thr_*/fp_nearest_tp.csv

No images are rendered here; this stage is purely numeric.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import math


def _progress(i: int, total: int, tag: str = "") -> None:
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar = "=" * int(frac * 50) + " " * (50 - int(frac * 50))
    print(f"\r{tag} [{bar}] {int(frac*100):3d}%", end="" if i < total else "\n")


def _read_points_by_frame(csv_path: Path) -> Dict[int, List[Tuple[float, float]]]:
    by_t: Dict[int, List[Tuple[float, float]]] = {}
    if not csv_path.exists():
        return by_t
    with csv_path.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                t = int(float(row.get("t", row.get("frame", "0"))))
                x = float(row["x"])
                y = float(row["y"])
            except Exception:
                continue
            by_t.setdefault(t, []).append((x, y))
    return by_t


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _analyze_threshold_dir(thr_dir: Path) -> Tuple[int, int]:
    """Compute nearest GT (TP or FN) for each FP under a given thr_* directory."""
    fps_csv = thr_dir / "fps.csv"
    tps_csv = thr_dir / "tps.csv"
    fns_csv = thr_dir / "fns.csv"
    out_csv = thr_dir / "fp_nearest_tp.csv"

    fps_by_t = _read_points_by_frame(fps_csv)
    tps_by_t = _read_points_by_frame(tps_csv)
    fns_by_t = _read_points_by_frame(fns_csv)

    frames = sorted(set(fps_by_t.keys()))
    total_fps = sum(len(v) for v in fps_by_t.values())
    no_gt_cnt = 0

    rows_out: List[Dict[str, object]] = []
    for t in frames:
        fps = fps_by_t.get(t, [])
        gts = (tps_by_t.get(t, []) or []) + (fns_by_t.get(t, []) or [])
        for (fx, fy) in fps:
            if not gts:
                rows_out.append(
                    {
                        "t": t,
                        "fp_x": fx,
                        "fp_y": fy,
                        "gt_x": None,
                        "gt_y": None,
                        "dist": None,
                    }
                )
                no_gt_cnt += 1
                continue
            min_d = None
            min_gt = None
            for (gx, gy) in gts:
                d = _euclid((fx, fy), (gx, gy))
                if min_d is None or d < min_d:
                    min_d = d
                    min_gt = (gx, gy)
            rows_out.append(
                {
                    "t": t,
                    "fp_x": fx,
                    "fp_y": fy,
                    "gt_x": float(min_gt[0]),
                    "gt_y": float(min_gt[1]),
                    "dist": float(min_d),
                }
            )
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t", "fp_x", "fp_y", "gt_x", "gt_y", "dist"]
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fps, no_gt_cnt


def stage8_test_fp_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    verbose: bool = True,
) -> None:
    """Run FP nearest-GT analysis for all thresholds for one video."""
    stage9_video_dir = Path(stage9_video_dir)
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage8_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return

    thr_dirs = sorted(
        [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")]
    )
    if not thr_dirs:
        if verbose:
            print(f"[stage8_test] No thr_* directories found in: {stage9_video_dir}")
        return

    for i, thr_dir in enumerate(thr_dirs, start=1):
        total_fps, no_gt = _analyze_threshold_dir(thr_dir)
        if verbose:
            print(
                f"[stage8_test] {stage9_video_dir.name} / {thr_dir.name}: "
                f"FPs={total_fps}, FPs_with_no_GT={no_gt}"
            )
        _progress(i, len(thr_dirs), "stage8-test-fp")


__all__ = ["stage8_test_fp_nearest_tp_analysis"]


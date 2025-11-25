#!/usr/bin/env python3
"""
Stage 7 (test) — Analyze false negatives (FNs) for the YOLO pipeline.

For each threshold directory thr_* under a per-video Stage 5 output folder,
compute the nearest TP to each FN and write a CSV:
  thr_*/fn_nearest_tp.csv

No images are rendered here; this stage is purely numeric.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


def _analyze_threshold_dir(
    thr_dir: Path,
) -> Tuple[int, int]:
    """Compute nearest TP for each FN under a given thr_* directory."""
    fns_csv = thr_dir / "fns.csv"
    tps_csv = thr_dir / "tps.csv"
    out_csv = thr_dir / "fn_nearest_tp.csv"

    fns_by_t = _read_points_by_frame(fns_csv)
    tps_by_t = _read_points_by_frame(tps_csv)

    frames = sorted(set(fns_by_t.keys()))
    total_fns = sum(len(v) for v in fns_by_t.values())
    no_tp_cnt = 0

    rows_out: List[Dict[str, object]] = []
    for t in frames:
        fns = fns_by_t.get(t, [])
        tps = tps_by_t.get(t, [])
        for (fx, fy) in fns:
            if not tps:
                rows_out.append(
                    {
                        "t": t,
                        "fn_x": fx,
                        "fn_y": fy,
                        "tp_x": None,
                        "tp_y": None,
                        "dist": None,
                    }
                )
                no_tp_cnt += 1
                continue
            min_d = None
            min_tp = None
            for (tx, ty) in tps:
                d = _euclid((fx, fy), (tx, ty))
                if min_d is None or d < min_d:
                    min_d = d
                    min_tp = (tx, ty)
            rows_out.append(
                {
                    "t": t,
                    "fn_x": fx,
                    "fn_y": fy,
                    "tp_x": float(min_tp[0]),
                    "tp_y": float(min_tp[1]),
                    "dist": float(min_d),
                }
            )
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t", "fn_x", "fn_y", "tp_x", "tp_y", "dist"]
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    return total_fns, no_tp_cnt


def stage7_test_fn_nearest_tp_analysis(
    *,
    stage9_video_dir: Path,
    orig_video_path: Path,
    box_w: int,
    box_h: int,
    thickness: int,
    verbose: bool = True,
) -> None:
    """Run FN nearest-TP analysis for all thresholds for one video."""
    stage9_video_dir = Path(stage9_video_dir)
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage7_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return

    thr_dirs = sorted(
        [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")]
    )
    if not thr_dirs:
        if verbose:
            print(f"[stage7_test] No thr_* directories found in: {stage9_video_dir}")
        return

    for i, thr_dir in enumerate(thr_dirs, start=1):
        total_fns, no_tp = _analyze_threshold_dir(thr_dir)
        if verbose:
            print(
                f"[stage7_test] {stage9_video_dir.name} / {thr_dir.name}: "
                f"FNs={total_fns}, FNs_with_no_TP={no_tp}"
            )
        _progress(i, len(thr_dirs), "stage7-test-fn")


__all__ = ["stage7_test_fn_nearest_tp_analysis"]


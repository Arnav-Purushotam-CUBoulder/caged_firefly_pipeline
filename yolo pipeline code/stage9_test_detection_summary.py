#!/usr/bin/env python3
"""
Stage 9 (test) — Detection summary for YOLO pipeline.

Reads Stage 5 per-threshold TP/FP/FN CSVs and the nearest-TP analyses from
Stages 7 and 8, then writes a JSON summary with counts and basic metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import csv
import json
from datetime import datetime, timezone


def _parse_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _load_detection_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _load_nearest_map(
    csv_path: Path, kind: str
) -> Dict[tuple[int, int, int], Dict[str, Optional[float]]]:
    """Load nearest-TP/FN info keyed by (t,x,y)."""
    mapping: Dict[tuple[int, int, int], Dict[str, Optional[float]]] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            t = _parse_int(r.get("t"))
            if t is None:
                continue
            if kind == "fn":
                x = _parse_int(r.get("fn_x"))
                y = _parse_int(r.get("fn_y"))
                nx = _parse_int(r.get("tp_x"))
                ny = _parse_int(r.get("tp_y"))
            else:
                x = _parse_int(r.get("fp_x"))
                y = _parse_int(r.get("fp_y"))
                nx = _parse_int(r.get("gt_x"))
                ny = _parse_int(r.get("gt_y"))
            if x is None or y is None:
                continue
            dist = _parse_float(r.get("dist"))
            key = (t, x, y)
            mapping[key] = {
                "nearest_x": nx,
                "nearest_y": ny,
                "distance_px": dist,
            }
        except Exception:
            continue
    return mapping


def stage9_test_generate_detection_summary(
    *,
    stage9_video_dir: Path,
    output_dir: Path,
    include_nearest_tp: bool = True,
    verbose: bool = True,
) -> Optional[Path]:
    """Aggregate detection metrics across thresholds into a JSON summary."""
    stage9_video_dir = Path(stage9_video_dir)
    output_dir = Path(output_dir) if output_dir is not None else stage9_video_dir
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage9_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return None

    threshold_dirs = sorted(
        [p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")]
    )
    if not threshold_dirs:
        if verbose:
            print(f"[stage9_test] No thr_* directories found in: {stage9_video_dir}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "detection_summary.json"
    video_name = stage9_video_dir.name
    thresholds_summary: List[Dict] = []

    for thr_dir in threshold_dirs:
        thr_str = thr_dir.name.replace("thr_", "").rstrip("px")
        try:
            thr_val = float(thr_str)
        except ValueError:
            thr_val = None

        tp_rows = _load_detection_rows(thr_dir / "tps.csv")
        fp_rows = _load_detection_rows(thr_dir / "fps.csv")
        fn_rows = _load_detection_rows(thr_dir / "fns.csv")

        fn_nearest_map = (
            _load_nearest_map(thr_dir / "fn_nearest_tp.csv", "fn")
            if include_nearest_tp
            else {}
        )
        fp_nearest_map = (
            _load_nearest_map(thr_dir / "fp_nearest_tp.csv", "fp")
            if include_nearest_tp
            else {}
        )

        thr_entry = {
            "threshold_px": thr_val,
            "threshold_folder": thr_dir.name,
            "counts": {
                "TP": len(tp_rows),
                "FP": len(fp_rows),
                "FN": len(fn_rows),
            },
            "true_positives": [],
            "false_positives": [],
            "false_negatives": [],
        }

        def _convert_row(row: Dict[str, str], detection_type: str) -> Dict:
            t = _parse_int(row.get("t"))
            x = _parse_int(row.get("x"))
            y = _parse_int(row.get("y"))
            conf = _parse_float(row.get("confidence"))
            return {
                "type": detection_type,
                "frame": t,
                "x": x,
                "y": y,
                "confidence": conf,
            }

        for row in tp_rows:
            thr_entry["true_positives"].append(_convert_row(row, "TP"))

        for row in fp_rows:
            det = _convert_row(row, "FP")
            if include_nearest_tp and det["frame"] is not None and det["x"] is not None and det["y"] is not None:
                key = (det["frame"], det["x"], det["y"])
                info = fp_nearest_map.get(key)
                if info:
                    det["nearest_gt"] = info
            thr_entry["false_positives"].append(det)

        for row in fn_rows:
            det = _convert_row(row, "FN")
            if include_nearest_tp and det["frame"] is not None and det["x"] is not None and det["y"] is not None:
                key = (det["frame"], det["x"], det["y"])
                info = fn_nearest_map.get(key)
                if info:
                    det["nearest_tp"] = info
            thr_entry["false_negatives"].append(det)

        thresholds_summary.append(thr_entry)

    summary_payload = {
        "video": video_name,
        "stage9_dir": str(stage9_video_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds_summary,
    }
    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)
    if verbose:
        print(f"[stage9_test] Detection summary written to {summary_path}")
    return summary_path


__all__ = ["stage9_test_generate_detection_summary"]


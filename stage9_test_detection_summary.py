#!/usr/bin/env python3
"""Stage 9 (test) — Aggregate FP/TP/FN details into a JSON summary (caged_fireflies)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import re


_MAX_AREA_RE = re.compile(r"_max(?P<bright>\d+)_brightpx(?P<area>\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class NearestInfo:
    nearest_x: Optional[float]
    nearest_y: Optional[float]
    distance_px: Optional[float]
    image_path: Optional[str]


def _parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(round(float(value)))
    except ValueError:
        return None


def _extract_brightness_area(filepath: str):
    if not filepath:
        return None, None
    m = _MAX_AREA_RE.search(Path(filepath).name)
    if not m:
        return None, None
    return int(m.group('bright')), int(m.group('area'))


def _load_nearest_map(csv_path: Path, key_x: str, key_y: str) -> Dict[Tuple[int, int, int], NearestInfo]:
    mapping: Dict[Tuple[int, int, int], NearestInfo] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = _parse_int(row.get('t'))
                x = _parse_int(row.get(key_x)); y = _parse_int(row.get(key_y))
            except Exception:
                continue
            if frame is None or x is None or y is None:
                continue
            nearest_x = _parse_float(row.get('nearest_tp_x') or row.get('nearest_gt_x'))
            nearest_y = _parse_float(row.get('nearest_tp_y') or row.get('nearest_gt_y'))
            dist = _parse_float(row.get('distance_px') or row.get('dist'))
            image_path = row.get('image_path')
            mapping[(frame, x, y)] = NearestInfo(nearest_x, nearest_y, dist, image_path)
    return mapping


def _load_detection_rows(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def stage9_test_generate_detection_summary(
    *,
    stage9_video_dir: Path,
    output_dir: Path,
    include_nearest_tp: bool = True,
    verbose: bool = True,
) -> Optional[Path]:
    stage9_video_dir = Path(stage9_video_dir)
    output_dir = Path(output_dir) if output_dir is not None else stage9_video_dir
    if not stage9_video_dir.exists():
        if verbose:
            print(f"[stage9_test] Stage5 test directory does not exist: {stage9_video_dir}")
        return None
    threshold_dirs = sorted([p for p in stage9_video_dir.iterdir() if p.is_dir() and p.name.startswith('thr_')])
    if not threshold_dirs:
        if verbose:
            print(f"[stage9_test] No thr_* directories found in: {stage9_video_dir}")
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'detection_summary.json'
    video_name = stage9_video_dir.name
    thresholds_summary: List[Dict] = []
    for thr_dir in threshold_dirs:
        thr_str = thr_dir.name.replace('thr_', '').rstrip('px')
        try:
            thr_val = float(thr_str)
        except ValueError:
            thr_val = None
        fn_nearest_map = _load_nearest_map(thr_dir / 'fn_nearest_tp.csv', 'fn_x', 'fn_y') if include_nearest_tp else {}
        fp_nearest_map = _load_nearest_map(thr_dir / 'fp_nearest_tp.csv', 'fp_x', 'fp_y') if include_nearest_tp else {}
        tp_rows = _load_detection_rows(thr_dir / 'tps.csv')
        fp_rows = _load_detection_rows(thr_dir / 'fps.csv')
        fn_rows = _load_detection_rows(thr_dir / 'fns.csv')
        thr_entry = {
            'threshold_px': thr_val,
            'threshold_folder': thr_dir.name,
            'counts': {'TP': len(tp_rows), 'FP': len(fp_rows), 'FN': len(fn_rows)},
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
        }
        def _convert_row(row: Dict[str, str], detection_type: str) -> Dict:
            frame = _parse_int(row.get('t'))
            x = _parse_int(row.get('x')); y = _parse_int(row.get('y'))
            crop_path = row.get('filepath', '')
            bright, area = _extract_brightness_area(crop_path)
            payload = {
                'type': detection_type,
                'frame': frame,
                'x': x,
                'y': y,
                'confidence': _parse_float(row.get('confidence')),
                'crop_path': crop_path,
                'brightness_max': bright,
                'blob_area': area,
                'source_csv': str(thr_dir / f"{detection_type.lower()}s.csv"),
            }
            if crop_path:
                payload['crop_exists'] = Path(crop_path).exists()
            return payload
        for row in tp_rows:
            thr_entry['true_positives'].append(_convert_row(row, 'TP'))
        for row in fp_rows:
            det = _convert_row(row, 'FP')
            if include_nearest_tp and det['frame'] is not None and det['x'] is not None and det['y'] is not None:
                key = (det['frame'], det['x'], det['y'])
                info = fp_nearest_map.get(key)
                if info:
                    det['nearest_gt'] = {
                        'x': info.nearest_x,
                        'y': info.nearest_y,
                        'distance_px': info.distance_px,
                        'image_path': info.image_path,
                    }
            thr_entry['false_positives'].append(det)
        for row in fn_rows:
            det = _convert_row(row, 'FN')
            if include_nearest_tp and det['frame'] is not None and det['x'] is not None and det['y'] is not None:
                key = (det['frame'], det['x'], det['y'])
                info = fn_nearest_map.get(key)
                if info:
                    det['nearest_tp'] = {
                        'x': info.nearest_x,
                        'y': info.nearest_y,
                        'distance_px': info.distance_px,
                        'image_path': info.image_path,
                    }
            thr_entry['false_negatives'].append(det)
        thresholds_summary.append(thr_entry)
    summary_payload = {
        'video': video_name,
        'stage9_dir': str(stage9_video_dir),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'thresholds': thresholds_summary,
    }
    with summary_path.open('w') as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[stage9_test] Detection summary written to {summary_path}")
    return summary_path


__all__ = ['stage9_test_generate_detection_summary']

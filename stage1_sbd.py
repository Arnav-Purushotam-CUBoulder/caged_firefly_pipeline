"""
Stage 1: SimpleBlobDetector detections.

Reads frames from each input video under 'original videos', detects
bright blobs on grayscale using OpenCV's SimpleBlobDetector, and writes
raw detections per-video to CSV with columns: frame,cx,cy,size

Outputs written to params.STAGE1_DIR as <video_stem>_detections.csv
"""
from pathlib import Path
import csv
from typing import List

import cv2
from tqdm import tqdm

import params


def make_blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor, p.blobColor = True, 255
    p.minThreshold, p.maxThreshold, p.thresholdStep = params.THRESHOLD_Y, 255, 1
    p.minDistBetweenBlobs = params.MIN_DIST_BETWEEN_BLOBS
    p.filterByArea = False
    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)


def run_stage1() -> Path:
    """Run SBD over all videos and write per-video CSV files."""
    out_dir = params.STAGE1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = params.list_videos()
    detector = make_blob_detector()

    for vpath in vids:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        out_csv = out_dir / f"{vpath.stem}_detections.csv"
        total_dets = 0
        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'cx', 'cy', 'size'])
            with tqdm(total=total, desc=f'Stage1 SBD: {vpath.name}', unit='frame', ncols=80) as bar:
                for idx in range(total):
                    ok, frame_bgr = cap.read()
                    bar.update(1)
                    if not ok:
                        break
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    kps = detector.detect(gray)
                    for kp in kps:
                        writer.writerow([idx, float(kp.pt[0]), float(kp.pt[1]), float(kp.size)])
                        total_dets += 1
        cap.release()
        print(f"Stage1: {vpath.stem} frames={total} dets={total_dets} dets_per_frame={total_dets/max(1,total):.2f}")

    return out_dir


__all__ = ['run_stage1']

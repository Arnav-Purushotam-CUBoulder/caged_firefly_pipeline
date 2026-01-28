#!/usr/bin/env python3
"""
YOLO-only pipeline parameters for caged fireflies.

Layout under ROOT:
  ROOT/
    original videos/        # input videos
    stage1_yolo_detect/     # per-video CSVs with detections
    stage2_render/          # annotated videos

Edit ROOT and YOLO_MODEL_WEIGHTS to match your environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import List


# Root folder for this pipeline (EDIT THIS)
# - All stage outputs are saved here.
# - Must contain a subfolder named "original videos" with input videos.
ROOT: str | Path = "~/Desktop/arnav's files/caged_firefly_pipeline/inference output data"
# Normalize ROOT to a Path object even if provided as a string
if not isinstance(ROOT, Path):
    ROOT = Path(str(ROOT)).expanduser()

# Input videos folder (inside ROOT)
ORIGINAL_VIDEOS_DIR: Path = ROOT / "original videos"

# Stage output roots
# Stage 1: YOLO detections
# Stage 2: brightest-pixel + patch-classifier filter
# Stage 3: Gaussian centroid refinement
# Stage 4: rendering
STAGE1_DIR: Path = ROOT / "stage1_yolo_detect"
STAGE2_DIR: Path = ROOT / "stage2_filter"
STAGE3_DIR: Path = ROOT / "stage3_gaussian_centroid"
STAGE4_DIR: Path = ROOT / "stage4_render"

# General
VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}
RUN_PRE_RUN_CLEANUP: bool = True

# Frame cap (None = full video)
MAX_FRAMES: int | None = 4500

# Stage 2 — brightest-pixel + patch-classifier filter
# Drop any detection whose patch max luminance is below this threshold.
# Luminance formula: 0.299*R + 0.587*G + 0.114*B (matching main caged_fireflies pipeline).
STAGE2_BRIGHT_MAX_THRESHOLD: float = 190.0
# Additional Stage 2 bright-area filter (like original Stage4.2)
# A pixel is "bright" if its grayscale >= STAGE2_AREA_INTENSITY_THR.
# Require at least STAGE2_AREA_MIN_BRIGHT_PIXELS such pixels in the patch.
STAGE2_AREA_INTENSITY_THR: int = 190
STAGE2_AREA_MIN_BRIGHT_PIXELS: int = 30
# Patch classifier (ResNet18) used to further filter Stage 2 detections.
STAGE2_PATCH_MODEL_PATH: Path = Path(
"/home/guest/Desktop/arnav's files/caged_firefly_pipeline/models/colo_real_dataset_ResNet18_best_model.pt"
)  # EDIT THIS if needed
STAGE2_PATCH_BOX_SIZE: int = 40
STAGE2_PATCH_BATCH_SIZE: int = 4096
STAGE2_PATCH_POSITIVE_THRESHOLD: float = 0.50  # prob(class 1) >= this to keep

# Stage 2.1 — trajectory + intensity "hill" filter (optional)
# Links Stage 2 detections across time into trajectories, samples a fixed-size
# intensity patch around each detection center, and keeps only trajectories that
# look like a single flash (rise then fall).
#
# When enabled, Stage 2.1 writes:
#   STAGE2_DIR/<stem>/<stem>_bright_traj.csv
# and Stage 3 will prefer that file as its input.
STAGE2_1_ENABLE: bool = False
STAGE2_1_LINK_RADIUS_PX: float = 19.0
STAGE2_1_MAX_FRAME_GAP: int = 3
STAGE2_1_TIME_SCALE: float = 1.0
STAGE2_1_MIN_TRACK_POINTS: int = 3
# Intensity is measured on a fixed-size patch (in pixels) centered on the box.
STAGE2_1_INTENSITY_PATCH_SIZE: int = 40
# "sum" | "mean" | "max" computed on grayscale values in the intensity patch.
STAGE2_1_INTENSITY_METRIC: str = "sum"
# Minimum (max-min) intensity range over a trajectory to be considered.
STAGE2_1_INTENSITY_MIN_RANGE: float = 3000.0
STAGE2_1_REQUIRE_HILL_SHAPE: bool = True
# Hill-shape parameters (rise then fall)
STAGE2_1_HILL_SMOOTH_WINDOW: int = 1
STAGE2_1_HILL_PEAK_POS_MIN_FRAC: float = 0.0
STAGE2_1_HILL_PEAK_POS_MAX_FRAC: float = 1.0
STAGE2_1_HILL_MIN_UP_STEPS: int = 2
STAGE2_1_HILL_MIN_DOWN_STEPS: int = 2
STAGE2_1_HILL_MIN_MONOTONIC_FRAC: float = 0.60

# Stage 3 — Gaussian centroid refinement
# GAUSS_SIGMA = 0 => uniform intensity centroid; >0 => Gaussian-weighted
GAUSS_SIGMA: float = 0.0

# ------------------------------------------------------------------
# Post-pipeline testing/validation configuration (mirrors main pipeline)
# ------------------------------------------------------------------
# Toggles to run test stages from the YOLO orchestrator
RUN_STAGE5_TEST_VALIDATE = False
RUN_STAGE6_TEST_OVERLAY = False
RUN_STAGE7_TEST_FN_ANALYSIS = False
RUN_STAGE8_TEST_FP_ANALYSIS = False
RUN_STAGE9_TEST_SUMMARY = False

# Test output directories (under ROOT)
DIR_STAGE5_TEST_OUT = ROOT / "stage5_test_validation"
DIR_STAGE6_TEST_OUT = ROOT / "stage6_test_overlay"
DIR_STAGE9_TEST_OUT = ROOT / "stage9_test_summary"

# Export a simplified detections CSV (x,y,w,h,frame) from final Stage 3 outputs.
RUN_EXPORT_DETECTIONS_CSV = True
DIR_DETECTIONS_CSV_OUT = ROOT / "detections_csv"

# Ground truth input and time offset
# Preferred layout: one CSV per video under:
#   ROOT / 'ground truth csv folder'
# with filenames mapping to video stem (e.g., <stem>.csv or <stem>_gt.csv)
GT_CSV_DIR = ROOT / "ground truth csv folder"
GT_CSV_PATH = None
GT_T_OFFSET = 0  # frames to subtract from GT 't' to align with video index

# Validation sweep thresholds (pixels)
DIST_THRESHOLDS_PX = [10.0, 20.0, 30.0, 40.0]

# Cropping sizes for validation and overlays (independent of YOLO box size)
TEST_CROP_W = 40
TEST_CROP_H = 40

# YOLO model + inference settings
# Path to trained YOLO model (.pt).
# NOTE: Ultralytics strips single quotes from weight paths internally, so avoid
# directories with "'" in their names (e.g., "arnav's files").
YOLO_MODEL_WEIGHTS: Path = Path(
    "/home/guest/caged_firefly_models/caged_pyrallis_v3_best_yolo.pt"  # EDIT THIS if you move the weights
)

# - YOLO_IMG_SIZE: None → use Ultralytics default (e.g. 640); else int side length.
# - YOLO_CONF_THRES: confidence threshold
# - YOLO_IOU_THRES: IoU threshold for NMS
# - YOLO_DEVICE: 'auto' | 'cpu' | 'cuda' | 'mps' | CUDA index
YOLO_IMG_SIZE: int | None = None
YOLO_CONF_THRES: float = 0.02
YOLO_IOU_THRES: float = 0.3
YOLO_DEVICE: str | int | None = "cpu"

# Rendering
# - RENDER_CODEC: e.g., "mp4v", "avc1", "MJPG"
# - RENDER_FPS_HINT: None → use source fps; else override
RENDER_CODEC: str = "mp4v"
RENDER_FPS_HINT: float | None = None
BOX_COLOR_BGR = (0, 0, 255)
BOX_THICKNESS: int = 1

# GT dedupe threshold (pixels) for test validator
TEST_GT_DEDUPE_DIST_PX = 30.0

# Overlay rendering thickness for test videos
OVERLAY_BOX_THICKNESS = BOX_THICKNESS


def list_videos() -> List[Path]:
    """Return all video files under ORIGINAL_VIDEOS_DIR."""
    ORIGINAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for p in sorted(ORIGINAL_VIDEOS_DIR.iterdir()):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            out.append(p)
    return out


__all__ = [
    # paths
    "ROOT",
    "ORIGINAL_VIDEOS_DIR",
    "STAGE1_DIR",
    "STAGE2_DIR",
    "STAGE3_DIR",
    "STAGE4_DIR",
    # general
    "VIDEO_EXTS",
    "RUN_PRE_RUN_CLEANUP",
    "MAX_FRAMES",
    "STAGE2_BRIGHT_MAX_THRESHOLD",
    "STAGE2_AREA_INTENSITY_THR",
    "STAGE2_AREA_MIN_BRIGHT_PIXELS",
    "STAGE2_PATCH_MODEL_PATH",
    "STAGE2_PATCH_BOX_SIZE",
    "STAGE2_PATCH_BATCH_SIZE",
    "STAGE2_PATCH_POSITIVE_THRESHOLD",
    "STAGE2_1_ENABLE",
    "STAGE2_1_LINK_RADIUS_PX",
    "STAGE2_1_MAX_FRAME_GAP",
    "STAGE2_1_TIME_SCALE",
    "STAGE2_1_MIN_TRACK_POINTS",
    "STAGE2_1_INTENSITY_PATCH_SIZE",
    "STAGE2_1_INTENSITY_METRIC",
    "STAGE2_1_INTENSITY_MIN_RANGE",
    "STAGE2_1_REQUIRE_HILL_SHAPE",
    "STAGE2_1_HILL_SMOOTH_WINDOW",
    "STAGE2_1_HILL_PEAK_POS_MIN_FRAC",
    "STAGE2_1_HILL_PEAK_POS_MAX_FRAC",
    "STAGE2_1_HILL_MIN_UP_STEPS",
    "STAGE2_1_HILL_MIN_DOWN_STEPS",
    "STAGE2_1_HILL_MIN_MONOTONIC_FRAC",
    "GAUSS_SIGMA",
    # yolo
    "YOLO_MODEL_WEIGHTS",
    "YOLO_IMG_SIZE",
    "YOLO_CONF_THRES",
    "YOLO_IOU_THRES",
    "YOLO_DEVICE",
    # render
    "RENDER_CODEC",
    "RENDER_FPS_HINT",
    "BOX_COLOR_BGR",
    "BOX_THICKNESS",
    # helpers
    "list_videos",
]

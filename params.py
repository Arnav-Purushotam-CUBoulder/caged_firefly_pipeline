"""
Centralized pipeline configuration and paths.
All code modules import settings from here.

Layout expectation under PIPELINE_ROOT:
  PIPELINE_ROOT/
    original videos/          # input videos to process
    stage0_outputs/
    stage1_outputs/
    stage2_outputs/
    stage3_outputs/
    stage4_outputs/
"""
from pathlib import Path
from typing import List

# Root path for data processing (contains 'original videos').
# Update this to your dataset root.
PIPELINE_ROOT = Path("/home/guest/Desktop/arnav's files/caged_firefly_pipeline/inference output data")

# Input directory containing the videos
ORIGINAL_VIDEOS_DIR = PIPELINE_ROOT / 'original videos'

# Stage output directories
STAGE0_DIR = PIPELINE_ROOT / 'stage0_outputs'
STAGE1_DIR = PIPELINE_ROOT / 'stage1_outputs'
STAGE2_DIR = PIPELINE_ROOT / 'stage2_outputs'
STAGE3_DIR = PIPELINE_ROOT / 'stage3_outputs'
STAGE4_DIR = PIPELINE_ROOT / 'stage4_outputs'
STAGE4_1_DIR = PIPELINE_ROOT / 'stage4_1_outputs'
STAGE4_2_DIR = PIPELINE_ROOT / 'stage4_2_outputs'
STAGE_RENDERER_DIR = PIPELINE_ROOT / 'stage_renderer_outputs'

# Post-pipeline testing/validation output directories (continue after Stage 4)
DIR_STAGE5_TEST_OUT = PIPELINE_ROOT / 'stage5_test_validation'
DIR_STAGE6_TEST_OUT = PIPELINE_ROOT / 'stage6_test_overlay'
DIR_STAGE9_TEST_OUT = PIPELINE_ROOT / 'stage9_test_summary'

# Video scanning
VIDEO_EXTS = {'.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV'}

def list_videos() -> List[Path]:
    ORIGINAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    vids = [p for p in ORIGINAL_VIDEOS_DIR.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS]
    vids.sort()
    return vids

# Detection and post-processing params
THRESHOLD_Y = 175            # grayscale threshold for SimpleBlobDetector
MIN_DIST_BETWEEN_BLOBS = 1   # min distance between blobs (px)
# Legacy: kept for reference; Stage3 now uses centroid-distance instead of IoU
OVERLAP_IOU = 0.25           # (unused in new Stage3)
BOX_SIZE_PX = 40             # square box size around centroid (px)
BOX_COLOR_BGR = (0, 0, 255)  # drawing color for final render
BOX_THICKNESS = 1
FOURCC = 'mp4v'
N = 4500                     # None → process full video
BATCH_SIZE = 32768            # batch size for CNN inference

# Stage1 variant: 'sbd' (OpenCV SimpleBlobDetector) or 'cucim' (GPU blob_log)
STAGE1_VARIANT = 'cucim'

# cuCIM Stage1 tuning (used when STAGE1_VARIANT='cucim')
# Detector: 'log' or 'dog' (DoG is typically faster)
CUCIM_DETECTOR = 'log'
# Sigma search space
CUCIM_MIN_SIGMA = 1.0
CUCIM_MAX_SIGMA = 4.0
CUCIM_NUM_SIGMA = 8
CUCIM_LOG_SCALE = False
CUCIM_OVERLAP = 0.5
# Threshold on LoG/DoG response (relative scale)
CUCIM_THRESHOLD_RESP = 0.05
# Optional downscale factor before detection (1=no downscale, 2=half res)
CUCIM_DOWNSCALE = 1
# Number of frames to upload to GPU at once (micro-batching)
CUCIM_BATCH_FRAMES = 250

# Stage3 (merge) by centroid distance and heaviest RGB sum
MERGE_DISTANCE_PX = 30.0

# Stage4 Gaussian centroid refinement
# Patch size now derived from BOX_SIZE_PX for both width/height
GAUSS_SIGMA = 0.0            # 0: uniform; >0: Gaussian-weighted centroid

# Stage 4.1 — brightest-pixel filter on Stage4 outputs
RUN_STAGE4_1 = True
STAGE4_1_BRIGHT_MAX_THRESHOLD = 190  # drop candidates whose patch max luminance < this

# Stage 4.2 — bright-area pixels filter
RUN_STAGE4_2 = True
STAGE4_2_INTENSITY_THR = 190        # pixel is "bright" if luminance >= this
STAGE4_2_MIN_BRIGHT_PIXELS = 20      # require at least this many bright pixels

# CNN classification
MODEL_PATH = str(Path("/home/guest/Desktop/arnav's files/caged_firefly_pipeline/models/colo_real_dataset_ResNet18_best_model.pt"))
CONFIDENCE_MIN = 0.98        # accept as firefly if conf >= this

# ------------------------------------------------------------------
# Post-pipeline testing/validation configuration
# ------------------------------------------------------------------
# Toggles to run test stages from the orchestrator
RUN_STAGE5_TEST_VALIDATE = True
RUN_STAGE6_TEST_OVERLAY = True
RUN_STAGE7_TEST_FN_ANALYSIS = True
RUN_STAGE8_TEST_FP_ANALYSIS = True
RUN_STAGE9_TEST_SUMMARY = True

# Ground truth input and time offset
# Preferred layout: one CSV per video under the folder literally named
# 'ground truth csv folder', with filenames that map to the input video stem
# (e.g., <video_stem>.csv or <video_stem>_gt.csv).
GT_CSV_DIR = PIPELINE_ROOT / 'ground truth csv folder'
# Optional single-CSV fallback (used only if provided and found):
GT_CSV_PATH = None
GT_T_OFFSET = 0  # frames to subtract from GT 't' to align with video index

# Validation sweep thresholds (pixels)
DIST_THRESHOLDS_PX = [10.0,20.0,30.0,40.0]

# Cropping sizes for validation and overlays (defaults derived from BOX_SIZE_PX)
TEST_CROP_W = int(BOX_SIZE_PX)
TEST_CROP_H = int(BOX_SIZE_PX)

# GT dedupe threshold (used by Stage 5 test validate)
TEST_GT_DEDUPE_DIST_PX = float(MERGE_DISTANCE_PX)

# Overlay rendering
OVERLAY_BOX_THICKNESS = BOX_THICKNESS

# Back-compat aliases (if referenced elsewhere in this repo)
# Old names (if any) map to new test-stage names
DIR_STAGE9_OUT = DIR_STAGE5_TEST_OUT
DIR_STAGE10_OUT = DIR_STAGE6_TEST_OUT
DIR_STAGE14_OUT = DIR_STAGE9_TEST_OUT
RUN_STAGE9 = RUN_STAGE5_TEST_VALIDATE
RUN_STAGE10 = RUN_STAGE6_TEST_OVERLAY
RUN_STAGE11 = RUN_STAGE7_TEST_FN_ANALYSIS
RUN_STAGE12 = RUN_STAGE8_TEST_FP_ANALYSIS
RUN_STAGE14 = RUN_STAGE9_TEST_SUMMARY
POST_STAGE9_DIR = DIR_STAGE5_TEST_OUT
POST_STAGE10_DIR = DIR_STAGE6_TEST_OUT
POST_STAGE14_DIR = DIR_STAGE9_TEST_OUT
POST_CROP_W = TEST_CROP_W
POST_CROP_H = TEST_CROP_H
POST_GT_DEDUPE_DIST_PX = TEST_GT_DEDUPE_DIST_PX

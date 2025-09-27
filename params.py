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
PIPELINE_ROOT = Path('/Users/arnavps/Desktop/New DL project data to transfer to external disk/caged pyrallis inference data')

# Input directory containing the videos
ORIGINAL_VIDEOS_DIR = PIPELINE_ROOT / 'original videos'

# Stage output directories
STAGE0_DIR = PIPELINE_ROOT / 'stage0_outputs'
STAGE1_DIR = PIPELINE_ROOT / 'stage1_outputs'
STAGE2_DIR = PIPELINE_ROOT / 'stage2_outputs'
STAGE3_DIR = PIPELINE_ROOT / 'stage3_outputs'
STAGE4_DIR = PIPELINE_ROOT / 'stage4_outputs'

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
OVERLAP_IOU = 0.25           # IoU threshold for merging boxes
BOX_SIZE_PX = 40             # square box size around centroid (px)
BOX_COLOR_BGR = (0, 0, 255)  # drawing color for final render
BOX_THICKNESS = 1
FOURCC = 'mp4v'
N = 100                      # None → process full video
BATCH_SIZE = 4096             # batch size for CNN inference (multiple of 2:: 256, 512, 1024, 2048)

# CNN classification
MODEL_PATH = str(Path('/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/caged_fireflies/models and other data/colo_real_dataset_ResNet18_best_model.pt'))
CONFIDENCE_MIN = 0.98        # accept as firefly if conf >= this

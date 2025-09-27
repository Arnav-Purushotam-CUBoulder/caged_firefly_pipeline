"""
Pipeline orchestrator: runs Stage0 → Stage1 → Stage2 → Stage3 → Stage4 (Gaussian) → Renderer.

Usage: python code/orchestrator.py
All configuration is in params.py.
"""
from pathlib import Path
from time import perf_counter
import os
import cv2

import params
from stage0_cleanup import run_stage0
def _select_stage1_runner():
    try:
        variant = getattr(params, 'STAGE1_VARIANT', 'sbd').lower()
    except Exception:
        variant = 'sbd'
    if variant == 'cucim':
        from stage1_cucim import run_stage1 as _run
    else:
        from stage1_sbd import run_stage1 as _run
    return _run
from stage2_cnn import run_stage2
from stage3_merge import run_stage3
from stage4_gaussian_centroid import run_stage4
from stage_renderer import run_stage_renderer


def main():
    # Tame OpenCV FFmpeg read warnings and increase read attempts for multi-stream MP4s
    os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '999999')
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    t0 = perf_counter()
    print("── Running Stage 0: Cleanup …")
    s0 = run_stage0()
    print(f"Stage0 outputs → {s0}")
    print("── Running Stage 1: detections …")
    run_stage1 = _select_stage1_runner()
    s1 = run_stage1()
    print(f"Stage1 outputs → {s1}")

    print("── Running Stage 2: CNN classification …")
    s2 = run_stage2()
    print(f"Stage2 outputs → {s2}")

    print("── Running Stage 3: Merge boxes …")
    s3 = run_stage3()
    print(f"Stage3 outputs → {s3}")

    print("── Running Stage 4: Gaussian centroid refinement …")
    s4 = run_stage4()
    print(f"Stage4 outputs → {s4}")

    print("── Running Renderer: Annotated videos …")
    sr = run_stage_renderer()
    print(f"Renderer outputs → {sr}")

    dt = perf_counter() - t0
    print(f"Done in {dt:.1f}s")


if __name__ == '__main__':
    main()

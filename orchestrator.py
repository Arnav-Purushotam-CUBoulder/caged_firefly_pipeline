"""
Pipeline orchestrator: runs Stage0 → Stage1 → Stage2 → Stage3 → Stage4 (Gaussian) → Renderer.

Usage: python code/orchestrator.py
All configuration is in params.py.
"""
from pathlib import Path
from time import perf_counter
import os
import sys
import cv2

# Ensure the parent of this file (the package root) is importable
_here = Path(__file__).resolve()
_pkg_root = _here.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

# Support both script and package execution
try:
    import params  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from importlib import import_module
    params = import_module('caged_firefly_pipeline.params')  # type: ignore
try:
    from stage0_cleanup import run_stage0
except ModuleNotFoundError:  # pragma: no cover
    from .stage0_cleanup import run_stage0
def _select_stage1_runner():
    try:
        variant = getattr(params, 'STAGE1_VARIANT', 'sbd').lower()
    except Exception:
        variant = 'sbd'
    if variant == 'cucim':
        try:
            from stage1_cucim import run_stage1 as _run
        except ModuleNotFoundError:  # pragma: no cover
            from .stage1_cucim import run_stage1 as _run
    else:
        try:
            from stage1_sbd import run_stage1 as _run
        except ModuleNotFoundError:  # pragma: no cover
            from .stage1_sbd import run_stage1 as _run
    return _run
try:
    from stage2_cnn import run_stage2
    from stage3_merge import run_stage3
    from stage4_gaussian_centroid import run_stage4
    from stage_renderer import run_stage_renderer
    from stage4_1_brightest_filter import run_stage4_1
    from stage4_2_bright_area_filter import run_stage4_2
except ModuleNotFoundError:  # pragma: no cover
    from .stage2_cnn import run_stage2
    from .stage3_merge import run_stage3
    from .stage4_gaussian_centroid import run_stage4
    from .stage_renderer import run_stage_renderer
    from .stage4_1_brightest_filter import run_stage4_1
    from .stage4_2_bright_area_filter import run_stage4_2

# Post-pipeline test modules
try:
    from stage5_test_validate import stage5_test_validate_against_gt
    from stage6_test_overlay_gt_vs_model import stage6_test_overlay_gt_vs_model
    from stage7_test_fn_analysis import stage7_test_fn_nearest_tp_analysis
    from stage8_test_fp_analysis import stage8_test_fp_nearest_tp_analysis
    from stage9_test_detection_summary import stage9_test_generate_detection_summary
except ModuleNotFoundError:  # pragma: no cover
    from .stage5_test_validate import stage5_test_validate_against_gt
    from .stage6_test_overlay_gt_vs_model import stage6_test_overlay_gt_vs_model
    from .stage7_test_fn_analysis import stage7_test_fn_nearest_tp_analysis
    from .stage8_test_fp_analysis import stage8_test_fp_nearest_tp_analysis
    from .stage9_test_detection_summary import stage9_test_generate_detection_summary


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

    # Optional Stage 4.1: brightest-pixel filter on Stage4 outputs
    if getattr(params, 'RUN_STAGE4_1', True):
        print("── Running Stage 4.1: Brightest-pixel filter …")
        s41 = run_stage4_1()
        print(f"Stage4.1 outputs → {s41}")

    # Optional Stage 4.2: bright-area pixels filter on Stage4/4.1 outputs
    if getattr(params, 'RUN_STAGE4_2', True):
        print("── Running Stage 4.2: Bright-area pixels filter …")
        s42 = run_stage4_2()
        print(f"Stage4.2 outputs → {s42}")

    print("── Running Renderer: Annotated videos …")
    sr = run_stage_renderer()
    print(f"Renderer outputs → {sr}")

    # ─────────────────────────────────────────────
    # Post-pipeline validation and analysis (optional)
    # ─────────────────────────────────────────────
    try:
        videos = params.list_videos()
        vid_map = {p.stem: p for p in videos}

        def _find_gt_csv_for_video(stem: str) -> Path | None:
            # Prefer per-video CSVs in GT_CSV_DIR
            try:
                gt_dir = getattr(params, 'GT_CSV_DIR', None)
                if gt_dir is not None and gt_dir.exists():
                    candidates = []
                    # strict matches first
                    for pat in (f"{stem}.csv", f"{stem}_gt.csv", f"{stem}-gt.csv"):
                        p = gt_dir / pat
                        if p.exists():
                            candidates.append(p)
                    if candidates:
                        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                    # fallback: any CSV containing the stem uniquely
                    loose = [p for p in gt_dir.glob('*.csv') if stem in p.stem]
                    if len(loose) == 1:
                        return loose[0]
            except Exception:
                pass
            # Fallback to single GT_CSV_PATH if provided
            try:
                gt_single = getattr(params, 'GT_CSV_PATH', None)
                if gt_single is not None:
                    gp = Path(gt_single)
                    return gp if gp.exists() else None
            except Exception:
                pass
            return None
        for stem, vpath in vid_map.items():
            # Use only Stage 4.2 filtered predictions for test/overlay stages
            pred_csv = (params.STAGE4_2_DIR / stem / f"{stem}_gauss_brightarea_filtered.csv").resolve()
            if not pred_csv.exists():
                print(f"[post] Warning: Stage4.2 CSV not found for video '{stem}'. Skipping test stages.")
                continue
            if vpath is None:
                print(f"[post] Warning: no matching video for {pred_csv.name}")
                continue

            # Stage 5 (test) — validation
            if getattr(params, 'RUN_STAGE5_TEST_VALIDATE', True):
                out9_dir = (params.DIR_STAGE5_TEST_OUT / stem).resolve()
                gt_csv_path = _find_gt_csv_for_video(stem)
                if gt_csv_path is None:
                    print(f"[post] Warning: no GT CSV found for video '{stem}'. Skipping test stages.")
                    continue
                stage5_test_validate_against_gt(
                    orig_video_path=vpath,
                    pred_csv_path=pred_csv,
                    gt_csv_path=gt_csv_path,
                    out_dir=out9_dir,
                    dist_thresholds=list(getattr(params, 'DIST_THRESHOLDS_PX', [1.0,2.0,3.0,4.0,5.0])),
                    crop_w=int(getattr(params, 'TEST_CROP_W', getattr(params, 'BOX_SIZE_PX', 40))),
                    crop_h=int(getattr(params, 'TEST_CROP_H', getattr(params, 'BOX_SIZE_PX', 40))),
                    gt_t_offset=int(getattr(params, 'GT_T_OFFSET', 0)),
                    max_frames=getattr(params, 'N', None),
                    only_firefly_rows=True,
                    gt_dedupe_dist_threshold_px=float(getattr(params, 'TEST_GT_DEDUPE_DIST_PX', 2.0)),
                )

            # Stage 6 (test) — overlay
            if getattr(params, 'RUN_STAGE6_TEST_OVERLAY', True):
                out10_path = (
                    params.DIR_STAGE6_TEST_OUT
                    / f"{stem}_overlay_GT-green_MODEL-red_overlap-yellow.mp4"
                ).resolve()
                stage6_test_overlay_gt_vs_model(
                    orig_video_path=vpath,
                    pred_csv_path=pred_csv,
                    post9_dir=(params.DIR_STAGE5_TEST_OUT / stem).resolve(),
                    out_video_path=out10_path,
                    gt_box_w=int(getattr(params, 'TEST_CROP_W', getattr(params, 'BOX_SIZE_PX', 40))),
                    gt_box_h=int(getattr(params, 'TEST_CROP_H', getattr(params, 'BOX_SIZE_PX', 40))),
                    thickness=int(getattr(params, 'OVERLAY_BOX_THICKNESS', getattr(params, 'BOX_THICKNESS', 1))),
                    max_frames=getattr(params, 'N', None),
                    render_threshold_overlays=True,
                )

            # Stage 7 (test) — FN analysis
            if getattr(params, 'RUN_STAGE7_TEST_FN_ANALYSIS', True):
                stage7_test_fn_nearest_tp_analysis(
                    stage9_video_dir=(params.DIR_STAGE5_TEST_OUT / stem).resolve(),
                    orig_video_path=vpath,
                    box_w=int(getattr(params, 'TEST_CROP_W', getattr(params, 'BOX_SIZE_PX', 40))),
                    box_h=int(getattr(params, 'TEST_CROP_H', getattr(params, 'BOX_SIZE_PX', 40))),
                    thickness=int(getattr(params, 'OVERLAY_BOX_THICKNESS', getattr(params, 'BOX_THICKNESS', 1))),
                    verbose=True,
                )

            # Stage 8 (test) — FP analysis
            if getattr(params, 'RUN_STAGE8_TEST_FP_ANALYSIS', True):
                stage8_test_fp_nearest_tp_analysis(
                    stage9_video_dir=(params.DIR_STAGE5_TEST_OUT / stem).resolve(),
                    orig_video_path=vpath,
                    box_w=int(getattr(params, 'TEST_CROP_W', getattr(params, 'BOX_SIZE_PX', 40))),
                    box_h=int(getattr(params, 'TEST_CROP_H', getattr(params, 'BOX_SIZE_PX', 40))),
                    thickness=int(getattr(params, 'OVERLAY_BOX_THICKNESS', getattr(params, 'BOX_THICKNESS', 1))),
                    verbose=True,
                )

            # Stage 9 (test) — summary JSON
            if getattr(params, 'RUN_STAGE9_TEST_SUMMARY', True):
                out14_dir = (params.DIR_STAGE9_TEST_OUT / stem).resolve()
                stage9_test_generate_detection_summary(
                    stage9_video_dir=(params.DIR_STAGE5_TEST_OUT / stem).resolve(),
                    output_dir=out14_dir,
                    include_nearest_tp=True,
                    verbose=True,
                )
    except Exception as e:
        print(f"[post] Warning: post-pipeline tests encountered an issue: {e}")

    dt = perf_counter() - t0
    print(f"Done in {dt:.1f}s")


if __name__ == '__main__':
    main()

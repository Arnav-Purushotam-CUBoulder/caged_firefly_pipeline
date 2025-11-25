#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2

import params
from stage0_cleanup import cleanup_root
from stage1_yolo_detect import run_for_video as stage1_run
from stage2_filter import run_for_video as stage2_run
from stage3_gaussian_centroid import run_for_video as stage3_run
from stage4_render import run_for_video as stage4_run

# Local wrappers around the original caged_fireflies test suite
from stage5_test_validate import stage5_test_validate_against_gt
from stage6_test_overlay_gt_vs_model import stage6_test_overlay_gt_vs_model
from stage7_test_fn_analysis import stage7_test_fn_nearest_tp_analysis
from stage8_test_fp_analysis import stage8_test_fp_nearest_tp_analysis
from stage9_test_detection_summary import stage9_test_generate_detection_summary


def _print_stage_times(stage_times: dict[str, float]) -> None:
    keys = ["stage1", "stage2", "stage3", "stage4"]
    print("\nTiming summary:")
    for k in keys:
        print(f"  {k}: {stage_times.get(k, 0.0):.2f}s")
    print(f"  total: {sum(stage_times.get(k, 0.0) for k in keys):.2f}s")


def _summarize_detections(csv_path: Path, video_path: Path) -> None:
    """Print simple metrics for a Stage1 CSV."""
    try:
        total_boxes = 0
        frames_with: set[int] = set()
        with csv_path.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                total_boxes += 1
                try:
                    frames_with.add(int(row["frame"]))
                except Exception:
                    continue

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if params.MAX_FRAMES is not None:
            frames_considered = min(total_frames or params.MAX_FRAMES, int(params.MAX_FRAMES))
        else:
            frames_considered = total_frames or len(frames_with)

        frames_considered = max(frames_considered, 1)
        avg_boxes_per_frame = total_boxes / frames_considered if frames_considered else 0.0

        print(
            f"[metrics] {video_path.name}: frames_considered={frames_considered}, "
            f"frames_with_detections={len(frames_with)}, total_boxes={total_boxes}, "
            f"avg_boxes/frame={avg_boxes_per_frame:.2f}"
        )
    except Exception as exc:
        print(f"[metrics] Warning: failed to summarize detections for {video_path.name}: {exc}")


def _find_gt_csv_for_video(stem: str) -> Path | None:
    """Locate a GT CSV for a given video stem, mirroring the main pipeline logic."""
    try:
        gt_dir = getattr(params, "GT_CSV_DIR", None)
    except Exception:
        gt_dir = None
    if gt_dir is not None:
        try:
            gt_dir = Path(gt_dir)
            if gt_dir.exists():
                # Prefer exact stem, then stem_gt, then first match
                candidates = [
                    gt_dir / f"{stem}.csv",
                    gt_dir / f"{stem}_gt.csv",
                ]
                for c in candidates:
                    if c.exists():
                        return c
                # Fallback: any CSV whose name starts with stem
                for c in sorted(gt_dir.glob(f"{stem}*.csv")):
                    return c
        except Exception:
            pass
    # Fallback: single GT_CSV_PATH if provided
    try:
        gt_single = getattr(params, "GT_CSV_PATH", None)
        if gt_single is not None:
            gp = Path(gt_single)
            return gp if gp.exists() else None
    except Exception:
        pass
    return None


def _build_test_pred_csv(stem: str, stage3_csv: Path, out_dir: Path) -> Path:
    """Build a point predictions CSV for the test suite from Stage 3 boxes.

    Converts (x,y,w,h,frame,firefly_logit,background_logit) into:
      x_center, y_center, t, firefly_logit, background_logit
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}_pred_points.csv"
    with stage3_csv.open("r", newline="") as f_in, out_csv.open("w", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(
            f_out,
            fieldnames=["x", "y", "t", "firefly_logit", "background_logit"],
        )
        w.writeheader()
        for row in r:
            try:
                x = float(row["x"])
                y = float(row["y"])
                w_box = float(row.get("w", 0.0))
                h_box = float(row.get("h", 0.0))
                t = int(row.get("frame", row.get("t", 0)))
                lf = float(row.get("firefly_logit", "nan"))
                lb = float(row.get("background_logit", "nan"))
            except Exception:
                continue
            cx = x + w_box / 2.0
            cy = y + h_box / 2.0
            w.writerow(
                {
                    "x": float(cx),
                    "y": float(cy),
                    "t": int(t),
                    "firefly_logit": lf,
                    "background_logit": lb,
                }
            )
    return out_csv


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    _ = argv  # unused for now

    params.ROOT.mkdir(parents=True, exist_ok=True)

    if params.RUN_PRE_RUN_CLEANUP:
        print("[orchestrator] Running Stage 0 cleanup…")
        cleanup_root(verbose=True)

    videos = params.list_videos()
    if not videos:
        print(f"No videos found. Place input files in: {params.ORIGINAL_VIDEOS_DIR}")
        return 1

    print(f"Found {len(videos)} video(s) in {params.ORIGINAL_VIDEOS_DIR}")

    for vid in videos:
        print(f"\n=== Processing: {vid.name} ===")
        stage_times: dict[str, float] = {}

        t0 = time.perf_counter()
        s1_csv = stage1_run(vid)
        stage_times["stage1"] = time.perf_counter() - t0
        print(f"Stage1  Time: {stage_times['stage1']:.2f}s (csv: {s1_csv.name})")

        t0 = time.perf_counter()
        s2_csv = stage2_run(vid)
        stage_times["stage2"] = time.perf_counter() - t0
        print(f"Stage2  Time: {stage_times['stage2']:.2f}s (csv: {s2_csv.name})")

        t0 = time.perf_counter()
        s3_csv = stage3_run(vid)
        stage_times["stage3"] = time.perf_counter() - t0
        print(f"Stage3  Time: {stage_times['stage3']:.2f}s (csv: {s3_csv.name})")

        t0 = time.perf_counter()
        out_vid = stage4_run(vid)
        stage_times["stage4"] = time.perf_counter() - t0
        print(f"Stage4  Time: {stage_times['stage4']:.2f}s (video: {out_vid.name})")

        _summarize_detections(s3_csv, vid)
        _print_stage_times(stage_times)

        # --- Post-pipeline test suite (per video) ---
        try:
            gt_csv_path = _find_gt_csv_for_video(vid.stem)
            if gt_csv_path is None:
                print(f"[post] Warning: no GT CSV found for video '{vid.stem}'. Skipping test stages.")
                continue

            # Build point-predictions CSV for test suite from Stage3 boxes
            test_root = params.DIR_STAGE5_TEST_OUT / vid.stem
            pred_points_csv = _build_test_pred_csv(vid.stem, s3_csv, test_root)

            # Stage 5 — validation vs GT
            if getattr(params, "RUN_STAGE5_TEST_VALIDATE", True):
                stage5_test_validate_against_gt(
                    orig_video_path=vid,
                    pred_csv_path=pred_points_csv,
                    gt_csv_path=gt_csv_path,
                    out_dir=test_root,
                    dist_thresholds=list(
                        getattr(
                            params,
                            "DIST_THRESHOLDS_PX",
                            [10.0, 20.0, 30.0, 40.0],
                        )
                    ),
                    crop_w=int(
                        getattr(
                            params,
                            "TEST_CROP_W",
                            40,
                        )
                    ),
                    crop_h=int(
                        getattr(
                            params,
                            "TEST_CROP_H",
                            40,
                        )
                    ),
                    gt_t_offset=int(getattr(params, "GT_T_OFFSET", 0)),
                    max_frames=getattr(params, "MAX_FRAMES", None),
                    only_firefly_rows=True,
                    gt_dedupe_dist_threshold_px=float(
                        getattr(params, "TEST_GT_DEDUPE_DIST_PX", 30.0)
                    ),
                )

            # Stage 6 — overlay GT vs model
            if getattr(params, "RUN_STAGE6_TEST_OVERLAY", True):
                out10_path = (
                    params.DIR_STAGE6_TEST_OUT / f"{vid.stem}_overlay.mp4"
                ).resolve()
                stage6_test_overlay_gt_vs_model(
                    orig_video_path=vid,
                    pred_csv_path=pred_points_csv,
                    post9_dir=test_root.resolve(),
                    out_video_path=out10_path,
                    gt_box_w=int(
                        getattr(
                            params,
                            "TEST_CROP_W",
                            40,
                        )
                    ),
                    gt_box_h=int(
                        getattr(
                            params,
                            "TEST_CROP_H",
                            40,
                        )
                    ),
                    thickness=int(
                        getattr(
                            params,
                            "OVERLAY_BOX_THICKNESS",
                            getattr(params, "BOX_THICKNESS", 1),
                        )
                    ),
                    max_frames=getattr(params, "MAX_FRAMES", None),
                    render_threshold_overlays=True,
                )

            # Stage 7 — FN analysis
            if getattr(params, "RUN_STAGE7_TEST_FN_ANALYSIS", True):
                stage7_test_fn_nearest_tp_analysis(
                    stage9_video_dir=test_root.resolve(),
                    orig_video_path=vid,
                    box_w=int(
                        getattr(
                            params,
                            "TEST_CROP_W",
                            40,
                        )
                    ),
                    box_h=int(
                        getattr(
                            params,
                            "TEST_CROP_H",
                            40,
                        )
                    ),
                    thickness=int(
                        getattr(
                            params,
                            "OVERLAY_BOX_THICKNESS",
                            getattr(params, "BOX_THICKNESS", 1),
                        )
                    ),
                    verbose=True,
                )

            # Stage 8 — FP analysis
            if getattr(params, "RUN_STAGE8_TEST_FP_ANALYSIS", True):
                stage8_test_fp_nearest_tp_analysis(
                    stage9_video_dir=test_root.resolve(),
                    orig_video_path=vid,
                    box_w=int(
                        getattr(
                            params,
                            "TEST_CROP_W",
                            40,
                        )
                    ),
                    box_h=int(
                        getattr(
                            params,
                            "TEST_CROP_H",
                            40,
                        )
                    ),
                    thickness=int(
                        getattr(
                            params,
                            "OVERLAY_BOX_THICKNESS",
                            getattr(params, "BOX_THICKNESS", 1),
                        )
                    ),
                    verbose=True,
                )

            # Stage 9 — summary JSON
            if getattr(params, "RUN_STAGE9_TEST_SUMMARY", True):
                out14_dir = (params.DIR_STAGE9_TEST_OUT / vid.stem).resolve()
                stage9_test_generate_detection_summary(
                    stage9_video_dir=test_root.resolve(),
                    output_dir=out14_dir,
                    include_nearest_tp=True,
                    verbose=True,
                )
        except Exception as e:
            print(f"[post] Warning: post-pipeline tests for {vid.stem} encountered an issue: {e}")

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import shutil
import sys
import time
from pathlib import Path

import params

import cv2
from stage0_cleanup import cleanup_root
from stage1_yolo_detect import run_for_video as stage1_run
from stage1_render_raw import run_for_video as stage1_render_raw_run
from stage2_filter import run_for_video as stage2_run
from stage2_1_trajectory_intensity_filter import run_for_video as stage2_1_run
from stage3_gaussian_centroid import run_for_video as stage3_run
from stage4_render import run_for_video as stage4_run

# Local wrappers around the original caged_fireflies test suite
from stage5_test_validate import stage5_test_validate_against_gt
from stage6_test_overlay_gt_vs_model import stage6_test_overlay_gt_vs_model
from stage7_test_fn_analysis import stage7_test_fn_nearest_tp_analysis
from stage8_test_fp_analysis import stage8_test_fp_nearest_tp_analysis
from stage9_test_detection_summary import stage9_test_generate_detection_summary


def _is_gvfs_smb_path(path: Path) -> bool:
    """Heuristic to detect GNOME GVFS SMB mounts (FUSE), which can be flaky for OpenCV/FFmpeg."""
    s = str(path)
    return "/gvfs/" in s and "smb-share:" in s


def _format_bytes(num_bytes: float) -> str:
    num = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            if unit == "B":
                return f"{int(num)}{unit}"
            return f"{num:0.1f}{unit}"
        num /= 1024.0
    return f"{num:0.1f}PB"


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.5))
    m, s = divmod(seconds, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _copy_file_with_progress(src: Path, dst: Path, *, label: str) -> None:
    """Copy file with a simple byte-based progress bar."""
    try:
        total = int(src.stat().st_size)
    except Exception:
        total = 0

    chunk_bytes = 8 * 1024 * 1024
    bar_w = 36
    copied = 0
    t0 = time.perf_counter()
    last_update = 0.0

    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            buf = fsrc.read(chunk_bytes)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)

            now = time.perf_counter()
            if now - last_update < 0.2:
                continue
            if total > 0 and copied >= total:
                continue

            elapsed = max(now - t0, 1e-6)
            speed = copied / elapsed
            if total > 0:
                frac = min(copied / total, 1.0)
                fill = int(frac * bar_w)
                bar = "█" * fill + "·" * (bar_w - fill)
                remaining = max(0, total - copied)
                eta = remaining / speed if speed > 0 else 0.0
                sys.stdout.write(
                    f"\r[{bar}] {frac*100:6.2f}% "
                    f"{_format_bytes(copied)}/{_format_bytes(total)} "
                    f"{_format_bytes(speed)}/s ETA {_format_eta(eta)}  {label}"
                )
            else:
                sys.stdout.write(
                    f"\r{_format_bytes(copied)} copied  {_format_bytes(speed)}/s  {label}"
                )
            sys.stdout.flush()
            last_update = now

    try:
        shutil.copystat(src, dst)
    except Exception:
        pass

    # Final line
    elapsed = max(time.perf_counter() - t0, 1e-6)
    speed = copied / elapsed
    if total > 0:
        bar = "█" * bar_w
        sys.stdout.write(
            f"\r[{bar}] {100.00:6.2f}% "
            f"{_format_bytes(copied)}/{_format_bytes(total)} "
            f"{_format_bytes(speed)}/s ETA {_format_eta(0)}  {label}\n"
        )
    else:
        sys.stdout.write(
            f"\r{_format_bytes(copied)} copied  {_format_bytes(speed)}/s  {label}\n"
        )
    sys.stdout.flush()


def _maybe_cache_video_locally(video_path: Path) -> Path:
    """Return a local cached copy of a GVFS SMB video (or the original path)."""
    if not bool(getattr(params, "CACHE_NETWORK_VIDEOS_LOCALLY", False)):
        return video_path
    if not _is_gvfs_smb_path(video_path):
        return video_path

    cache_root = getattr(params, "LOCAL_VIDEO_CACHE_DIR", None)
    try:
        cache_root = Path(cache_root).expanduser() if cache_root is not None else None
    except Exception:
        cache_root = None
    if cache_root is None:
        cache_root = Path("/tmp/caged_firefly_pipeline_video_cache")

    key = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    cache_dir = cache_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / video_path.name

    try:
        src_size = int(video_path.stat().st_size)
        if cached_path.exists() and int(cached_path.stat().st_size) == src_size and src_size > 0:
            return cached_path
    except Exception:
        if cached_path.exists():
            return cached_path

    tmp_path = cached_path.with_name(cached_path.name + ".partial")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    print(f"[orchestrator] Caching network video locally → {cached_path}")
    try:
        _copy_file_with_progress(video_path, tmp_path, label=f"Caching {video_path.name}")
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise
    tmp_path.replace(cached_path)
    return cached_path


def _maybe_delete_cached_video(original_path: Path, cached_path: Path) -> None:
    if cached_path == original_path:
        return
    if not bool(getattr(params, "DELETE_CACHED_VIDEOS_AFTER_PROCESSING", False)):
        return
    try:
        cached_path.unlink()
    except Exception:
        return
    try:
        cached_path.parent.rmdir()
    except Exception:
        pass


def _print_stage_times(stage_times: dict[str, float]) -> None:
    keys = ["stage1", "stage2", "stage2_1", "stage3", "stage4"]
    print("\nTiming summary:")
    for k in keys:
        if k == "stage2_1" and not bool(getattr(params, "STAGE2_1_ENABLE", False)):
            continue
        print(f"  {k}: {stage_times.get(k, 0.0):.2f}s")
    print(
        f"  total: {sum(stage_times.get(k, 0.0) for k in keys if not (k == 'stage2_1' and not bool(getattr(params, 'STAGE2_1_ENABLE', False)))):.2f}s"
    )


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


def _export_pipeline_detections_csv(stem: str, stage3_csv: Path, out_dir: Path) -> Path | None:
    """Export a simple x,y,w,h,frame CSV of final pipeline detections (Stage 3 boxes)."""
    try:
        stage3_csv = Path(stage3_csv)
        out_dir = Path(out_dir)
    except Exception:
        return None
    if not stage3_csv.exists():
        return None

    out_root = out_dir / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_detections_xywh.csv"

    total = 0
    frames: set[int] = set()
    with stage3_csv.open("r", newline="") as f_in, out_csv.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["x", "y", "w", "h", "frame"])
        writer.writeheader()
        for row in reader:
            try:
                x = int(round(float(row.get("x", ""))))
                y = int(round(float(row.get("y", ""))))
                w_box = int(round(float(row.get("w", ""))))
                h_box = int(round(float(row.get("h", ""))))
                frame = int(round(float(row.get("frame", row.get("t", "")))))
            except Exception:
                continue
            writer.writerow({"x": x, "y": y, "w": w_box, "h": h_box, "frame": frame})
            total += 1
            frames.add(frame)

    print(
        f"[export] {stem}: wrote detections CSV → {out_csv} "
        f"(boxes={total}, frames_with_boxes={len(frames)})"
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
        local_vid = _maybe_cache_video_locally(vid)
        if local_vid != vid:
            print(f"[orchestrator] Using local cached copy for reads: {local_vid}")
        stage_times: dict[str, float] = {}

        t0 = time.perf_counter()
        s1_csv = stage1_run(local_vid)
        stage_times["stage1"] = time.perf_counter() - t0
        print(f"Stage1  Time: {stage_times['stage1']:.2f}s (csv: {s1_csv.name})")

        # Render raw YOLO detections from Stage1 for debugging/inspection
        try:
            raw_vid = stage1_render_raw_run(local_vid)
            print(f"Stage1_raw  Raw YOLO render: {raw_vid.name}")
        except Exception as e:
            print(f"Stage1_raw  Raw YOLO render skipped due to error: {e}")

        t0 = time.perf_counter()
        s2_csv = stage2_run(local_vid)
        stage_times["stage2"] = time.perf_counter() - t0
        print(f"Stage2  Time: {stage_times['stage2']:.2f}s (csv: {s2_csv.name})")

        # Optional: Stage 2.1 trajectory + intensity hill filter (noise reduction)
        if bool(getattr(params, "STAGE2_1_ENABLE", False)):
            t0 = time.perf_counter()
            s2_1_csv = stage2_1_run(local_vid)
            stage_times["stage2_1"] = time.perf_counter() - t0
            print(f"Stage2_1  Time: {stage_times['stage2_1']:.2f}s (csv: {s2_1_csv.name})")

        t0 = time.perf_counter()
        s3_csv = stage3_run(local_vid)
        stage_times["stage3"] = time.perf_counter() - t0
        print(f"Stage3  Time: {stage_times['stage3']:.2f}s (csv: {s3_csv.name})")

        # Export a simplified x,y,w,h,frame detections CSV (from Stage3 boxes)
        if getattr(params, "RUN_EXPORT_DETECTIONS_CSV", True):
            try:
                export_root = Path(
                    getattr(params, "DIR_DETECTIONS_CSV_OUT", params.ROOT / "detections_csv")
                )
                _export_pipeline_detections_csv(vid.stem, s3_csv, export_root)
            except Exception as e:
                print(f"[export] Warning: failed to export detections CSV for {vid.stem}: {e}")

        t0 = time.perf_counter()
        out_vid = stage4_run(local_vid)
        stage_times["stage4"] = time.perf_counter() - t0
        print(f"Stage4  Time: {stage_times['stage4']:.2f}s (video: {out_vid.name})")

        _summarize_detections(s3_csv, local_vid)
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
                    orig_video_path=local_vid,
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
                    params.DIR_STAGE6_TEST_OUT
                    / f"{vid.stem}_overlay_GT-green_MODEL-red_overlap-yellow.mp4"
                ).resolve()
                stage6_test_overlay_gt_vs_model(
                    orig_video_path=local_vid,
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
                    orig_video_path=local_vid,
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
                    orig_video_path=local_vid,
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
        finally:
            _maybe_delete_cached_video(vid, local_vid)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Save frames from a video for given minute.second ranges.

Time format:
    - Numbers are in MINUTES.SECONDS, where the part after the dot is
      whole seconds in the range 0–59.
      e.g.  0.12  →  0 minutes 12 seconds
            6.44  →  6 minutes 44 seconds
    - Ranges can be written with '-' or 'to', for example:
          0.12 - 0.13
          1.01 to 1.11
          2.36
      A single time like "5.58" means “all frames at minute 5 sec 58”
      (i.e. from 5:58 to 5:59).

All selected frames are written as PNG into a single output folder.
Filenames are:  <video_stem>_<frame_index_padded>.png
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import cv2


# ================= Global Configuration (edit these) =================

# Path to input video
INPUT_VIDEO_PATH = (
    '/Volumes/DL Project SSD/caged fireflies videos original/yolo model training video and data/yolo model dataset video/GH020182.MP4'
)

# Output directory where all PNG frames will be written
OUTPUT_DIR = (
    '/Volumes/DL Project SSD/caged fireflies videos original/yolo model training video and data/selected frames'
)

# Multiline string with your time ranges, one per line.
# Format examples:
#   0.12 - 0.13
#   0.18-0.21
#   1.01 to 1.11
#   2.36
RANGES_TEXT = """
0.12 - 0.13
0.18 - 0.21
0.27 - 0.29
0.43 - 0.44
0.51 - 0.54
0.57 - 0.58
1.01 - 1.11
1.18 - 1.20
1.45 - 1.46
1.53 - 1.54
2.01 - 2.03
2.10 - 2.11
2.20 - 2.21
2.27 - 2.28
2.36
2.46 - 2.47
2.50 - 2.51
2.55 - 2.56
3.11 - 3.12
3.46 - 3.49
3.55 - 3.56
4.11 - 4.12
4.18 - 4.19
4.34 - 4.35
4.57 - 4.58
5.16 - 5.19
5.24 - 5.25
5.33 - 5.39
5.41 - 5.42
5.49 - 5.51
5.58
6.15
6.28
6.37 - 6.38
6.44 - 6.45
""".strip()

# Number of digits used when zero‑padding the frame index in filenames
ZERO_PAD = 6

# If True, do everything except actually writing image files
DRY_RUN = False

# If not None, use this FPS instead of reading from the video metadata.
# Set to e.g. 60.0 if you want to force 60 fps mapping.
FPS_OVERRIDE: float | None = None

# Progress bar settings
BAR_LEN = 50

# =====================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(i: int, total: int, tag: str = "", live: str = "") -> None:
    total = max(1, int(total))
    i = min(max(0, int(i)), total)
    frac = i / total
    fill = int(frac * BAR_LEN)
    bar = "=" * fill + " " * (BAR_LEN - fill)
    sys.stdout.write(f"\r{tag} [{bar}] {int(frac * 100):3d}% {live}")
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write("\n")


@dataclass(frozen=True)
class TimeRange:
    """Inclusive range in whole seconds: [start_sec, end_sec]."""

    start_sec: int
    end_sec: int


def parse_minute_second(token: str) -> int:
    """
    Parse a token like '0.12' or '6.44' into total seconds.

    Interprets token as MINUTES.SECONDS where the part after the dot
    is seconds in [0, 59]. If there is no dot, it is treated as minutes.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty time token")

    if "." in token:
        mins_str, secs_str = token.split(".", 1)
        mins = int(mins_str)
        secs = int(secs_str)
    else:
        mins = int(token)
        secs = 0

    if not (0 <= secs < 60):
        raise ValueError(f"Seconds must be in [0, 59], got {secs} in token '{token}'")

    return mins * 60 + secs


def parse_ranges(text: str) -> List[TimeRange]:
    """
    Parse the multiline RANGES_TEXT into a list of TimeRange objects.

    Lines can be:
        start - end
        start to end
        single
    where start/end are in MINUTES.SECONDS as described above.
    A single time 'T' means [T, T] seconds (i.e. just that one second).
    """
    ranges: List[TimeRange] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        norm = line.lower().replace("to", "-")
        norm = norm.replace("–", "-").replace("—", "-")

        if "-" in norm:
            left, right = norm.split("-", 1)
            start_sec = parse_minute_second(left)
            end_sec = parse_minute_second(right)
        else:
            start_sec = parse_minute_second(norm)
            end_sec = start_sec

        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec

        ranges.append(TimeRange(start_sec=start_sec, end_sec=end_sec))

    return ranges


def ranges_to_frame_indices(
    ranges: Iterable[TimeRange],
    fps: float,
    total_frames: int | None = None,
) -> List[int]:
    """
    Convert second-based ranges to concrete frame indices.

    For each whole second S from start_sec to end_sec (inclusive),
    we include all frames whose timestamps are in [S, S+1).
    """
    if fps <= 0:
        raise ValueError("FPS must be positive")

    indices: Set[int] = set()

    for r in ranges:
        # seconds S in [start_sec, end_sec], inclusive
        for sec in range(r.start_sec, r.end_sec + 1):
            t0 = sec
            t1 = sec + 1
            start_idx = int(round(t0 * fps))
            end_idx_excl = int(round(t1 * fps))

            if total_frames is not None:
                start_idx = max(0, min(start_idx, total_frames))
                end_idx_excl = max(0, min(end_idx_excl, total_frames))

            for idx in range(start_idx, end_idx_excl):
                indices.add(idx)

    return sorted(indices)


def save_selected_frames(
    video_path: Path,
    out_dir: Path,
    frame_indices: List[int],
    dry_run: bool = False,
) -> None:
    if not frame_indices:
        log("No frame indices to save (empty selection).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_set = set(frame_indices)
    last_target = frame_indices[-1]

    log(f"Will attempt to save {len(frame_indices)} unique frames.")
    progress(0, len(frame_indices), tag="extract", live="starting…")

    saved = 0
    seen = 0
    video_stem = video_path.stem

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # CAP_PROP_POS_FRAMES is usually "index of next frame to read"
        # so subtract 1 to get the current frame index.
        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
        if current_idx < 0:
            current_idx = seen

        seen += 1

        if current_idx in target_set:
            fname = f"{video_stem}_{current_idx:0{ZERO_PAD}d}.png"
            fpath = out_dir / fname

            if not dry_run:
                ok_write = cv2.imwrite(str(fpath), frame)
                if not ok_write:
                    log(f"Warning: failed to write frame {current_idx} to {fpath}")
            saved += 1

            elapsed = max(1e-6, time.time() - t0)
            fps_eff = saved / elapsed
            live = f"saved {saved}/{len(frame_indices)} | {fps_eff:.1f} fps"
            progress(saved, len(frame_indices), tag="extract", live=live)

            if current_idx >= last_target:
                # We have reached or passed the last requested frame.
                break

        if total_frames and current_idx >= total_frames - 1:
            break

    cap.release()

    log("=== Summary ===")
    log(f"Requested frames: {len(frame_indices)}")
    log(f"Saved frames    : {saved}")
    log(f"Video frames    : {total_frames}")
    log(f"Output directory: {out_dir}")
    if dry_run:
        log("[DRY RUN] No files were written.")
    log("===============")


def main() -> None:
    vid_path = Path(INPUT_VIDEO_PATH).expanduser()
    out_dir = Path(OUTPUT_DIR).expanduser()

    if not vid_path.exists():
        sys.exit(f"Video file does not exist: {vid_path}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {vid_path}")

    meta_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    fps = float(FPS_OVERRIDE) if FPS_OVERRIDE else meta_fps
    if fps <= 0:
        sys.exit("Could not determine FPS and FPS_OVERRIDE is not set.")

    log(f"Input : {vid_path}")
    log(f"Output: {out_dir}")
    log(f"Meta  : frames={total_frames}, res={width}x{height} @ {fps:.3f} fps")

    ranges = parse_ranges(RANGES_TEXT)
    if not ranges:
        sys.exit("No valid ranges parsed from RANGES_TEXT.")

    log("Parsed ranges (minute.second → seconds):")
    for r in ranges:
        log(f"  {r.start_sec // 60}.{r.start_sec % 60:02d} - {r.end_sec // 60}.{r.end_sec % 60:02d}")

    frame_indices = ranges_to_frame_indices(ranges, fps=fps, total_frames=total_frames or None)
    log(f"Total unique frame indices selected: {len(frame_indices)}")

    if not frame_indices:
        log("Nothing to extract (no indices). Exiting.")
        return

    save_selected_frames(
        video_path=vid_path,
        out_dir=out_dir,
        frame_indices=frame_indices,
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()


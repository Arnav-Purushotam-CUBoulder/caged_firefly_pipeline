#!/usr/bin/env python3
"""
Stage 2: brightest-pixel filter on Stage 1 YOLO detections.

Reads Stage 1 CSVs:
  STAGE1_DIR/<video_stem>/<video_stem>.csv

For each detection (x,y,w,h,frame), extracts the corresponding patch from the
original video frame, computes the maximum luminance using:

  luminance = 0.299*R + 0.587*G + 0.114*B

and drops the detection if max luminance < STAGE2_BRIGHT_MAX_THRESHOLD.

Outputs per-video CSV under:
  STAGE2_DIR/<video_stem>/<video_stem>_bright.csv

with the same schema as Stage 1.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import csv

import cv2
import numpy as np

import params


def _device_for_patch_model():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Stage 2 patch classifier filter.") from e

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_resnet18(num_classes: int = 2):
    import torch.nn as nn
    from torchvision import models

    net = models.resnet18(weights=None)
    old_conv = net.conv1
    net.conv1 = nn.Conv2d(
        3,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    if net.conv1.bias is not None:
        nn.init.zeros_(net.conv1.bias)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net


def _load_patch_model(model_path: Path):
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"Stage2 patch model not found: {model_path}")

    device = _device_for_patch_model()
    model = _build_resnet18(num_classes=2).to(device)

    ckpt = torch.load(str(model_path), map_location=torch.device(device))
    # Handle various checkpoint formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        print(
            f"Stage2  Warning: patch model load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    return model, device


def _open_video(path: Path):
    assert path.exists(), f"Input not found: {path}"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, w, h, fps, count


def _read_stage1(stem: str) -> List[dict]:
    """Read Stage 1 CSV rows for a video."""
    s1_csv = (params.STAGE1_DIR / stem) / f"{stem}.csv"
    rows: List[dict] = []
    with s1_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _group_by_frame(rows: List[dict]) -> Dict[int, List[dict]]:
    by_t: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        try:
            t = int(row["frame"])
        except Exception:
            continue
        by_t[t].append(row)
    return by_t


def run_for_video(video_path: Path) -> Path:
    """Filter Stage 1 detections by brightness + patch classifier; return filtered CSV path."""
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    s1_rows = _read_stage1(stem)
    if not s1_rows:
        print(f"Stage2  NOTE: no Stage1 detections for {video_path.name}")
        out_root = params.STAGE2_DIR / stem
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / f"{stem}_bright.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "x",
                    "y",
                    "w",
                    "h",
                    "frame",
                    "video_name",
                    "firefly_logit",
                    "background_logit",
                ],
            )
            writer.writeheader()
        return out_csv

    by_t = _group_by_frame(s1_rows)

    # Load patch classifier model
    model, device = _load_patch_model(Path(params.STAGE2_PATCH_MODEL_PATH))
    patch_size = int(getattr(params, "STAGE2_PATCH_BOX_SIZE", 40))
    # Choose batch size depending on device
    bs_default = int(getattr(params, "STAGE2_PATCH_BATCH_SIZE", 4096))
    if str(device) == "cpu":
        batch_size = max(1, bs_default // 8)
    else:
        batch_size = bs_default
    pos_thr = float(getattr(params, "STAGE2_PATCH_POSITIVE_THRESHOLD", 0.60))

    cap, W, H, fps, total = _open_video(video_path)
    max_frames = getattr(params, "MAX_FRAMES", None)
    if max_frames is not None:
        limit = min(total, int(max_frames))
    else:
        limit = total

    thr = float(getattr(params, "STAGE2_BRIGHT_MAX_THRESHOLD", 190.0))

    out_root = params.STAGE2_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / f"{stem}_bright.csv"

    kept_rows: List[dict] = []
    bright_kept = 0
    bright_dropped = 0
    cnn_kept = 0
    cnn_dropped = 0

    # Accumulate patches for CNN in mini-batches
    pending_patches: List[np.ndarray] = []
    pending_rows: List[dict] = []

    def _flush_batch():
        nonlocal pending_patches, pending_rows, kept_rows, cnn_kept, cnn_dropped
        if not pending_patches:
            return
        import torch

        arr = np.stack(pending_patches, axis=0)  # (N, H, W, 3), uint8
        batch = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            pos_probs = probs[:, 1].detach().cpu().numpy()
        for row, p_pos in zip(pending_rows, pos_probs):
            if float(p_pos) >= pos_thr:
                kept_rows.append(row)
                cnn_kept += 1
            else:
                cnn_dropped += 1
        pending_patches.clear()
        pending_rows.clear()

    try:
        for t in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            dets = by_t.get(t, [])
            if not dets:
                continue
            # Convert once per frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Hf, Wf = frame_rgb.shape[:2]
            for row in dets:
                try:
                    x = int(float(row["x"]))
                    y = int(float(row["y"]))
                    w = int(float(row["w"]))
                    h = int(float(row["h"]))
                except Exception:
                    bright_dropped += 1
                    continue
                if w <= 0 or h <= 0:
                    bright_dropped += 1
                    continue
                # Clip to frame bounds
                x0 = max(0, min(x, Wf - 1))
                y0 = max(0, min(y, Hf - 1))
                x1 = max(0, min(x0 + w, Wf))
                y1 = max(0, min(y0 + h, Hf))
                if x1 <= x0 or y1 <= y0:
                    bright_dropped += 1
                    continue
                patch = frame_rgb[y0:y1, x0:x1]
                if patch.size == 0:
                    bright_dropped += 1
                    continue
                # Luminance = 0.299 R + 0.587 G + 0.114 B
                lumin = (
                    0.299 * patch[:, :, 0]
                    + 0.587 * patch[:, :, 1]
                    + 0.114 * patch[:, :, 2]
                )
                max_val = float(lumin.max()) if lumin.size else 0.0
                if max_val < thr:
                    bright_dropped += 1
                    continue
                # Brightness passed; enqueue for CNN
                bright_kept += 1
                patch_resized = cv2.resize(
                    patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA
                )
                pending_patches.append(patch_resized)
                pending_rows.append(row)
                if len(pending_patches) >= batch_size:
                    _flush_batch()
    finally:
        cap.release()

    # Flush remaining CNN batch
    _flush_batch()

    # Write filtered CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x",
                "y",
                "w",
                "h",
                "frame",
                "video_name",
                "firefly_logit",
                "background_logit",
            ],
        )
        writer.writeheader()
        writer.writerows(kept_rows)

    total_in = len(s1_rows)
    print(
        f"Stage2  Video: {video_path.name}; detections_in={total_in}; "
        f"bright_kept={bright_kept}; bright_dropped={bright_dropped}; "
        f"cnn_kept={cnn_kept}; cnn_dropped={cnn_dropped}; "
        f"thr_bright={thr}; thr_patch={pos_thr}"
    )
    return out_csv


__all__ = ["run_for_video"]

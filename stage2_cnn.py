"""
Stage 2: CNN classification of SBD detections.

Reads Stage1 CSVs (frame,cx,cy,size) per video, iterates video frames, extracts
BOX_SIZE_PX x BOX_SIZE_PX RGB crops centered at each centroid (with
zero padding for out-of-bounds), and runs a ResNet18 classifier.

Outputs Stage2 CSVs with: frame,cx,cy,size,pred,conf
where pred in {0,1} and conf is probability for class 1.
"""
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models

import params


def build_resnet18(num_classes: int = 2) -> nn.Module:
    net = models.resnet18(weights=None)
    old_conv = net.conv1
    net.conv1 = nn.Conv2d(
        3, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    if net.conv1.bias is not None:
        nn.init.zeros_(net.conv1.bias)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net


def _extract_crop(frame_rgb: np.ndarray, cx: float, cy: float, box_size: int) -> np.ndarray:
    half = box_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + box_size
    y2 = y1 + box_size
    h, w = frame_rgb.shape[:2]
    crop = np.zeros((box_size, box_size, 3), dtype=frame_rgb.dtype)

    sx0 = max(x1, 0); sy0 = max(y1, 0)
    sx1 = min(x2, w); sy1 = min(y2, h)
    dx0 = sx0 - x1; dy0 = sy0 - y1
    dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)
    if dx1 > dx0 and dy1 > dy0:
        crop[dy0:dy1, dx0:dx1] = frame_rgb[sy0:sy1, sx0:sx1]
    return crop


def _load_stage1(path: Path) -> Dict[int, List[Tuple[float, float, float]]]:
    """Return mapping: frame_idx -> list of (cx, cy, size)."""
    mapping: Dict[int, List[Tuple[float, float, float]]] = {}
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            cx = float(row['cx']); cy = float(row['cy']); size = float(row['size'])
            mapping.setdefault(fi, []).append((cx, cy, size))
    return mapping


def run_stage2() -> Path:
    """Run CNN classification for each Stage1 detection and write Stage2 CSVs per video."""
    stage1_dir = params.STAGE1_DIR
    out_dir = params.STAGE2_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map video stem -> video path
    videos = params.list_videos()
    vid_map = {p.stem: p for p in videos}

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    model = build_resnet18().to(device)
    ckpt = torch.load(params.MODEL_PATH, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # All crops are created at BOX_SIZE_PX already; no resizing/ToTensor needed.

    for csv_path in sorted(stage1_dir.glob('*_detections.csv')):
        stem = csv_path.stem.replace('_detections', '')
        vpath = vid_map.get(stem)
        if vpath is None:
            print(f"Warning: no matching video for {csv_path.name}")
            continue
        dets = _load_stage1(csv_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        out_csv = out_dir / f"{stem}_classified.csv"
        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'cx', 'cy', 'size', 'pred', 'conf'])
            with tqdm(total=total, desc=f'Stage2 CNN: {vpath.name}', unit='frame', ncols=80) as bar:
                for idx in range(total):
                    ok, frame_bgr = cap.read()
                    bar.update(1)
                    if not ok:
                        break
                    if idx not in dets:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # Build crops for this frame
                    det_list = dets[idx]
                    crops = [
                        _extract_crop(frame_rgb, cx, cy, params.BOX_SIZE_PX)
                        for (cx, cy, size) in det_list
                    ]
                    if not crops:
                        continue
                    # Stack into a single tensor batch (N, 3, H, W) in [0,1]
                    arr = np.stack(crops, axis=0)  # (N, H, W, 3), uint8
                    batch = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)
                    # Process in sub-batches if needed
                    bs = int(params.BATCH_SIZE) if params.BATCH_SIZE else len(det_list)
                    with torch.no_grad():
                        for start in range(0, batch.size(0), bs):
                            end = min(start + bs, batch.size(0))
                            b = batch[start:end].to(device)
                            logits = model(b)
                            probs = torch.softmax(logits, dim=1)
                            confs = probs[:, 1].detach().cpu().tolist()
                            preds = probs.argmax(dim=1).detach().cpu().tolist()
                            for off, (pred, conf) in enumerate(zip(preds, confs)):
                                i = start + off
                                cx, cy, size = det_list[i]
                                writer.writerow([idx, cx, cy, size, int(pred), float(conf)])
        cap.release()

    return out_dir


__all__ = ['run_stage2']

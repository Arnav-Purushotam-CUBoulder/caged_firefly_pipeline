#!/usr/bin/env python3
"""
firefly_inference.py  –  crops are now always 40×40, even at the borders,
and feeds full-color crops into the model.
"""

# ─── GLOBAL CONSTANTS ──────────────────────────────────────────
INPUT_VIDEO   = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/caged pyrallis inference data/original videos/GH010181.MP4'
OUTPUT_VIDEO  = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/caged pyrallis inference data/original videos/OP_GH010181.MP4'

FRAMES_DIR    = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/caged pyrallis inference data/original videos/GH010181_firefly_frames'
CROPS_DIR     = r'/Users/arnavps/Desktop/New DL project data to transfer to external disk/caged pyrallis inference data/original videos/GH010181_firefly_crops'

MODEL_PATH    = r'/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/caged_fireflies/models and other data/colo_real_dataset_ResNet18_best_model.pt'

THRESHOLD_Y   = 175
OVERLAP_IOU   = 0.25
BOX_SIZE_PX   = 40
BOX_COLOR_BGR = (0, 0, 255)
BOX_THICKNESS = 1
FOURCC        = "mp4v"
N             = 100       # None → process full video

CONFIDENCE_MIN = 0.98      # firefly class probability threshold (0‒1)

# ─── imports ───────────────────────────────────────────────────
import cv2, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─── model helper ──────────────────────────────────────────────
def build_resnet18(num_classes: int = 2) -> nn.Module:
    net = models.resnet18(weights=None)
    # adapt first conv to 3-channel input
    old_conv = net.conv1
    net.conv1 = nn.Conv2d(
        3, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
    if net.conv1.bias is not None:
        nn.init.zeros_(net.conv1.bias)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    nn.init.xavier_uniform_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
    return net

# ─── detector helper ───────────────────────────────────────────
def make_blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.filterByColor, p.blobColor = True, 255
    p.minThreshold, p.maxThreshold, p.thresholdStep = THRESHOLD_Y, 255, 1
    p.minDistBetweenBlobs = 1
    p.filterByArea = p.filterByCircularity = p.filterByConvexity = p.filterByInertia = False
    return cv2.SimpleBlobDetector_create(p)

# ─── IoU merge helpers ────────────────────────────────────────
_HALF = BOX_SIZE_PX / 2.0
def box_from_centroid(cx, cy):
    return np.array([cx - _HALF, cy - _HALF, cx + _HALF, cy + _HALF], float)

def iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter)

def merge_boxes_iterative(cents):
    boxes = [box_from_centroid(x, y) for x, y in cents]
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(boxes):
            base, grp = boxes[i], [boxes[i]]
            j = i + 1
            while j < len(boxes):
                if iou(base, boxes[j]) >= OVERLAP_IOU:
                    grp.append(boxes.pop(j))
                    changed = True
                else:
                    j += 1
            xs = [(b[0]+b[2])/2 for b in grp]
            ys = [(b[1]+b[3])/2 for b in grp]
            new.append(box_from_centroid(sum(xs)/len(xs), sum(ys)/len(ys)))
            i += 1
        boxes = new
    return boxes

# ─── new: always-valid crop extractor ──────────────────────────
def extract_crop(frame_rgb, box):
    """
    Return a BOX_SIZE_PX×BOX_SIZE_PX RGB crop. Out-of-bounds areas are padded with zeros.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame_rgb.shape[:2]

    crop = np.zeros((BOX_SIZE_PX, BOX_SIZE_PX, 3), dtype=frame_rgb.dtype)

    sx0 = max(x1, 0); sy0 = max(y1, 0)
    sx1 = min(x2, w); sy1 = min(y2, h)
    dx0 = sx0 - x1; dy0 = sy0 - y1
    dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)

    crop[dy0:dy1, dx0:dx1] = frame_rgb[sy0:sy1, sx0:sx1]
    return crop

# ─── robust writer ─────────────────────────────────────────────
def safe_imwrite(path: Path, img):
    try:
        if cv2.imwrite(str(path), img):
            return True
    except:
        pass
    try:
        ret, buf = cv2.imencode('.png', img)
        if ret:
            path.write_bytes(buf.tobytes())
            return True
    except:
        pass
    print(f"⚠️  Could not write {path}", file=sys.stderr)
    return False

# ─── main ──────────────────────────────────────────────────────
def main():
    for p in (Path(OUTPUT_VIDEO).parent, Path(FRAMES_DIR), Path(CROPS_DIR)):
        p.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model'])
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((BOX_SIZE_PX, BOX_SIZE_PX)),
        transforms.ToTensor()
    ])

    detector = make_blob_detector()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open {INPUT_VIDEO!r}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total, N) if N else total
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*FOURCC),
        fps,
        (w, h),
        True
    )

    det_total = firefly_total = 0

    with tqdm(total=total, desc="Inferring", unit="frame", ncols=80) as bar:
        for idx in range(total):
            ok, frame_bgr = cap.read()
            bar.update(1)
            if not ok: break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            cents = [(kp.pt[0], kp.pt[1]) for kp in detector.detect(
                     cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))]
            boxes = merge_boxes_iterative(cents)
            det_total += len(boxes)

            draw_frame = frame_bgr.copy()

            for box in boxes:
                crop_rgb = extract_crop(frame_rgb, box)
                t = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = model(t).softmax(dim=1)
                conf = probs[0,1].item()
                pred = probs.argmax(1).item()
                if pred == 1 and conf >= CONFIDENCE_MIN:
                    firefly_total += 1
                    conf_pct = int(round(conf*100))
                    x1, y1, _, _ = map(int, box)
                    fname = f"{x1}_{y1}_{BOX_SIZE_PX}_{BOX_SIZE_PX}_frame_{idx:06d}_firefly_{conf_pct}.png"
                    safe_imwrite(Path(CROPS_DIR)/fname,
                                 cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                    cv2.rectangle(draw_frame,
                                  (x1, y1),
                                  (x1+BOX_SIZE_PX, y1+BOX_SIZE_PX),
                                  BOX_COLOR_BGR,
                                  BOX_THICKNESS)

            writer.write(draw_frame)
            safe_imwrite(Path(FRAMES_DIR)/f"frame_{idx:06d}.png", draw_frame)

    cap.release()
    writer.release()

    print("\n───── Inference complete ─────")
    print(f"Frames processed:         {total}")
    print(f"Total merged detections:  {det_total}")
    print(f"Fireflies detected:       {firefly_total}")
    print(f"Annotated video →        {OUTPUT_VIDEO}")
    print(f"Frames in        →        {FRAMES_DIR}")
    print(f"Crops in         →        {CROPS_DIR}")

if __name__ == "__main__":
    main()

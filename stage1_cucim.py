"""
Stage 1 (cuCIM): GPU blob detections with cuCIM/CuPy.

Mimics the existing SimpleBlobDetector stage by outputting per-frame
centroids and a size value. The output CSV schema matches stage1_sbd:
  frame,cx,cy,size

Notes:
- Applies a simple grayscale intensity mask using params.THRESHOLD_Y so
  only bright pixels contribute to detection (similar to SBD behavior).
- Uses cuCIM's blob_log on a GPU float image in [0,1].
- If cuCIM/CuPy are unavailable, raises a clear ImportError.
"""
from pathlib import Path
import csv
from typing import Tuple
import sys
import os
import ctypes

import cv2
import numpy as np
from tqdm import tqdm

import params

# Encourage stable NVRTC behavior; do not touch global system installs.
os.environ.setdefault('CUPY_DISABLE_JITIFY', '0')
os.environ.setdefault('CUPY_JITIFY_ENABLE', '1')
os.environ.setdefault('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', '1')

try:
    import cupy as cp
    from cucim.skimage.feature import blob_log
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Stage-1 cuCIM detector requires CuPy and cuCIM. Install e.g. 'pip install cupy-cuda12x cucim'."
    ) from exc


def _ensure_nvrtc_loaded():
    """Ensure libnvrtc is loaded before CuPy JIT compiles kernels.
    Does not modify system installs; only attempts to dlopen known locations.
    """
    try:
        ctypes.CDLL('libnvrtc.so.12')
        return
    except OSError:
        pass
    try:
        import nvidia.cuda_nvrtc as cuda_nvrtc  # from pip package if present
    except Exception:
        return  # best-effort only; fall back to system resolution
    base_dir = getattr(cuda_nvrtc, '__file__', None)
    if base_dir is None:
        base_dir = list(getattr(cuda_nvrtc, '__path__', []))[0]
    base = Path(base_dir).resolve()
    if base.name != 'lib':
        base = base / 'lib'
    candidates = list(base.glob('libnvrtc.so*')) + list(base.glob('libnvrtc-builtins.so*'))
    for cand in candidates:
        try:
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


_ensure_nvrtc_loaded()

# Patch CuPy NVRTC compile flags to use C++17 and sane includes without touching system files
try:
    from cupy.cuda import compiler as _cupy_compiler

    _orig_compile = getattr(_cupy_compiler, 'compile_using_nvrtc', None)
    if _orig_compile is not None and not getattr(_cupy_compiler, '_caged_firefly_nvrtc_patch', False):
        def _compile_using_nvrtc_cxx17(source, options=(), *args, **kwargs):
            # ensure c++17 and avoid forcing JITIFY via macro
            def _ok(opt: str) -> bool:
                if not isinstance(opt, str):
                    return True
                if opt == '-DCUPY_USE_JITIFY':
                    return False
                return True
            opts = []
            has_std = False
            for opt in options:
                if not _ok(opt):
                    continue
                if opt in ('-std=c++11', '--std=c++11'):
                    opts.append('--std=c++17')
                    has_std = True
                elif isinstance(opt, str) and (opt.startswith('-std=') or opt.startswith('--std=')):
                    opts.append(opt)
                    has_std = True
                else:
                    opts.append(opt)
            if not has_std:
                opts.append('--std=c++17')
            # Add libcudacxx include shipped within CuPy to satisfy cuda/std headers
            try:
                _root = Path(cp.__file__).resolve().parent
                _libcxx_inc = _root / '_core/include/cupy/_cccl/libcudacxx/include'
                if _libcxx_inc.is_dir():
                    flag = f'-I{str(_libcxx_inc)}'
                    if flag not in opts:
                        opts.append(flag)
            except Exception:
                pass
            # Helpful defines to quiet dialect warnings
            for define in ('-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT=1', '-D__CUDACC_RTC__=1'):
                if define not in opts:
                    opts.append(define)
            try:
                return _orig_compile(source, tuple(opts), *args, **kwargs)
            except Exception:
                # Retry once without JITIFY positional bool if present
                try:
                    if args and isinstance(args[-1], bool):
                        args = tuple(args[:-1])
                    kwargs = dict(kwargs)
                    kwargs.pop('jitify', None)
                except Exception:
                    pass
                return _orig_compile(source, tuple(opts), *args, **kwargs)

        _cupy_compiler.compile_using_nvrtc = _compile_using_nvrtc_cxx17
        _cupy_compiler._caged_firefly_nvrtc_patch = True
except Exception:
    pass


def _sigma_to_radius_log(sigma_val: float, log_scale: bool) -> float:
    # For LoG, if log_scale=True, radius ~ sigma*sqrt(2); else ~ sigma
    return float(sigma_val) * (np.sqrt(2.0) if log_scale else 1.0)


def run_stage1() -> Path:
    out_dir = params.STAGE1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = params.list_videos()

    # cuCIM settings from params with sensible defaults
    detector_kind = str(getattr(params, 'CUCIM_DETECTOR', 'log')).lower()
    min_sigma = float(getattr(params, 'CUCIM_MIN_SIGMA', 1.0))
    max_sigma = float(getattr(params, 'CUCIM_MAX_SIGMA', 4.0))
    num_sigma = int(getattr(params, 'CUCIM_NUM_SIGMA', 8))
    log_scale = bool(getattr(params, 'CUCIM_LOG_SCALE', False))
    overlap = float(getattr(params, 'CUCIM_OVERLAP', 0.5))
    thr_resp = float(getattr(params, 'CUCIM_THRESHOLD_RESP', 0.05))
    downscale = int(getattr(params, 'CUCIM_DOWNSCALE', 1))

    for vpath in vids:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Warning: cannot open video {vpath}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(total, params.N) if params.N else total
        out_csv = out_dir / f"{vpath.stem}_detections.csv"

        total_dets = 0
        with out_csv.open('w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['frame', 'cx', 'cy', 'size'])
            batch_size = max(1, int(getattr(params, 'CUCIM_BATCH_FRAMES', 1)))
            with tqdm(total=total, desc=f'Stage1 cuCIM: {vpath.name}', unit='frame', dynamic_ncols=True) as bar:
                idx = 0
                while idx < total:
                    # Load CPU batch
                    frames_cpu = []
                    frame_indices = []
                    to_read = min(batch_size, total - idx)
                    for _ in range(to_read):
                        ok, frame_bgr = cap.read()
                        bar.update(1)
                        if not ok:
                            break
                        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        if downscale and downscale > 1:
                            gray = cv2.resize(
                                gray,
                                (gray.shape[1] // downscale, gray.shape[0] // downscale),
                                interpolation=cv2.INTER_AREA,
                            )
                        frames_cpu.append(gray)
                        frame_indices.append(idx)
                        idx += 1
                    if not frames_cpu:
                        break
                    thr = int(getattr(params, 'THRESHOLD_Y', 175))
                    # Build a single numpy stack and zero below-threshold on CPU
                    masks = [(g >= thr) for g in frames_cpu]
                    arr = np.stack(frames_cpu, axis=0).astype(np.float32) / 255.0
                    for i, m in enumerate(masks):
                        if not m.any():
                            arr[i, ...] *= 0.0
                        else:
                            arr[i, ~m] = 0.0
                    gpu_batch = cp.asarray(arr, dtype=cp.float32)
                    # Detect per slice on GPU
                    if detector_kind == 'dog':
                        from cucim.skimage.feature import blob_dog as _blob
                    else:
                        from cucim.skimage.feature import blob_log as _blob
                    for bi in range(gpu_batch.shape[0]):
                        gpu_img = gpu_batch[bi]
                        if detector_kind == 'dog':
                            blobs = _blob(
                                gpu_img,
                                min_sigma=float(min_sigma),
                                max_sigma=float(max_sigma),
                                sigma_ratio=float(1.6),
                                threshold=float(thr_resp),
                                overlap=float(overlap),
                            )
                        else:
                            blobs = _blob(
                                gpu_img,
                                min_sigma=float(min_sigma),
                                max_sigma=float(max_sigma),
                                num_sigma=int(max(1, num_sigma)),
                                threshold=float(thr_resp),
                                overlap=float(overlap),
                                log_scale=bool(log_scale),
                            )
                        if blobs.size == 0:
                            continue
                        blobs_np = cp.asnumpy(blobs)
                        fidx = frame_indices[bi]
                        for (yy, xx, ss) in blobs_np:
                            r = _sigma_to_radius_log(float(ss), log_scale) if detector_kind != 'dog' else float(ss) * np.sqrt(2.0)
                            size = max(1.0, 2.0 * r)
                            if downscale and downscale > 1:
                                cx = float(xx) * downscale
                                cy = float(yy) * downscale
                                sz = float(size) * downscale
                            else:
                                cx = float(xx)
                                cy = float(yy)
                                sz = float(size)
                            writer.writerow([fidx, cx, cy, sz])
                            total_dets += 1
        cap.release()
        print(f"Stage1: {vpath.stem} (cuCIM) frames={total} dets={total_dets} dets_per_frame={total_dets/max(1,total):.2f}")

    return out_dir


__all__ = ['run_stage1']

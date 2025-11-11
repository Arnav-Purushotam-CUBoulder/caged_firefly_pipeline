"""
Stage 0: Cleanup working tree in PIPELINE_ROOT.

Behavior:
- Keeps only required input folders for the pipeline + test suite:
  • 'original videos' (params.ORIGINAL_VIDEOS_DIR)
  • 'ground truth' (parent of params.GT_CSV_PATH, when under PIPELINE_ROOT)
- Deletes everything else under PIPELINE_ROOT (outputs, artifacts, temp files).
- Recreates output directories for all pipeline stages and test stages:
  • STAGE0_DIR … STAGE4_DIR, STAGE_RENDERER_DIR
  • DIR_STAGE5_TEST_OUT, DIR_STAGE6_TEST_OUT, DIR_STAGE9_TEST_OUT
"""
from pathlib import Path
import shutil

import params


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def run_stage0() -> Path:
    root = params.PIPELINE_ROOT
    root.mkdir(parents=True, exist_ok=True)

    # Determine required input directories to preserve
    keep_dirs = set()

    # Always keep original videos folder (create if missing)
    if _is_under(params.ORIGINAL_VIDEOS_DIR, root):
        keep_dirs.add(params.ORIGINAL_VIDEOS_DIR.name)
        params.ORIGINAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep ground truth folder(s) if they live under PIPELINE_ROOT
    # Preferred folder name: 'ground truth csv folder'
    gt_dir = getattr(params, 'GT_CSV_DIR', None)
    if isinstance(gt_dir, Path) and _is_under(gt_dir, root):
        keep_dirs.add(gt_dir.name)
        gt_dir.mkdir(parents=True, exist_ok=True)
    # Fallback: keep parent of single GT_CSV_PATH if provided
    gt_path = getattr(params, 'GT_CSV_PATH', None)
    if gt_path is not None:
        gt_parent = Path(gt_path).parent
        if _is_under(gt_parent, root):
            keep_dirs.add(gt_parent.name)
            gt_parent.mkdir(parents=True, exist_ok=True)

    # Remove everything in root that's not an essential input directory or a required input file
    for entry in root.iterdir():
        try:
            if entry.is_dir():
                if entry.name not in keep_dirs:
                    shutil.rmtree(entry, ignore_errors=True)
            elif entry.is_file():
                # Preserve nothing at root by default (inputs are expected in subfolders)
                try:
                    entry.unlink()
                except Exception:
                    pass
        except Exception:
            # Best-effort cleanup; continue
            continue

    # Recreate pipeline output directories
    out_dirs = [
        params.STAGE0_DIR,
        params.STAGE1_DIR,
        params.STAGE2_DIR,
        params.STAGE3_DIR,
        params.STAGE4_DIR,
        getattr(params, 'STAGE4_1_DIR', params.PIPELINE_ROOT / 'stage4_1_outputs'),
        getattr(params, 'STAGE4_2_DIR', params.PIPELINE_ROOT / 'stage4_2_outputs'),
        params.STAGE_RENDERER_DIR,
    ]

    # Recreate test output directories if defined in params
    for attr in ('DIR_STAGE5_TEST_OUT', 'DIR_STAGE6_TEST_OUT', 'DIR_STAGE9_TEST_OUT'):
        d = getattr(params, attr, None)
        if isinstance(d, Path):
            out_dirs.append(d)

    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    # Write a cleanup marker
    try:
        (params.STAGE0_DIR / 'CLEANED').write_text('stage0 cleanup completed')
    except Exception:
        pass

    return params.STAGE0_DIR


__all__ = ['run_stage0']

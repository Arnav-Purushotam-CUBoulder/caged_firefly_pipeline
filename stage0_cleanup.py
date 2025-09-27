"""
Stage 0: Cleanup outputs.

Deletes all folders in PIPELINE_ROOT except 'original videos', then
re-creates the stage output directories.
"""
from pathlib import Path
import shutil

import params


def run_stage0() -> Path:
    root = params.PIPELINE_ROOT
    keep = {'original videos'}
    root.mkdir(parents=True, exist_ok=True)

    # Remove all directories except 'original videos'
    for entry in root.iterdir():
        if entry.is_dir() and entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)

    # Recreate stage outputs
    for d in (params.STAGE0_DIR, params.STAGE1_DIR, params.STAGE2_DIR, params.STAGE3_DIR, params.STAGE4_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Optionally record a marker file
    (params.STAGE0_DIR / 'CLEANED').write_text('stage0 cleanup completed')
    return params.STAGE0_DIR


__all__ = ['run_stage0']


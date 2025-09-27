"""
Pipeline orchestrator: runs Stage0 → Stage1 → Stage2 → Stage3 → Stage4.

Usage: python "models and other data/code/orchestrator.py"
All configuration is in params.py.
"""
from pathlib import Path
from time import perf_counter

import params
from stage0_cleanup import run_stage0
from stage1_sbd import run_stage1
from stage2_cnn import run_stage2
from stage3_merge import run_stage3
from stage4_render import run_stage4


def main():
    t0 = perf_counter()
    print("── Running Stage 0: Cleanup …")
    s0 = run_stage0()
    print(f"Stage0 outputs → {s0}")
    print("── Running Stage 1: SBD detections …")
    s1 = run_stage1()
    print(f"Stage1 outputs → {s1}")

    print("── Running Stage 2: CNN classification …")
    s2 = run_stage2()
    print(f"Stage2 outputs → {s2}")

    print("── Running Stage 3: Merge boxes …")
    s3 = run_stage3()
    print(f"Stage3 outputs → {s3}")

    print("── Running Stage 4: Render + Final CSV …")
    s4 = run_stage4()
    print(f"Stage4 outputs → {s4}")

    dt = perf_counter() - t0
    print(f"Done in {dt:.1f}s")


if __name__ == '__main__':
    main()

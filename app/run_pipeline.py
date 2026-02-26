"""
run_pipeline.py — Run the full raga recognition pipeline (stages 1-6).

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --from-stage 5     # resume from stage 5
    python run_pipeline.py --quick            # dev run (10 segments/raga)
    python run_pipeline.py --skip-lstm        # skip LSTM in stage 6
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


STAGES = [
    {
        "num": 1,
        "name": "Preprocessing",
        "script": "carnatic_scripts/01_preprocess.py",
        "args": [],
    },
    {
        "num": 2,
        "name": "Pitch Extraction",
        "script": "carnatic_scripts/02_pitch_baseline.py",
        "args": [],
        "quick_args": ["--max-per-raga", "10"],
    },
    {
        "num": 3,
        "name": "Tonic Estimation",
        "script": "carnatic_scripts/03_tonic_baseline.py",
        "args": [],
    },
    {
        "num": 4,
        "name": "Swara Mapping",
        "script": "carnatic_scripts/04_swara_mapping.py",
        "args": [],
    },
    {
        "num": 5,
        "name": "Feature Extraction",
        "script": "carnatic_scripts/05_feature_extraction.py",
        "args": [],
    },
    {
        "num": 6,
        "name": "Classification",
        "script": "carnatic_scripts/06_train_classifiers.py",
        "args": [],
    },
]


def run_stage(stage: dict, extra_args: list = None, dry_run: bool = False):
    script = stage["script"]
    args = stage["args"] + (extra_args or [])
    cmd = [sys.executable, script] + args

    print(f"\n{'='*60}")
    print(f"  STAGE {stage['num']}: {stage['name']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    elapsed = time.time() - start

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Stage {stage['num']} finished in {mins}m {secs}s "
          f"(exit code {result.returncode})")

    if result.returncode != 0:
        print(f"  ERROR: Stage {stage['num']} failed. Stopping pipeline.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full raga recognition pipeline")
    parser.add_argument("--from-stage", type=int, default=1,
                        help="Start from this stage (default: 1)")
    parser.add_argument("--to-stage", type=int, default=6,
                        help="Stop after this stage (default: 6)")
    parser.add_argument("--quick", action="store_true",
                        help="Dev run: limit segments per raga")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training in stage 6")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--min-segments-per-raga", type=int, default=None,
                        help="Pass to stage 6 (e.g. 60 for 10-raga run)")
    args = parser.parse_args()

    print("RAGA RECOGNITION PIPELINE")
    print(f"Stages: {args.from_stage} -> {args.to_stage}")
    if args.quick:
        print("Mode: QUICK (limited segments)")
    print()

    total_start = time.time()

    for stage in STAGES:
        if stage["num"] < args.from_stage or stage["num"] > args.to_stage:
            continue

        extra_args = []

        if args.quick and "quick_args" in stage:
            extra_args.extend(stage["quick_args"])

        if stage["num"] == 6:
            if args.skip_lstm:
                extra_args.append("--skip-lstm")
            if args.min_segments_per_raga is not None:
                extra_args.extend(["--min-segments-per-raga",
                                   str(args.min_segments_per_raga)])

        ok = run_stage(stage, extra_args, dry_run=args.dry_run)
        if not ok:
            sys.exit(1)

    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — total time: {mins}m {secs}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

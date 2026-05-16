from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train contract-level OGPO frequency-severity models and build a submission."
    )
    parser.add_argument("--train", default="dataset/train.csv", help="Path to train.csv.")
    parser.add_argument("--test", default="dataset/test.csv", help="Path to test.csv.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for models and metrics.")
    parser.add_argument("--submission", default="submissions/submission.csv", help="Submission output path.")
    parser.add_argument("--train-rows", type=int, default=None, help="Read only the first N train rows for smoke tests.")
    parser.add_argument("--test-rows", type=int, default=None, help="Read only the first N test rows for smoke tests.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--rare-min-count", type=int, default=50)
    parser.add_argument("--severity-min-claims", type=int, default=50)
    parser.add_argument(
        "--progress-period",
        type=int,
        default=50,
        help="Print boosting progress every N iterations.",
    )
    parser.add_argument(
        "--tune-severity",
        action="store_true",
        help="Enable CatBoost severity tuning on the validation split.",
    )
    parser.add_argument(
        "--severity-trials",
        type=int,
        default=30,
        help="Number of random-search trials for CatBoost severity tuning.",
    )
    parser.add_argument(
        "--severity-time-budget",
        type=int,
        default=1800,
        help="Maximum tuning time for severity model, in seconds.",
    )
    parser.add_argument(
        "--severity-objective",
        choices=["rmse", "rmse_r2"],
        default="rmse_r2",
        help="Severity tuning objective. rmse_r2 blends RMSE with R2 on positive claims.",
    )
    parser.add_argument(
        "--severity-r2-weight",
        type=float,
        default=100000.0,
        help="Weight for R2 penalty in rmse_r2 objective.",
    )
    parser.add_argument(
        "--severity-target",
        choices=["amount", "claim_per_premium"],
        default="claim_per_premium",
        help="Severity target. claim_per_premium predicts claim_amount relative to premium.",
    )
    parser.add_argument(
        "--disable-severity-calibration",
        action="store_true",
        help="Disable validation-fitted affine calibration of severity predictions.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    parser.add_argument(
        "--model-backend",
        choices=["auto", "sklearn", "lightgbm", "catboost"],
        default="auto",
        help="Model backend to train. auto prefers CatBoost, then LightGBM, then sklearn.",
    )
    parser.add_argument(
        "--force-sklearn",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

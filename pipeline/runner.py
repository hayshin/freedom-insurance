from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.contracts import build_contract_frame
from pipeline.evaluation import best_f1_threshold, evaluate
from pipeline.features import align_features, apply_rare_categories, build_feature_lists, split_train_valid
from pipeline.io import frame_len, read_input, save_pickle
from pipeline.logging import log_stage, log_stage_done
from pipeline.models import (
    predict_frequency,
    predict_severity,
    resolve_model_backend,
    train_frequency_model,
    train_severity_model,
)
from pipeline.pricing import apply_pricing, calibrate_pricing


def run_pipeline(args) -> None:
    show_progress = not args.quiet
    model_backend = resolve_model_backend(args.model_backend, args.force_sklearn)
    artifacts_dir = Path(args.artifacts_dir)
    submission_path = Path(args.submission)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = log_stage(1, f"read train/test CSV ({args.train}, {args.test})", enabled=show_progress)
    raw_train = read_input(args.train, args.train_rows)
    raw_test = read_input(args.test, args.test_rows)
    log_stage_done(
        started_at,
        f"loaded {frame_len(raw_train):,} train rows and {frame_len(raw_test):,} test rows",
        enabled=show_progress,
    )

    started_at = log_stage(
        2,
        "aggregate train rows to contract level and build features (polars)",
        enabled=show_progress,
    )
    train_contracts = build_contract_frame(raw_train, is_train=True)
    log_stage_done(started_at, f"built {len(train_contracts):,} train contracts", enabled=show_progress)

    started_at = log_stage(
        3,
        "aggregate test rows to contract level and build features (polars)",
        enabled=show_progress,
    )
    test_contracts = build_contract_frame(raw_test, is_train=False)
    log_stage_done(started_at, f"built {len(test_contracts):,} test contracts", enabled=show_progress)

    started_at = log_stage(4, "split train/validation contracts", enabled=show_progress)
    train_part, valid_part = split_train_valid(train_contracts, args.valid_size, args.random_state)
    full_feature_columns, numeric_columns, categorical_columns = build_feature_lists(train_part)
    log_stage_done(
        started_at,
        (
            f"train={len(train_part):,}, valid={len(valid_part):,}, "
            f"features={len(full_feature_columns):,}, categorical={len(categorical_columns):,}"
        ),
        enabled=show_progress,
    )

    started_at = log_stage(5, "align train/valid/test feature matrices", enabled=show_progress)
    train_x = train_part[full_feature_columns].copy()
    valid_x = valid_part[full_feature_columns].copy()
    test_x = test_contracts[[col for col in full_feature_columns if col in test_contracts.columns]].copy()
    train_x, test_x = align_features(train_x, test_x)
    valid_x = valid_x[train_x.columns]

    categorical_columns = [col for col in categorical_columns if col in train_x.columns]
    numeric_columns = [col for col in train_x.columns if col not in categorical_columns]
    log_stage_done(
        started_at,
        f"matrix shape train={train_x.shape}, valid={valid_x.shape}, test={test_x.shape}",
        enabled=show_progress,
    )

    started_at = log_stage(6, "apply rare-category vocabulary from train only", enabled=show_progress)
    category_vocab = apply_rare_categories(
        train_x,
        [valid_x, test_x],
        categorical_columns,
        min_count=args.rare_min_count,
    )
    log_stage_done(started_at, f"processed {len(categorical_columns):,} categorical columns", enabled=show_progress)

    started_at = log_stage(7, "train frequency model for is_claim", enabled=show_progress)
    frequency_model, valid_probability, frequency_backend = train_frequency_model(
        train_x,
        train_part["is_claim"],
        valid_x,
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
    )
    f1_threshold, best_f1 = best_f1_threshold(valid_part["is_claim"], valid_probability)
    log_stage_done(started_at, f"backend={frequency_backend}, best_f1={best_f1:.4f}", enabled=show_progress)

    started_at = log_stage(8, "train severity model on positive claim contracts", enabled=show_progress)
    severity_train_mask = train_part["claim_amount"] > 0
    if int(severity_train_mask.sum()) < args.severity_min_claims:
        raise ValueError(
            f"Not enough positive claim rows for severity model: {int(severity_train_mask.sum())}. "
            f"Lower --severity-min-claims if this is expected."
        )
    severity_model, valid_severity, severity_backend = train_severity_model(
        train_x.loc[severity_train_mask],
        train_part.loc[severity_train_mask, "claim_amount"],
        valid_x,
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
    )
    log_stage_done(
        started_at,
        f"backend={severity_backend}, positive_train={int(severity_train_mask.sum()):,}",
        enabled=show_progress,
    )

    started_at = log_stage(9, "calibrate pricing on validation set", enabled=show_progress)
    expected_claim_valid = valid_probability * valid_severity
    new_premium_valid, calibration = calibrate_pricing(valid_part, expected_claim_valid)
    log_stage_done(
        started_at,
        (
            f"loss_ratio={calibration.validation_loss_ratio:.4f}, "
            f"keep_or_decrease_share={calibration.keep_or_decrease_share:.4f}"
        ),
        enabled=show_progress,
    )

    started_at = log_stage(10, "evaluate validation metrics", enabled=show_progress)
    metrics = evaluate(
        valid_part,
        valid_probability,
        valid_severity,
        expected_claim_valid,
        new_premium_valid,
        f1_threshold,
    )
    metrics["frequency"]["best_f1_from_pr_curve"] = best_f1
    metrics["models"] = {
        "requested_backend": args.model_backend,
        "resolved_backend": model_backend,
        "frequency_backend": frequency_backend,
        "severity_backend": severity_backend,
        "n_train_contracts": int(len(train_part)),
        "n_valid_contracts": int(len(valid_part)),
        "n_test_contracts": int(len(test_contracts)),
        "n_features": int(len(train_x.columns)),
        "n_numeric_features": int(len(numeric_columns)),
        "n_categorical_features": int(len(categorical_columns)),
    }
    metrics["pricing_calibration"] = asdict(calibration)
    log_stage_done(
        started_at,
        (
            f"roc_auc={metrics['frequency']['roc_auc']:.4f}, "
            f"post_loss_ratio={metrics['business']['post_pricing_loss_ratio']:.4f}"
        ),
        enabled=show_progress,
    )

    started_at = log_stage(11, "predict test contracts and build submission", enabled=show_progress)
    test_probability = predict_frequency(frequency_model, test_x, categorical_columns, frequency_backend)
    test_severity = predict_severity(severity_model, test_x, categorical_columns, severity_backend)
    test_expected_claim = test_probability * test_severity
    test_new_premium = apply_pricing(test_contracts, test_expected_claim, calibration)
    test_pred_loss_ratio = test_expected_claim / np.maximum(test_contracts["premium"].astype(float).to_numpy(), 1.0)

    submission = pd.DataFrame(
        {
            "contract_number": test_contracts["contract_number"],
            "claim_probability": test_probability,
            "pred_loss_ratio": test_pred_loss_ratio,
            "new_premium": test_new_premium,
        }
    )
    submission.to_csv(submission_path, index=False)
    log_stage_done(started_at, f"saved {len(submission):,} rows to {submission_path}", enabled=show_progress)

    started_at = log_stage(12, "save model artifacts and metrics", enabled=show_progress)
    save_pickle(
        artifacts_dir / "models.pkl",
        {
            "frequency_model": frequency_model,
            "severity_model": severity_model,
            "frequency_backend": frequency_backend,
            "severity_backend": severity_backend,
            "feature_columns": list(train_x.columns),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "category_vocab": category_vocab,
            "pricing_calibration": calibration,
            "rare_min_count": args.rare_min_count,
        },
    )
    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    (artifacts_dir / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (artifacts_dir / f"metrics_{model_backend}.json").write_text(metrics_text, encoding="utf-8")
    log_stage_done(
        started_at,
        f"saved models.pkl, metrics.json, and metrics_{model_backend}.json to {artifacts_dir}",
        enabled=show_progress,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved submission: {submission_path}")
    print(f"Saved artifacts: {artifacts_dir}")

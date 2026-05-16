from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipeline.config import LEAKAGE_COLUMNS, OUTLIER_CAP_COLUMNS
from pipeline.contracts import build_contract_frame
from pipeline.evaluation import best_f1_threshold, evaluate
from pipeline.features import apply_rare_categories, build_feature_lists, split_train_valid
from pipeline.io import frame_len, read_input, save_pickle
from pipeline.logging import log_progress, log_stage, log_stage_done
from pipeline.models import (
    apply_severity_calibration,
    fit_severity_calibration,
    predict_frequency,
    predict_severity,
    resolve_model_backend,
    train_frequency_final_model,
    train_frequency_model,
    train_severity_model,
)
from pipeline.outliers import compute_engine_ratio_caps, compute_outlier_caps
from pipeline.pricing import apply_pricing, calibrate_pricing
from pipeline.risk_encoding import add_oof_risk_encoding_features


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _split_train_calibration_holdout(
    contracts: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if "operation_month" in contracts:
        train = contracts[contracts["operation_month"] <= 8].copy()
        calibration = contracts[contracts["operation_month"].between(9, 10)].copy()
        holdout = contracts[contracts["operation_month"] >= 11].copy()
        if (
            len(train) > 0
            and len(calibration) > 0
            and len(holdout) > 0
            and train["is_claim"].sum() >= 50
            and calibration["is_claim"].sum() > 0
            and holdout["is_claim"].sum() > 0
        ):
            return train, calibration, holdout, "time_month_1_8__9_10__11_12"

    train_idx, rest_idx = train_test_split(
        contracts.index,
        test_size=0.4,
        random_state=random_state,
        stratify=contracts["is_claim"],
    )
    rest = contracts.loc[rest_idx]
    calibration_idx, holdout_idx = train_test_split(
        rest.index,
        test_size=0.5,
        random_state=random_state,
        stratify=rest["is_claim"],
    )
    return (
        contracts.loc[train_idx].copy(),
        contracts.loc[calibration_idx].copy(),
        contracts.loc[holdout_idx].copy(),
        "stratified_60_20_20",
    )


def _make_feature_matrices(
    train: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    *,
    rare_min_count: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str], list[str], list[str], dict[str, list[str]]]:
    feature_columns, numeric_columns, categorical_columns = build_feature_lists(train)
    leaked = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns in feature matrix: {leaked}")

    train_x = train[feature_columns].copy()
    other_x: dict[str, pd.DataFrame] = {}
    for name, frame in others.items():
        matrix = frame[[col for col in feature_columns if col in frame.columns]].copy()
        for col in train_x.columns:
            if col not in matrix.columns:
                matrix[col] = np.nan
        other_x[name] = matrix[train_x.columns].copy()

    categorical_columns = [col for col in categorical_columns if col in train_x.columns]
    numeric_columns = [col for col in train_x.columns if col not in categorical_columns]
    category_vocab = apply_rare_categories(
        train_x,
        list(other_x.values()),
        categorical_columns,
        min_count=rare_min_count,
    )
    return train_x, other_x, list(train_x.columns), numeric_columns, categorical_columns, category_vocab


def _add_risk_encoding(
    train: pd.DataFrame,
    others: list[pd.DataFrame],
    args,
    *,
    enabled: bool,
) -> Any | None:
    if not enabled:
        return None
    if not others:
        scratch = train.iloc[0:0].copy()
        return add_oof_risk_encoding_features(
            train,
            scratch,
            scratch,
            random_state=args.random_state,
            n_splits=args.risk_encoding_splits,
            smoothing=args.risk_encoding_smoothing,
            include_id_columns=False,
        )
    valid = others[0]
    test = others[1] if len(others) > 1 else others[0].iloc[0:0].copy()
    return add_oof_risk_encoding_features(
        train,
        valid,
        test,
        random_state=args.random_state,
        n_splits=args.risk_encoding_splits,
        smoothing=args.risk_encoding_smoothing,
        include_id_columns=False,
    )


def _train_models_and_predict(
    train_part: pd.DataFrame,
    calibration_part: pd.DataFrame,
    predict_part: pd.DataFrame,
    args,
    model_backend: str,
    *,
    enable_risk_encoding: bool,
    show_progress: bool,
) -> dict[str, Any]:
    train_work = train_part.copy()
    calibration_work = calibration_part.copy()
    predict_work = predict_part.copy()
    risk_encoding_info = _add_risk_encoding(
        train_work,
        [calibration_work, predict_work],
        args,
        enabled=enable_risk_encoding,
    )
    train_x, other_x, feature_columns, numeric_columns, categorical_columns, category_vocab = _make_feature_matrices(
        train_work,
        {"calibration": calibration_work, "predict": predict_work},
        rare_min_count=args.rare_min_count,
    )
    calibration_x = other_x["calibration"]
    predict_x = other_x["predict"]

    frequency_model, calibration_probability, frequency_backend = train_frequency_model(
        train_x,
        train_work["is_claim"],
        calibration_x,
        calibration_work["is_claim"],
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
    )
    predict_probability = predict_frequency(frequency_model, predict_x, categorical_columns, frequency_backend)

    severity_train_mask = train_work["claim_amount"] > 0
    if int(severity_train_mask.sum()) < args.severity_min_claims:
        raise ValueError(
            f"Not enough positive claim rows for severity model: {int(severity_train_mask.sum())}. "
            f"Lower --severity-min-claims if this is expected."
        )
    severity_model, calibration_severity, severity_backend = train_severity_model(
        train_x.loc[severity_train_mask],
        train_work.loc[severity_train_mask, "claim_amount"],
        calibration_x,
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
        args.severity_target,
        not args.disable_severity_calibration,
        calibration_work["claim_amount"],
    )
    predict_severity_values = predict_severity(severity_model, predict_x, categorical_columns, severity_backend)

    return {
        "frequency_model": frequency_model,
        "severity_model": severity_model,
        "frequency_backend": frequency_backend,
        "severity_backend": severity_backend,
        "calibration_probability": calibration_probability,
        "calibration_severity": calibration_severity,
        "predict_probability": predict_probability,
        "predict_severity": predict_severity_values,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "category_vocab": category_vocab,
        "risk_encoding_info": risk_encoding_info,
    }


def _score_business_objective(metrics: dict) -> tuple[float, float, float, float]:
    business = metrics["business"]
    group_gap = business["group_loss_ratio_gap"]
    if group_gap is None:
        group_gap = 10.0
    total_gap = abs(business["post_pricing_loss_ratio"] - 0.70)
    keep_share = business["keep_or_decrease_share"]
    roc_auc = metrics["frequency"]["roc_auc"]
    return float(group_gap), float(total_gap), -float(keep_share), -float(roc_auc)


def _run_evaluation_config(
    train_contracts: pd.DataFrame,
    args,
    model_backend: str,
    *,
    enable_risk_encoding: bool,
    show_progress: bool,
) -> dict[str, Any]:
    train_part, calibration_part, holdout_part, split_strategy = _split_train_calibration_holdout(
        train_contracts,
        args.random_state,
    )
    trained = _train_models_and_predict(
        train_part,
        calibration_part,
        holdout_part,
        args,
        model_backend,
        enable_risk_encoding=enable_risk_encoding,
        show_progress=show_progress,
    )
    calibration_expected_claim = trained["calibration_probability"] * trained["calibration_severity"]
    _, pricing_calibration = calibrate_pricing(calibration_part, calibration_expected_claim)

    holdout_expected_claim = trained["predict_probability"] * trained["predict_severity"]
    holdout_new_premium = apply_pricing(holdout_part, holdout_expected_claim, pricing_calibration)
    f1_threshold, best_f1 = best_f1_threshold(calibration_part["is_claim"], trained["calibration_probability"])
    metrics = evaluate(
        holdout_part,
        trained["predict_probability"],
        trained["predict_severity"],
        holdout_expected_claim,
        holdout_new_premium,
        f1_threshold,
    )
    metrics["frequency"]["best_f1_from_calibration_pr_curve"] = best_f1
    metrics["models"] = {
        "requested_backend": args.model_backend,
        "resolved_backend": model_backend,
        "frequency_backend": trained["frequency_backend"],
        "severity_backend": trained["severity_backend"],
        "n_model_train_contracts": int(len(train_part)),
        "n_calibration_contracts": int(len(calibration_part)),
        "n_holdout_contracts": int(len(holdout_part)),
        "n_features": int(len(trained["feature_columns"])),
        "n_numeric_features": int(len(trained["numeric_columns"])),
        "n_categorical_features": int(len(trained["categorical_columns"])),
        "severity_target": args.severity_target,
        "severity_calibrated": not args.disable_severity_calibration,
        "risk_encoding_enabled": enable_risk_encoding,
        "id_risk_encoding_enabled": False,
        "n_risk_encoding_features": 0
        if trained["risk_encoding_info"] is None
        else len(trained["risk_encoding_info"].encoded_columns),
        "split_strategy": split_strategy,
    }
    metrics["pricing_calibration"] = asdict(pricing_calibration)
    metrics["evaluation_note"] = (
        "Models are fit on model_train, severity/pricing are calibrated on calibration, "
        "and all reported quality/business metrics are measured on untouched holdout."
    )
    return {
        "metrics": metrics,
        "pricing_calibration": pricing_calibration,
        "split_strategy": split_strategy,
        "enable_risk_encoding": enable_risk_encoding,
    }


def _run_evaluation(
    train_contracts: pd.DataFrame,
    args,
    model_backend: str,
    artifacts_dir: Path,
    *,
    show_progress: bool,
) -> dict[str, Any]:
    configs = [False, True] if args.compare_risk_encoding else [args.enable_risk_encoding]
    results: list[dict[str, Any]] = []
    for enabled in configs:
        log_progress(f"Evaluation config: risk_encoding={enabled}", enabled=show_progress)
        result = _run_evaluation_config(
            train_contracts,
            args,
            model_backend,
            enable_risk_encoding=enabled,
            show_progress=show_progress,
        )
        results.append(result)
        suffix = "risk_encoding" if enabled else "baseline"
        _write_json(artifacts_dir / f"metrics_holdout_{suffix}.json", result["metrics"])

    selected = min(results, key=lambda item: _score_business_objective(item["metrics"]))
    final_metrics = {
        "selected_config": {
            "risk_encoding_enabled": selected["enable_risk_encoding"],
            "selection_objective": "group_gap, total_gap, keep_or_decrease_share, roc_auc",
        },
        "holdout": selected["metrics"],
        "all_evaluation_configs": [
            {
                "risk_encoding_enabled": item["enable_risk_encoding"],
                "objective": _score_business_objective(item["metrics"]),
                "roc_auc": item["metrics"]["frequency"]["roc_auc"],
                "post_pricing_loss_ratio": item["metrics"]["business"]["post_pricing_loss_ratio"],
                "group_loss_ratio_gap": item["metrics"]["business"]["group_loss_ratio_gap"],
                "keep_or_decrease_share": item["metrics"]["business"]["keep_or_decrease_share"],
            }
            for item in results
        ],
    }
    _write_json(artifacts_dir / "metrics.json", selected["metrics"])
    _write_json(artifacts_dir / "final_metrics.json", final_metrics)
    return selected


def _run_production(
    train_contracts: pd.DataFrame,
    test_contracts: pd.DataFrame,
    args,
    model_backend: str,
    artifacts_dir: Path,
    submission_path: Path,
    *,
    enable_risk_encoding: bool,
    outlier_caps: dict[str, tuple[float, float]],
    show_progress: bool,
) -> dict[str, Any]:
    train_work = train_contracts.copy()
    test_work = test_contracts.copy()
    risk_encoding_info = _add_risk_encoding(
        train_work,
        [test_work],
        args,
        enabled=enable_risk_encoding,
    )
    full_x, other_x, feature_columns, numeric_columns, categorical_columns, category_vocab = _make_feature_matrices(
        train_work,
        {"test": test_work},
        rare_min_count=args.rare_min_count,
    )
    test_x = other_x["test"]

    y = train_work["is_claim"].astype(int)
    n_splits = min(max(2, args.production_splits), int(y.value_counts().min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    oof_probability = np.zeros(len(train_work), dtype=float)
    oof_raw_severity = np.zeros(len(train_work), dtype=float)

    for fold_idx, (fold_train_idx, fold_valid_idx) in enumerate(splitter.split(full_x, y), start=1):
        log_progress(f"Production OOF fold {fold_idx}/{n_splits}", enabled=show_progress)
        fold_train_x = full_x.iloc[fold_train_idx].copy()
        fold_valid_x = full_x.iloc[fold_valid_idx].copy()
        fold_train = train_work.iloc[fold_train_idx]
        fold_valid = train_work.iloc[fold_valid_idx]

        frequency_model, frequency_backend = train_frequency_final_model(
            fold_train_x,
            fold_train["is_claim"],
            categorical_columns,
            numeric_columns,
            args.random_state + fold_idx,
            model_backend,
            args.progress_period,
            show_progress,
        )
        fold_probability = predict_frequency(frequency_model, fold_valid_x, categorical_columns, frequency_backend)
        oof_probability[fold_valid_idx] = fold_probability

        severity_mask = fold_train["claim_amount"] > 0
        if int(severity_mask.sum()) < args.severity_min_claims:
            raise ValueError(
                f"Not enough positive claim rows for OOF severity fold {fold_idx}: "
                f"{int(severity_mask.sum())}."
            )
        severity_model, fold_severity, severity_backend = train_severity_model(
            fold_train_x.loc[severity_mask],
            fold_train.loc[severity_mask, "claim_amount"],
            fold_valid_x,
            categorical_columns,
            numeric_columns,
            args.random_state + fold_idx,
            model_backend,
            args.progress_period,
            show_progress,
            args.severity_target,
            False,
            None,
        )
        oof_raw_severity[fold_valid_idx] = fold_severity

    severity_calibration = (
        fit_severity_calibration(oof_raw_severity, train_work["claim_amount"])
        if not args.disable_severity_calibration
        else None
    )
    oof_severity = apply_severity_calibration(oof_raw_severity, severity_calibration)
    oof_expected_claim = oof_probability * oof_severity
    oof_new_premium, pricing_calibration = calibrate_pricing(train_work, oof_expected_claim)
    f1_threshold, best_f1 = best_f1_threshold(train_work["is_claim"], oof_probability)
    oof_metrics = evaluate(
        train_work,
        oof_probability,
        oof_severity,
        oof_expected_claim,
        oof_new_premium,
        f1_threshold,
    )
    oof_metrics["frequency"]["best_f1_from_oof_pr_curve"] = best_f1
    oof_metrics["models"] = {
        "requested_backend": args.model_backend,
        "resolved_backend": model_backend,
        "frequency_backend": frequency_backend,
        "severity_backend": severity_backend,
        "n_train_contracts": int(len(train_work)),
        "n_test_contracts": int(len(test_work)),
        "n_oof_splits": int(n_splits),
        "n_features": int(len(feature_columns)),
        "n_numeric_features": int(len(numeric_columns)),
        "n_categorical_features": int(len(categorical_columns)),
        "severity_target": args.severity_target,
        "severity_calibrated": not args.disable_severity_calibration,
        "risk_encoding_enabled": enable_risk_encoding,
        "id_risk_encoding_enabled": False,
        "n_risk_encoding_features": 0 if risk_encoding_info is None else len(risk_encoding_info.encoded_columns),
    }
    oof_metrics["pricing_calibration"] = asdict(pricing_calibration)
    oof_metrics["production_note"] = (
        "OOF predictions are used only to calibrate pricing on train without in-sample predictions; "
        "final models below are fit on all train contracts for test inference."
    )

    frequency_model, final_frequency_backend = train_frequency_final_model(
        full_x,
        train_work["is_claim"],
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
    )
    severity_mask = train_work["claim_amount"] > 0
    severity_model, _, final_severity_backend = train_severity_model(
        full_x.loc[severity_mask],
        train_work.loc[severity_mask, "claim_amount"],
        full_x.iloc[:1].copy(),
        categorical_columns,
        numeric_columns,
        args.random_state,
        model_backend,
        args.progress_period,
        show_progress,
        args.severity_target,
        False,
        None,
    )
    severity_model.calibration = severity_calibration

    test_probability = predict_frequency(frequency_model, test_x, categorical_columns, final_frequency_backend)
    test_severity = predict_severity(severity_model, test_x, categorical_columns, final_severity_backend)
    test_expected_claim = test_probability * test_severity
    test_new_premium = apply_pricing(test_work, test_expected_claim, pricing_calibration)
    test_pred_loss_ratio = test_expected_claim / np.maximum(test_work["premium"].astype(float).to_numpy(), 1.0)

    if not np.isfinite(test_probability).all() or not np.isfinite(test_pred_loss_ratio).all():
        raise ValueError("Submission contains non-finite probability or loss-ratio predictions.")
    max_allowed = test_work["premium"].astype(float).to_numpy() * 3.0
    if (test_new_premium < -1e-9).any() or (test_new_premium > max_allowed + 1e-9).any():
        raise ValueError("Submission violates new_premium bounds [0, 3 * premium].")

    submission = pd.DataFrame(
        {
            "contract_number": test_work["contract_number"],
            "claim_probability": test_probability,
            "pred_loss_ratio": test_pred_loss_ratio,
            "new_premium": test_new_premium,
        }
    )
    submission.to_csv(submission_path, index=False)

    save_pickle(
        artifacts_dir / "models.pkl",
        {
            "frequency_model": frequency_model,
            "severity_model": severity_model,
            "frequency_backend": final_frequency_backend,
            "severity_backend": final_severity_backend,
            "severity_target": args.severity_target,
            "severity_calibrated": not args.disable_severity_calibration,
            "severity_calibration": severity_calibration,
            "risk_encoding": risk_encoding_info,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "category_vocab": category_vocab,
            "pricing_calibration": pricing_calibration,
            "rare_min_count": args.rare_min_count,
            "outlier_caps": outlier_caps,
        },
    )
    _write_json(artifacts_dir / "production_metrics.json", oof_metrics)
    return {
        "metrics": oof_metrics,
        "submission_rows": int(len(submission)),
        "pricing_calibration": pricing_calibration,
    }


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

    outlier_caps = compute_outlier_caps(raw_train, OUTLIER_CAP_COLUMNS)
    outlier_caps.update(compute_engine_ratio_caps(raw_train))

    started_at = log_stage(2, "aggregate train/test rows to contract level", enabled=show_progress)
    train_contracts = build_contract_frame(raw_train, is_train=True, outlier_caps=outlier_caps)
    test_contracts = build_contract_frame(raw_test, is_train=False, outlier_caps=outlier_caps)
    log_stage_done(
        started_at,
        f"built {len(train_contracts):,} train contracts and {len(test_contracts):,} test contracts",
        enabled=show_progress,
    )

    if args.final_mode == "legacy":
        train_part, valid_part = split_train_valid(train_contracts, args.valid_size, args.random_state)
        legacy = _train_models_and_predict(
            train_part,
            valid_part,
            test_contracts,
            args,
            model_backend,
            enable_risk_encoding=args.enable_risk_encoding,
            show_progress=show_progress,
        )
        expected_claim_valid = legacy["calibration_probability"] * legacy["calibration_severity"]
        new_premium_valid, calibration = calibrate_pricing(valid_part, expected_claim_valid)
        f1_threshold, best_f1 = best_f1_threshold(valid_part["is_claim"], legacy["calibration_probability"])
        metrics = evaluate(
            valid_part,
            legacy["calibration_probability"],
            legacy["calibration_severity"],
            expected_claim_valid,
            new_premium_valid,
            f1_threshold,
        )
        metrics["frequency"]["best_f1_from_pr_curve"] = best_f1
        metrics["pricing_calibration"] = asdict(calibration)
        _write_json(artifacts_dir / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default))
        return

    selected_eval: dict[str, Any] | None = None
    if args.final_mode in {"evaluation", "both"}:
        started_at = log_stage(3, "run leakage-safe holdout evaluation", enabled=show_progress)
        selected_eval = _run_evaluation(
            train_contracts,
            args,
            model_backend,
            artifacts_dir,
            show_progress=show_progress,
        )
        log_stage_done(
            started_at,
            (
                f"selected risk_encoding={selected_eval['enable_risk_encoding']}, "
                f"holdout_loss_ratio={selected_eval['metrics']['business']['post_pricing_loss_ratio']:.4f}"
            ),
            enabled=show_progress,
        )

    production_result: dict[str, Any] | None = None
    if args.final_mode in {"production", "both"}:
        production_risk_encoding = (
            bool(selected_eval["enable_risk_encoding"])
            if selected_eval is not None
            else bool(args.enable_risk_encoding)
        )
        started_at = log_stage(4, "run OOF-calibrated production submission", enabled=show_progress)
        production_result = _run_production(
            train_contracts,
            test_contracts,
            args,
            model_backend,
            artifacts_dir,
            submission_path,
            enable_risk_encoding=production_risk_encoding,
            outlier_caps=outlier_caps,
            show_progress=show_progress,
        )
        log_stage_done(
            started_at,
            f"saved {production_result['submission_rows']:,} submission rows to {submission_path}",
            enabled=show_progress,
        )

    combined = {
        "selected_holdout_metrics": None if selected_eval is None else selected_eval["metrics"],
        "production_oof_metrics": None if production_result is None else production_result["metrics"],
        "submission_path": str(submission_path) if production_result is not None else None,
        "artifacts_dir": str(artifacts_dir),
    }
    _write_json(artifacts_dir / "final_metrics.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2, default=_json_default))
    if production_result is not None:
        print(f"Saved submission: {submission_path}")
    print(f"Saved artifacts: {artifacts_dir}")

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.car import add_car_features
from features.model_mark import add_model_mark_features
from features.premium import add_premium_features
from features.region import add_region_features
from features.score import build_score_features, score_group

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


TARGET_COLUMNS = {"claim_amount", "claim_cnt", "is_claim"}
RAW_ID_COLUMNS = {"unique_id", "contract_number", "insurer_iin", "driver_iin", "car_number"}
FINANCIAL_METRIC_ONLY_COLUMNS = {
    "premium_wo_term",
    "premium_wo_term_log1p",
    "premium_wo_term_is_zero",
    "premium_wo_term_is_missing",
    "premium_wo_term_ratio",
    "premium_return_ratio",
}
DATE_COLUMNS = {"operation_date"}
PREPROCESSED_SOURCE_COLUMNS = {"mark", "model", "car_age", "car_year"}
LEAKAGE_COLUMNS = TARGET_COLUMNS | RAW_ID_COLUMNS | FINANCIAL_METRIC_ONLY_COLUMNS
HIGH_CARDINALITY_COLUMNS = {
    "mark_clean_mode",
    "model_clean_mode",
    "mark_model_pair",
    "ownerkato",
    "ownerkato_short",
}
TARGET_LOSS_RATIO = 0.70
N_PIPELINE_STAGES = 12


@dataclass
class PricingCalibration:
    scale: float
    threshold: float
    validation_loss_ratio: float
    group_keep_or_decrease_loss_ratio: float | None
    group_increase_loss_ratio: float | None
    keep_or_decrease_share: float


def is_categorical_feature(series: pd.Series, column_name: str) -> bool:
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or column_name in HIGH_CARDINALITY_COLUMNS
    )


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
        help="Print LightGBM training progress every N boosting iterations.",
    )
    parser.add_argument(
        "--aggregation-backend",
        choices=["polars", "pandas"],
        default="polars",
        help="Backend used for contract-level feature aggregation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress messages.")
    parser.add_argument(
        "--force-sklearn",
        action="store_true",
        help="Use sklearn fallback models even when LightGBM is installed.",
    )
    return parser.parse_args()


def log_progress(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def log_stage(stage: int, message: str, *, enabled: bool = True) -> float:
    log_progress(f"Stage {stage}/{N_PIPELINE_STAGES}: {message}", enabled=enabled)
    return time.perf_counter()


def log_stage_done(started_at: float, message: str, *, enabled: bool = True) -> None:
    elapsed = time.perf_counter() - started_at
    log_progress(f"Done in {elapsed:.1f}s: {message}", enabled=enabled)


def make_lgbm_progress_callback(label: str, total_iterations: int, period: int, enabled: bool):
    period = max(1, period)

    def _callback(env) -> None:
        iteration = env.iteration + 1
        if iteration == 1 or iteration % period == 0 or iteration >= total_iterations:
            pct = 100.0 * iteration / max(total_iterations, 1)
            log_progress(f"{label}: iteration {iteration}/{total_iterations} ({pct:.1f}%)", enabled=enabled)

    _callback.order = 10
    return _callback


def read_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig", nrows=nrows)


def read_input(path: str | Path, nrows: int | None, backend: str) -> pd.DataFrame | pl.DataFrame:
    if backend == "polars":
        return pl.read_csv(path, n_rows=nrows, infer_schema_length=None, encoding="utf8-lossy")
    return read_csv(path, nrows=nrows)


def frame_len(frame: pd.DataFrame | pl.DataFrame) -> int:
    return len(frame)


def frame_columns(frame: pd.DataFrame | pl.DataFrame) -> list[str]:
    return list(frame.columns)


def polars_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def mode_or_missing(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return "__MISSING__"
    mode = non_null.mode(dropna=True)
    if mode.empty:
        return str(non_null.iloc[0])
    return str(mode.iloc[0])


def nunique_non_null(series: pd.Series) -> int:
    return int(series.nunique(dropna=True))


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan))


def pl_first_non_null(column: str) -> pl.Expr:
    return pl.col(column).drop_nulls().first().alias(column)


def pl_mode_or_missing(column: str, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .drop_nulls()
        .mode()
        .first()
        .cast(pl.String)
        .fill_null("__MISSING__")
        .alias(alias)
    )


def pl_nunique_non_null(column: str, alias: str) -> pl.Expr:
    return pl.col(column).drop_nulls().n_unique().cast(pl.Int32).alias(alias)


def assert_contract_targets_are_constant(df: pd.DataFrame) -> None:
    present = [col for col in ["premium", "premium_wo_term", "claim_amount", "claim_cnt", "is_claim"] if col in df]
    inconsistent = {}
    grouped = df.groupby("contract_number", sort=False)
    for col in present:
        n_bad = int((grouped[col].nunique(dropna=False) > 1).sum())
        if n_bad:
            inconsistent[col] = n_bad
    if inconsistent:
        raise ValueError(f"Contract-level fields vary within contract_number: {inconsistent}")


def assert_contract_targets_are_constant_polars(df: pl.DataFrame) -> None:
    present = [col for col in ["premium", "premium_wo_term", "claim_amount", "claim_cnt", "is_claim"] if col in df.columns]
    if not present:
        return
    agg = df.group_by("contract_number").agg([pl.col(col).n_unique().alias(col) for col in present])
    inconsistent = {
        col: int(agg.filter(pl.col(col) > 1).height)
        for col in present
        if int(agg.filter(pl.col(col) > 1).height) > 0
    }
    if inconsistent:
        raise ValueError(f"Contract-level fields vary within contract_number: {inconsistent}")


def add_date_features(frame: pd.DataFrame, raw: pd.DataFrame) -> None:
    if "operation_date" not in raw:
        return
    operation_date = raw.groupby("contract_number", sort=False)["operation_date"].agg(first_non_null)
    dates = pd.to_datetime(operation_date, errors="coerce")
    frame["operation_month"] = dates.dt.month
    frame["operation_quarter"] = dates.dt.quarter
    frame["operation_dayofweek"] = dates.dt.dayofweek


def add_special_features(frame: pd.DataFrame, raw: pd.DataFrame) -> None:
    grouped = raw.groupby("contract_number", sort=False)
    frame["n_rows"] = grouped.size().astype("int32")

    if "driver_iin" in raw:
        frame["n_drivers"] = grouped["driver_iin"].nunique(dropna=True).astype("int32")
    if "insurer_iin" in raw:
        frame["n_insurers"] = grouped["insurer_iin"].nunique(dropna=True).astype("int32")
    if {"insurer_iin", "driver_iin"}.issubset(raw.columns):
        same_person = raw["insurer_iin"].fillna("") == raw["driver_iin"].fillna("__driver_missing__")
        frame["insurer_is_driver_any"] = same_person.groupby(raw["contract_number"], sort=False).max().astype("int8")
        frame["insurer_is_driver_all"] = same_person.groupby(raw["contract_number"], sort=False).min().astype("int8")
    if "car_number" in raw:
        frame["n_cars"] = grouped["car_number"].nunique(dropna=True).astype("int32")
    if "region_id" in raw:
        frame["n_regions"] = grouped["region_id"].nunique(dropna=True).astype("int32")

    for source, target in [
        ("n_drivers", "is_multi_driver"),
        ("n_cars", "is_multi_car"),
        ("n_regions", "is_multi_region"),
    ]:
        if source in frame:
            frame[target] = (frame[source] > 1).astype("int8")


def build_polars_score_features(raw_pl: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if "contract_number" not in raw_pl.columns:
        return pd.DataFrame(index=index)

    score_cols = [col for col in raw_pl.columns if col.startswith("SCORE_")]
    if not score_cols:
        return pd.DataFrame(index=index)

    score_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in score_cols]
    available_exprs = [pl.col(col).cast(pl.Float64, strict=False).is_not_null().cast(pl.Int32) for col in score_cols]
    missing_exprs = [pl.col(col).cast(pl.Float64, strict=False).is_null().cast(pl.Int32) for col in score_cols]
    score_available = pl.sum_horizontal(available_exprs)
    score_sum = pl.sum_horizontal([expr.fill_null(0.0) for expr in score_exprs])
    score_sum_sq = pl.sum_horizontal([(expr * expr).fill_null(0.0) for expr in score_exprs])
    score_row_std = (
        pl.when(score_available > 1)
        .then(((score_sum_sq - (score_sum * score_sum / score_available)) / (score_available - 1)).clip(0.0).sqrt())
        .otherwise(None)
    )

    grouped_columns: dict[str, list[str]] = {}
    for col in score_cols:
        group = score_group(col)
        if group is not None:
            grouped_columns.setdefault(group, []).append(col)

    group_row_columns: list[pl.Expr] = []
    group_agg_exprs: list[pl.Expr] = []
    for group, cols in sorted(grouped_columns.items(), key=lambda item: int(item[0])):
        group_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in cols]
        prefix = f"score_g{group}"
        group_row_columns.extend(
            [
                pl.mean_horizontal(group_exprs).alias(f"_{prefix}_mean"),
                pl.mean_horizontal([expr.is_null().cast(pl.Int32) for expr in group_exprs]).alias(
                    f"_{prefix}_missing_rate"
                ),
            ]
        )
        group_agg_exprs.extend(
            [
                pl.col(f"_{prefix}_mean").mean().alias(f"{prefix}_mean_mean"),
                pl.col(f"_{prefix}_mean").min().alias(f"{prefix}_mean_min"),
                pl.col(f"_{prefix}_mean").max().alias(f"{prefix}_mean_max"),
                pl.col(f"_{prefix}_mean").std().alias(f"{prefix}_mean_std"),
                pl.col(f"_{prefix}_missing_rate").mean().alias(f"{prefix}_missing_rate_mean"),
            ]
        )

    with_rows = raw_pl.select(
        "contract_number",
        score_available.alias("_score_available_count"),
        pl.mean_horizontal(missing_exprs).alias("_score_missing_rate"),
        pl.mean_horizontal(score_exprs).alias("_score_row_mean"),
        pl.min_horizontal(score_exprs).alias("_score_row_min"),
        pl.max_horizontal(score_exprs).alias("_score_row_max"),
        score_row_std.alias("_score_row_std"),
        *group_row_columns,
    )

    agg_exprs = [
        pl.col("_score_available_count").mean().alias("score_available_count_mean"),
        pl.col("_score_available_count").min().cast(pl.Int32).alias("score_available_count_min"),
        pl.col("_score_missing_rate").mean().alias("score_missing_rate_mean"),
        pl.col("_score_missing_rate").max().alias("score_missing_rate_max"),
        pl.col("_score_row_mean").mean().alias("score_row_mean_mean"),
        pl.col("_score_row_mean").min().alias("score_row_mean_min"),
        pl.col("_score_row_mean").max().alias("score_row_mean_max"),
        pl.col("_score_row_mean").std().alias("score_row_mean_std"),
        pl.col("_score_row_min").mean().alias("score_row_min_mean"),
        pl.col("_score_row_min").min().alias("score_row_min_min"),
        pl.col("_score_row_max").mean().alias("score_row_max_mean"),
        pl.col("_score_row_max").max().alias("score_row_max_max"),
        pl.col("_score_row_std").mean().alias("score_row_std_mean"),
        pl.col("_score_row_std").max().alias("score_row_std_max"),
        *group_agg_exprs,
    ]

    features = polars_to_pandas(with_rows.group_by("contract_number", maintain_order=True).agg(agg_exprs))
    features = features.set_index("contract_number").reindex(index).replace([np.inf, -np.inf], np.nan)
    return features


def build_contract_frame_polars(raw: pd.DataFrame | pl.DataFrame, is_train: bool) -> pd.DataFrame:
    if "contract_number" not in frame_columns(raw):
        raise ValueError("Input data must contain contract_number.")
    raw_pl = raw if isinstance(raw, pl.DataFrame) else pl.from_dict(raw.to_dict("list"))
    if is_train:
        assert_contract_targets_are_constant_polars(raw_pl)

    metric_columns = ["premium", "premium_wo_term", "claim_amount", "claim_cnt", "is_claim"]
    base_exprs = [pl_first_non_null(col) for col in metric_columns if col in raw_pl.columns]
    special_exprs: list[pl.Expr] = [pl.len().cast(pl.Int32).alias("n_rows")]
    if "driver_iin" in raw_pl.columns:
        special_exprs.append(pl_nunique_non_null("driver_iin", "n_drivers"))
    if "insurer_iin" in raw_pl.columns:
        special_exprs.append(pl_nunique_non_null("insurer_iin", "n_insurers"))
    if "car_number" in raw_pl.columns:
        special_exprs.append(pl_nunique_non_null("car_number", "n_cars"))
    if "region_id" in raw_pl.columns:
        special_exprs.append(pl_nunique_non_null("region_id", "n_regions"))
    if {"insurer_iin", "driver_iin"}.issubset(raw_pl.columns):
        same_person = pl.col("insurer_iin").fill_null("") == pl.col("driver_iin").fill_null("__driver_missing__")
        special_exprs.extend(
            [
                same_person.max().cast(pl.Int8).alias("insurer_is_driver_any"),
                same_person.min().cast(pl.Int8).alias("insurer_is_driver_all"),
            ]
        )

    grouped = raw_pl.group_by("contract_number", maintain_order=True).agg([*base_exprs, *special_exprs])
    frame = polars_to_pandas(grouped).set_index("contract_number")

    for source, target in [
        ("n_drivers", "is_multi_driver"),
        ("n_cars", "is_multi_car"),
        ("n_regions", "is_multi_region"),
    ]:
        if source in frame:
            frame[target] = (frame[source] > 1).astype("int8")

    pandas_needed = [
        col
        for col in [
            "contract_number",
            "region_name",
            "mark",
            "model",
            "car_year",
            "car_age",
            "operation_date",
        ]
        if col in raw_pl.columns
    ]
    raw_for_local_features = polars_to_pandas(raw_pl.select(pandas_needed))
    add_region_features(raw_for_local_features, frame)
    add_model_mark_features(raw_for_local_features, frame)
    add_car_features(raw_for_local_features, frame)
    score_features = build_polars_score_features(raw_pl, frame.index)
    if not score_features.empty:
        frame = pd.concat([frame, score_features], axis=1).copy()
    add_premium_features(frame)
    add_date_features(frame, raw_for_local_features)

    excluded = LEAKAGE_COLUMNS | DATE_COLUMNS | PREPROCESSED_SOURCE_COLUMNS
    candidate_columns = [col for col in raw_pl.columns if col not in excluded and not col.startswith("SCORE_")]
    numeric_columns = [
        col
        for col in candidate_columns
        if raw_pl.schema[col].is_numeric() and col not in {"ownerkato", "ownerkato_short"}
    ]
    categorical_columns = [col for col in candidate_columns if col not in numeric_columns]
    categorical_columns.extend([col for col in ["ownerkato", "ownerkato_short"] if col in raw])
    categorical_columns = list(dict.fromkeys(categorical_columns))

    agg_exprs: list[pl.Expr] = []
    for col in numeric_columns:
        value = pl.col(col).cast(pl.Float64, strict=False)
        agg_exprs.extend(
            [
                value.min().alias(f"{col}_min"),
                value.max().alias(f"{col}_max"),
                value.mean().alias(f"{col}_mean"),
                value.std().alias(f"{col}_std"),
                pl_nunique_non_null(col, f"{col}_nunique"),
            ]
        )
    for col in categorical_columns:
        agg_exprs.extend(
            [
                pl_mode_or_missing(col, f"{col}_mode"),
                pl_nunique_non_null(col, f"{col}_nunique"),
            ]
        )

    if agg_exprs:
        generic_features = polars_to_pandas(raw_pl.group_by("contract_number", maintain_order=True).agg(agg_exprs))
        generic_features = generic_features.set_index("contract_number").reindex(frame.index)
        frame = pd.concat([frame, generic_features], axis=1).copy()

    if "claim_amount" in frame:
        frame["claim_amount"] = frame["claim_amount"].fillna(0.0)
    if "claim_cnt" in frame:
        frame["claim_cnt"] = frame["claim_cnt"].fillna(0.0)
    if "is_claim" in frame:
        frame["is_claim"] = frame["is_claim"].fillna(0).astype("int8")
    if {"claim_amount", "premium_wo_term"}.issubset(frame.columns):
        frame["loss_ratio"] = safe_divide(frame["claim_amount"], frame["premium_wo_term"]).fillna(0.0)

    return frame.copy().reset_index()


def build_contract_frame(raw: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    if "contract_number" not in raw:
        raise ValueError("Input data must contain contract_number.")
    if is_train:
        assert_contract_targets_are_constant(raw)

    grouped = raw.groupby("contract_number", sort=False)
    frame = pd.DataFrame(index=grouped.size().index)
    frame.index.name = "contract_number"

    base_columns: list[pd.Series] = []
    metric_columns = ["premium", "premium_wo_term", "claim_amount", "claim_cnt", "is_claim"]
    for col in metric_columns:
        if col in raw:
            base_columns.append(grouped[col].agg(first_non_null).rename(col))
    if base_columns:
        frame = pd.concat([frame, *base_columns], axis=1)

    add_special_features(frame, raw)
    add_region_features(raw, frame)
    add_model_mark_features(raw, frame)
    add_car_features(raw, frame)
    score_features = build_score_features(raw, frame.index)
    if not score_features.empty:
        frame = pd.concat([frame, score_features], axis=1).copy()
    add_premium_features(frame)
    add_date_features(frame, raw)

    excluded = LEAKAGE_COLUMNS | DATE_COLUMNS | PREPROCESSED_SOURCE_COLUMNS
    candidate_columns = [col for col in raw.columns if col not in excluded and not col.startswith("SCORE_")]
    numeric_columns = [
        col
        for col in candidate_columns
        if pd.api.types.is_numeric_dtype(raw[col]) and col not in {"ownerkato", "ownerkato_short"}
    ]
    categorical_columns = [col for col in candidate_columns if col not in numeric_columns]
    categorical_columns.extend([col for col in ["ownerkato", "ownerkato_short"] if col in raw])
    categorical_columns = list(dict.fromkeys(categorical_columns))

    feature_blocks: list[pd.DataFrame] = []
    for col in numeric_columns:
        values = grouped[col].agg(["min", "max", "mean", "std"])
        values.columns = [f"{col}_{suffix}" for suffix in values.columns]
        nunique = grouped[col].nunique(dropna=True).astype("int32").rename(f"{col}_nunique")
        feature_blocks.append(pd.concat([values, nunique], axis=1))

    categorical_features: list[pd.Series] = []
    for col in categorical_columns:
        categorical_features.append(grouped[col].agg(mode_or_missing).rename(f"{col}_mode"))
        categorical_features.append(grouped[col].agg(nunique_non_null).astype("int32").rename(f"{col}_nunique"))

    if categorical_features:
        feature_blocks.append(pd.concat(categorical_features, axis=1))
    if feature_blocks:
        frame = pd.concat([frame, *feature_blocks], axis=1).copy()

    if "claim_amount" in frame:
        frame["claim_amount"] = frame["claim_amount"].fillna(0.0)
    if "claim_cnt" in frame:
        frame["claim_cnt"] = frame["claim_cnt"].fillna(0.0)
    if "is_claim" in frame:
        frame["is_claim"] = frame["is_claim"].fillna(0).astype("int8")
    if {"claim_amount", "premium_wo_term"}.issubset(frame.columns):
        frame["loss_ratio"] = safe_divide(frame["claim_amount"], frame["premium_wo_term"]).fillna(0.0)

    return frame.copy().reset_index()


def build_feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_columns = [
        col
        for col in frame.columns
        if col not in LEAKAGE_COLUMNS
        and col not in {"loss_ratio"}
        and not col.startswith("claim_amount")
        and not col.startswith("claim_cnt")
        and not col.startswith("is_claim")
    ]
    categorical = [col for col in feature_columns if is_categorical_feature(frame[col], col)]
    numeric = [col for col in feature_columns if col not in categorical]
    leaked = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns in features: {leaked}")
    return feature_columns, numeric, categorical


def apply_rare_categories(
    train: pd.DataFrame,
    other_frames: Iterable[pd.DataFrame],
    categorical_columns: list[str],
    min_count: int,
) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = {}
    for col in categorical_columns:
        train[col] = train[col].fillna("__MISSING__").astype(str)
        counts = train[col].value_counts(dropna=False)
        keep = set(counts[counts >= min_count].index.astype(str))
        vocab[col] = sorted(keep)
        train[col] = train[col].where(train[col].isin(keep), "__RARE__")
        for frame in other_frames:
            frame[col] = frame[col].fillna("__MISSING__").astype(str)
            frame[col] = frame[col].where(frame[col].isin(keep), "__RARE__")
    return vocab


def align_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_in_test = [col for col in train.columns if col not in test.columns]
    missing_in_train = [col for col in test.columns if col not in train.columns]
    for col in missing_in_test:
        test[col] = np.nan
    for col in missing_in_train:
        train[col] = np.nan
    return train, test[train.columns]


def split_train_valid(
    contracts: pd.DataFrame, valid_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "operation_month" in contracts:
        cutoff = contracts["operation_month"].quantile(1.0 - valid_size)
        train = contracts[contracts["operation_month"] <= cutoff].copy()
        valid = contracts[contracts["operation_month"] > cutoff].copy()
        if len(valid) > 0 and valid["is_claim"].sum() > 0 and len(train) > len(valid):
            return train, valid

    train_idx, valid_idx = train_test_split(
        contracts.index,
        test_size=valid_size,
        random_state=random_state,
        stratify=contracts["is_claim"],
    )
    return contracts.loc[train_idx].copy(), contracts.loc[valid_idx].copy()


def make_lgbm_classifier(random_state: int):
    return LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_lgbm_regressor(random_state: int):
    return LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_sklearn_model(model_type: str, numeric: list[str], categorical: list[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    if model_type == "classifier":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def prepare_lgbm_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for col in categorical:
        prepared[col] = prepared[col].fillna("__MISSING__").astype("category")
    missed_categorical = [
        col for col in prepared.columns if col not in categorical and is_categorical_feature(prepared[col], col)
    ]
    if missed_categorical:
        raise ValueError(
            "LightGBM input contains string/category columns not declared as categorical: "
            f"{missed_categorical}"
        )
    return prepared


def train_frequency_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    random_state: int,
    force_sklearn: bool,
    progress_period: int,
    show_progress: bool,
):
    if HAS_LIGHTGBM and not force_sklearn:
        model = make_lgbm_classifier(random_state)
        model.fit(
            prepare_lgbm_frame(train_x, categorical),
            train_y,
            categorical_feature=categorical,
            callbacks=[
                make_lgbm_progress_callback(
                    "Frequency model",
                    total_iterations=model.n_estimators,
                    period=progress_period,
                    enabled=show_progress,
                )
            ],
        )
        valid_pred = model.predict_proba(prepare_lgbm_frame(valid_x, categorical))[:, 1]
        return model, valid_pred, "lightgbm"

    model = make_sklearn_model("classifier", numeric, categorical, random_state)
    model.fit(train_x, train_y)
    valid_pred = model.predict_proba(valid_x)[:, 1]
    return model, valid_pred, "sklearn_hist_gradient_boosting"


def train_severity_model(
    train_x: pd.DataFrame,
    train_amount: pd.Series,
    valid_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    random_state: int,
    force_sklearn: bool,
    progress_period: int,
    show_progress: bool,
):
    y = np.log1p(train_amount)
    if HAS_LIGHTGBM and not force_sklearn:
        model = make_lgbm_regressor(random_state)
        model.fit(
            prepare_lgbm_frame(train_x, categorical),
            y,
            categorical_feature=categorical,
            callbacks=[
                make_lgbm_progress_callback(
                    "Severity model",
                    total_iterations=model.n_estimators,
                    period=progress_period,
                    enabled=show_progress,
                )
            ],
        )
        valid_pred = np.expm1(model.predict(prepare_lgbm_frame(valid_x, categorical))).clip(min=0)
        return model, valid_pred, "lightgbm"

    model = make_sklearn_model("regressor", numeric, categorical, random_state)
    model.fit(train_x, y)
    valid_pred = np.expm1(model.predict(valid_x)).clip(min=0)
    return model, valid_pred, "sklearn_hist_gradient_boosting"


def predict_frequency(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if backend == "lightgbm":
        return model.predict_proba(prepare_lgbm_frame(x, categorical))[:, 1]
    return model.predict_proba(x)[:, 1]


def predict_severity(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if backend == "lightgbm":
        pred = model.predict(prepare_lgbm_frame(x, categorical))
    else:
        pred = model.predict(x)
    return np.expm1(pred).clip(min=0)


def best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def portfolio_loss_ratio(claim_amount: pd.Series | np.ndarray, premium: pd.Series | np.ndarray) -> float:
    premium_sum = float(np.sum(premium))
    if premium_sum <= 0:
        return math.nan
    return float(np.sum(claim_amount) / premium_sum)


def calibrate_pricing(valid: pd.DataFrame, expected_claim: np.ndarray) -> tuple[np.ndarray, PricingCalibration]:
    base_premium = valid["premium"].astype(float).to_numpy()
    actual_claim = valid["claim_amount"].astype(float).to_numpy()

    best: tuple[float, float, float, float, float | None, float | None, float] | None = None
    scales = np.linspace(0.55, 2.50, 80)
    thresholds = np.quantile(expected_claim / np.maximum(base_premium, 1.0), np.linspace(0.0, 0.90, 31))

    for scale in scales:
        raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * scale, 0.0, base_premium * 3.0)
        for threshold in thresholds:
            risk_ratio = expected_claim / np.maximum(base_premium, 1.0)
            proposed = np.where(risk_ratio >= threshold, raw_new_premium, np.minimum(raw_new_premium, base_premium))
            proposed = np.clip(proposed, 0.0, base_premium * 3.0)
            lr_total = portfolio_loss_ratio(actual_claim, proposed)
            increased = proposed > base_premium
            keep_share = float((~increased).mean())
            lr_keep = portfolio_loss_ratio(actual_claim[~increased], proposed[~increased]) if (~increased).any() else None
            lr_inc = portfolio_loss_ratio(actual_claim[increased], proposed[increased]) if increased.any() else None

            group_penalty = 0.0
            if lr_keep is not None:
                group_penalty += abs(lr_keep - TARGET_LOSS_RATIO)
            if lr_inc is not None:
                group_penalty += abs(lr_inc - TARGET_LOSS_RATIO)
            objective = abs(lr_total - TARGET_LOSS_RATIO) + 0.35 * group_penalty - 0.08 * keep_share

            if best is None or objective < best[0]:
                best = (objective, scale, threshold, lr_total, lr_keep, lr_inc, keep_share)

    if best is None:
        raise RuntimeError("Could not calibrate pricing.")

    _, scale, threshold, lr_total, lr_keep, lr_inc, keep_share = best
    risk_ratio = expected_claim / np.maximum(base_premium, 1.0)
    raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * scale, 0.0, base_premium * 3.0)
    calibrated = np.where(risk_ratio >= threshold, raw_new_premium, np.minimum(raw_new_premium, base_premium))
    calibrated = np.clip(calibrated, 0.0, base_premium * 3.0)

    calibration = PricingCalibration(
        scale=float(scale),
        threshold=float(threshold),
        validation_loss_ratio=float(lr_total),
        group_keep_or_decrease_loss_ratio=None if lr_keep is None else float(lr_keep),
        group_increase_loss_ratio=None if lr_inc is None else float(lr_inc),
        keep_or_decrease_share=float(keep_share),
    )
    return calibrated, calibration


def apply_pricing(frame: pd.DataFrame, expected_claim: np.ndarray, calibration: PricingCalibration) -> np.ndarray:
    base_premium = frame["premium"].astype(float).to_numpy()
    risk_ratio = expected_claim / np.maximum(base_premium, 1.0)
    raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * calibration.scale, 0.0, base_premium * 3.0)
    proposed = np.where(
        risk_ratio >= calibration.threshold,
        raw_new_premium,
        np.minimum(raw_new_premium, base_premium),
    )
    return np.clip(proposed, 0.0, base_premium * 3.0)


def evaluate(
    valid: pd.DataFrame,
    claim_probability: np.ndarray,
    severity_pred: np.ndarray,
    expected_claim: np.ndarray,
    new_premium: np.ndarray,
    f1_threshold: float,
) -> dict:
    actual_claim = valid["claim_amount"].astype(float)
    is_claim = valid["is_claim"].astype(int)
    y_pred = claim_probability >= f1_threshold
    positive = actual_claim > 0
    base_premium = valid["premium"].astype(float).to_numpy()
    increased = new_premium > base_premium
    keep_or_decrease = ~increased

    metrics = {
        "frequency": {
            "roc_auc": float(roc_auc_score(is_claim, claim_probability)),
            "gini": float(2 * roc_auc_score(is_claim, claim_probability) - 1),
            "pr_auc": float(average_precision_score(is_claim, claim_probability)),
            "f1_threshold": float(f1_threshold),
            "f1": float(f1_score(is_claim, y_pred)),
            "claim_rate": float(is_claim.mean()),
        },
        "severity": {
            "n_positive_valid": int(positive.sum()),
            "mae_positive": None,
            "rmse_positive": None,
            "r2_positive": None,
        },
        "business": {
            "baseline_loss_ratio_premium": portfolio_loss_ratio(actual_claim, valid["premium"]),
            "baseline_loss_ratio_premium_wo_term": portfolio_loss_ratio(actual_claim, valid["premium_wo_term"]),
            "post_pricing_loss_ratio": portfolio_loss_ratio(actual_claim, new_premium),
            "mean_expected_claim": float(np.mean(expected_claim)),
            "mean_new_premium": float(np.mean(new_premium)),
            "increase_share": float((new_premium > valid["premium"].to_numpy()).mean()),
            "keep_or_decrease_share": float((new_premium <= valid["premium"].to_numpy()).mean()),
            "keep_or_decrease_loss_ratio": portfolio_loss_ratio(
                actual_claim[keep_or_decrease],
                new_premium[keep_or_decrease],
            )
            if keep_or_decrease.any()
            else None,
            "increase_loss_ratio": portfolio_loss_ratio(
                actual_claim[increased],
                new_premium[increased],
            )
            if increased.any()
            else None,
            "min_new_to_old_premium_ratio": float(np.min(new_premium / np.maximum(base_premium, 1.0))),
            "max_new_to_old_premium_ratio": float(np.max(new_premium / np.maximum(base_premium, 1.0))),
        },
    }
    if positive.any():
        metrics["severity"]["mae_positive"] = float(mean_absolute_error(actual_claim[positive], severity_pred[positive]))
        metrics["severity"]["rmse_positive"] = float(
            mean_squared_error(actual_claim[positive], severity_pred[positive]) ** 0.5
        )
        metrics["severity"]["r2_positive"] = float(r2_score(actual_claim[positive], severity_pred[positive]))
    return metrics


def save_pickle(path: Path, obj) -> None:
    with path.open("wb") as file:
        pickle.dump(obj, file)


def main() -> None:
    args = parse_args()
    show_progress = not args.quiet
    artifacts_dir = Path(args.artifacts_dir)
    submission_path = Path(args.submission)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = log_stage(1, f"read train/test CSV ({args.train}, {args.test})", enabled=show_progress)
    raw_train = read_input(args.train, args.train_rows, args.aggregation_backend)
    raw_test = read_input(args.test, args.test_rows, args.aggregation_backend)
    log_stage_done(
        started_at,
        f"loaded {frame_len(raw_train):,} train rows and {frame_len(raw_test):,} test rows",
        enabled=show_progress,
    )

    contract_builder = build_contract_frame_polars if args.aggregation_backend == "polars" else build_contract_frame

    started_at = log_stage(
        2,
        f"aggregate train rows to contract level and build features ({args.aggregation_backend})",
        enabled=show_progress,
    )
    train_contracts = contract_builder(raw_train, is_train=True)
    log_stage_done(started_at, f"built {len(train_contracts):,} train contracts", enabled=show_progress)

    started_at = log_stage(
        3,
        f"aggregate test rows to contract level and build features ({args.aggregation_backend})",
        enabled=show_progress,
    )
    test_contracts = contract_builder(raw_test, is_train=False)
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
        args.force_sklearn,
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
        args.force_sklearn,
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
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log_stage_done(started_at, f"saved models.pkl and metrics.json to {artifacts_dir}", enabled=show_progress)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved submission: {submission_path}")
    print(f"Saved artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()

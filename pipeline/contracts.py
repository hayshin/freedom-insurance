from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from features.car import add_car_features
from features.driver import add_driver_features
from features.engine import add_engine_features
from features.model_mark import add_model_mark_features
from features.premium import add_premium_features
from features.region import add_region_features
from features.score import build_score_features
from features.vehicle_type import add_vehicle_type_features
from pipeline.config import DATE_COLUMNS, LEAKAGE_COLUMNS, PREPROCESSED_SOURCE_COLUMNS
from pipeline.io import polars_to_pandas


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


def add_date_features(frame: pd.DataFrame, raw: pl.DataFrame) -> None:
    if "operation_date" not in raw.columns:
        return
    date_features = raw.group_by("contract_number", maintain_order=True).agg(
        pl.col("operation_date").drop_nulls().first().str.to_date(strict=False).alias("_operation_date")
    ).select(
        [
            "contract_number",
            pl.col("_operation_date").dt.month().alias("operation_month"),
            pl.col("_operation_date").dt.quarter().alias("operation_quarter"),
            pl.col("_operation_date").dt.weekday().sub(1).alias("operation_dayofweek"),
        ]
    )
    indexed = polars_to_pandas(date_features).set_index("contract_number").reindex(frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]


def build_contract_frame(raw: pl.DataFrame, is_train: bool) -> pd.DataFrame:
    if "contract_number" not in raw.columns:
        raise ValueError("Input data must contain contract_number.")
    if is_train:
        assert_contract_targets_are_constant_polars(raw)

    metric_columns = ["premium", "premium_wo_term", "claim_amount", "claim_cnt", "is_claim"]
    base_exprs = [pl_first_non_null(col) for col in metric_columns if col in raw.columns]
    special_exprs: list[pl.Expr] = [pl.len().cast(pl.Int32).alias("n_rows")]
    if "driver_iin" in raw.columns:
        special_exprs.append(pl_nunique_non_null("driver_iin", "n_drivers"))
    if "insurer_iin" in raw.columns:
        special_exprs.append(pl_nunique_non_null("insurer_iin", "n_insurers"))
    if "car_number" in raw.columns:
        special_exprs.append(pl_nunique_non_null("car_number", "n_cars"))
    if "region_id" in raw.columns:
        special_exprs.append(pl_nunique_non_null("region_id", "n_regions"))
    if {"insurer_iin", "driver_iin"}.issubset(raw.columns):
        same_person = pl.col("insurer_iin").fill_null("") == pl.col("driver_iin").fill_null("__driver_missing__")
        special_exprs.extend(
            [
                same_person.max().cast(pl.Int8).alias("insurer_is_driver_any"),
                same_person.min().cast(pl.Int8).alias("insurer_is_driver_all"),
            ]
        )

    grouped = raw.group_by("contract_number", maintain_order=True).agg([*base_exprs, *special_exprs])
    frame = polars_to_pandas(grouped).set_index("contract_number")

    for source, target in [
        ("n_drivers", "is_multi_driver"),
        ("n_cars", "is_multi_car"),
        ("n_regions", "is_multi_region"),
    ]:
        if source in frame:
            frame[target] = (frame[source] > 1).astype("int8")

    add_region_features(raw, frame)
    add_model_mark_features(raw, frame)
    add_car_features(raw, frame)
    add_vehicle_type_features(raw, frame)
    add_engine_features(raw, frame)
    add_driver_features(raw, frame)
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
        if raw.schema[col].is_numeric() and col not in {"ownerkato", "ownerkato_short"}
    ]
    categorical_columns = [col for col in candidate_columns if col not in numeric_columns]
    categorical_columns.extend([col for col in ["ownerkato", "ownerkato_short"] if col in raw.columns])
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
        generic_features = polars_to_pandas(raw.group_by("contract_number", maintain_order=True).agg(agg_exprs))
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

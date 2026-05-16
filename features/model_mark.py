from __future__ import annotations

import re

import pandas as pd
import polars as pl


MISSING_VALUE = "__MISSING__"
PAIR_SEPARATOR = "__"
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_vehicle_text(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING_VALUE
    normalized = _WHITESPACE_RE.sub(" ", str(value).strip().upper())
    if not normalized:
        return MISSING_VALUE
    return normalized


def make_mark_model_pair(mark: object, model: object) -> str:
    return f"{normalize_vehicle_text(mark)}{PAIR_SEPARATOR}{normalize_vehicle_text(model)}"


def mode_or_missing(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return MISSING_VALUE
    mode = non_null.mode(dropna=True)
    if mode.empty:
        return str(non_null.iloc[0])
    return str(mode.iloc[0])


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def _normalize_vehicle_expr(column: str) -> pl.Expr:
    normalized = (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.to_uppercase()
    )
    return pl.when(normalized.is_null() | (normalized == "")).then(pl.lit(MISSING_VALUE)).otherwise(normalized)


def _mode_or_missing_expr(column: str, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .drop_nulls()
        .mode()
        .first()
        .cast(pl.String)
        .fill_null(MISSING_VALUE)
        .alias(alias)
    )


def add_model_mark_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    required = {"contract_number", "mark", "model"}
    if not required.issubset(raw.columns):
        return

    vehicles = raw.select(
        [
            "contract_number",
            _normalize_vehicle_expr("mark").alias("mark_clean"),
            _normalize_vehicle_expr("model").alias("model_clean"),
        ]
    ).with_columns(
        (pl.col("mark_clean") + pl.lit(PAIR_SEPARATOR) + pl.col("model_clean")).alias("mark_model_pair")
    )

    features = vehicles.group_by("contract_number", maintain_order=True).agg(
        [
            _mode_or_missing_expr("mark_clean", "mark_clean_mode"),
            pl.col("mark_clean").drop_nulls().n_unique().cast(pl.Int32).alias("mark_clean_nunique"),
            _mode_or_missing_expr("model_clean", "model_clean_mode"),
            pl.col("model_clean").drop_nulls().n_unique().cast(pl.Int32).alias("model_clean_nunique"),
            _mode_or_missing_expr("mark_model_pair", "mark_model_pair"),
            pl.col("mark_model_pair").drop_nulls().n_unique().cast(pl.Int32).alias("mark_model_pair_nunique"),
        ]
    ).with_columns(
        [
            (pl.col("mark_clean_nunique") > 1).cast(pl.Int8).alias("is_multi_mark"),
            (pl.col("model_clean_nunique") > 1).cast(pl.Int8).alias("is_multi_model"),
            (pl.col("mark_model_pair_nunique") > 1).cast(pl.Int8).alias("is_multi_mark_model_pair"),
        ]
    )
    indexed = _to_indexed_pandas(features, frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]

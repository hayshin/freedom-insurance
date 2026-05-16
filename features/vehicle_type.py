from __future__ import annotations

import pandas as pd
import polars as pl


MISSING_VALUE = "MISSING"


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def _normalize_vehicle_type_expr() -> pl.Expr:
    normalized = pl.col("vehicle_type_name").cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()
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


def add_vehicle_type_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    if "contract_number" not in raw.columns or "vehicle_type_name" not in raw.columns:
        return

    normalized = _normalize_vehicle_type_expr()
    typed = raw.select([
        "contract_number",
        normalized.alias("vehicle_type_name_normalized"),
    ])

    features = typed.group_by("contract_number", maintain_order=True).agg(
        [
            _mode_or_missing_expr("vehicle_type_name_normalized", "vehicle_type_name_mode"),
            pl.col("vehicle_type_name_normalized")
            .drop_nulls()
            .n_unique()
            .cast(pl.Int32)
            .alias("vehicle_type_name_nunique"),
        ]
    ).with_columns(
        (pl.col("vehicle_type_name_nunique") > 1).cast(pl.Int8).alias("is_multi_vehicle_type_name")
    )

    indexed = _to_indexed_pandas(features, frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def _clean_positive(column: str) -> pl.Expr:
    values = pl.col(column).cast(pl.Float64, strict=False)
    return pl.when(values > 0).then(values).otherwise(None)


def add_engine_features(
    raw: pl.DataFrame,
    frame: pd.DataFrame,
    outlier_caps: dict[str, tuple[float, float]] | None = None,
) -> None:
    if "contract_number" not in raw.columns:
        return

    available = {col for col in ["engine_volume", "engine_power"] if col in raw.columns}
    if not available:
        return

    volume = _clean_positive("engine_volume") if "engine_volume" in raw.columns else pl.lit(None, dtype=pl.Float64)
    power = _clean_positive("engine_power") if "engine_power" in raw.columns else pl.lit(None, dtype=pl.Float64)
    ratio = pl.when(volume.is_not_null() & power.is_not_null()).then(power / volume).otherwise(None)
    if outlier_caps and "engine_power_per_volume" in outlier_caps:
        lower, upper = outlier_caps["engine_power_per_volume"]
        ratio = ratio.clip(lower, upper)

    rows = raw.select(
        [
            "contract_number",
            volume.alias("_engine_volume"),
            power.alias("_engine_power"),
            ratio.alias("_engine_power_per_volume"),
            volume.is_null().cast(pl.Float64).alias("_engine_volume_missing"),
            power.is_null().cast(pl.Float64).alias("_engine_power_missing"),
        ]
    )

    agg_exprs: list[pl.Expr] = []
    if "engine_volume" in raw.columns:
        agg_exprs.extend(
            [
                pl.col("_engine_volume").min().alias("engine_volume_min"),
                pl.col("_engine_volume").max().alias("engine_volume_max"),
                pl.col("_engine_volume").mean().alias("engine_volume_mean"),
                pl.col("_engine_volume").std().alias("engine_volume_std"),
                pl.col("_engine_volume").drop_nulls().n_unique().cast(pl.Int32).alias("engine_volume_nunique"),
                pl.col("_engine_volume_missing").mean().fill_null(0.0).alias("engine_volume_missing_share"),
            ]
        )
    if "engine_power" in raw.columns:
        agg_exprs.extend(
            [
                pl.col("_engine_power").min().alias("engine_power_min"),
                pl.col("_engine_power").max().alias("engine_power_max"),
                pl.col("_engine_power").mean().alias("engine_power_mean"),
                pl.col("_engine_power").std().alias("engine_power_std"),
                pl.col("_engine_power").drop_nulls().n_unique().cast(pl.Int32).alias("engine_power_nunique"),
                pl.col("_engine_power_missing").mean().fill_null(0.0).alias("engine_power_missing_share"),
            ]
        )
    if {"engine_volume", "engine_power"}.issubset(raw.columns):
        agg_exprs.extend(
            [
                pl.col("_engine_power_per_volume").mean().alias("engine_power_per_volume_mean"),
                pl.col("_engine_power_per_volume").min().alias("engine_power_per_volume_min"),
                pl.col("_engine_power_per_volume").max().alias("engine_power_per_volume_max"),
                pl.col("_engine_power_per_volume").std().alias("engine_power_per_volume_std"),
            ]
        )

    if not agg_exprs:
        return

    features = rows.group_by("contract_number", maintain_order=True).agg(agg_exprs)

    extra_columns: list[pl.Expr] = []
    if "engine_volume" in raw.columns:
        extra_columns.extend(
            [
                pl.col("engine_volume_mean").clip(0, None).log1p().alias("engine_volume_mean_log1p"),
                (pl.col("engine_volume_nunique") > 1).fill_null(False).cast(pl.Int8).alias(
                    "is_multi_engine_volume"
                ),
            ]
        )
    if "engine_power" in raw.columns:
        extra_columns.extend(
            [
                pl.col("engine_power_mean").clip(0, None).log1p().alias("engine_power_mean_log1p"),
                (pl.col("engine_power_nunique") > 1).fill_null(False).cast(pl.Int8).alias(
                    "is_multi_engine_power"
                ),
            ]
        )
    if extra_columns:
        features = features.with_columns(extra_columns)

    indexed = _to_indexed_pandas(features, frame.index).replace([np.inf, -np.inf], np.nan)
    for col in indexed.columns:
        frame[col] = indexed[col]

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import polars as pl


SCORE_GROUP_RE = re.compile(r"^SCORE_(\d+)_")


def score_group(column_name: str) -> str | None:
    match = SCORE_GROUP_RE.match(column_name)
    if match is None:
        return None
    return match.group(1)


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def build_score_features(raw: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if "contract_number" not in raw.columns:
        return pd.DataFrame(index=index)

    score_cols = [col for col in raw.columns if col.startswith("SCORE_")]
    if not score_cols:
        return pd.DataFrame(index=index)

    score_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in score_cols]
    available_exprs = [expr.is_not_null().cast(pl.Int32) for expr in score_exprs]
    missing_exprs = [expr.is_null().cast(pl.Int32) for expr in score_exprs]
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

    rows = raw.select(
        "contract_number",
        score_available.alias("_score_available_count"),
        pl.mean_horizontal(missing_exprs).alias("_score_missing_rate"),
        pl.mean_horizontal(score_exprs).alias("_score_row_mean"),
        pl.min_horizontal(score_exprs).alias("_score_row_min"),
        pl.max_horizontal(score_exprs).alias("_score_row_max"),
        score_row_std.alias("_score_row_std"),
        *group_row_columns,
    )
    features = rows.group_by("contract_number", maintain_order=True).agg(
        [
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
    )
    score_features = _to_indexed_pandas(features, index).replace([np.inf, -np.inf], np.nan)
    return score_features


def add_score_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    score_features = build_score_features(raw, frame.index)
    if score_features.empty:
        return
    frame[score_features.columns] = score_features

from __future__ import annotations

from typing import Iterable

import numpy as np
import polars as pl


DEFAULT_QUANTILES = (0.01, 0.99)


def _quantile_value(raw: pl.DataFrame, expr: pl.Expr, q: float) -> float | None:
    if raw.is_empty():
        return None
    result = raw.select(expr.drop_nulls().quantile(q, interpolation="nearest").alias("q")).to_series(0)[0]
    if result is None:
        return None
    value = float(result)
    if np.isnan(value):
        return None
    return value


def compute_outlier_caps(
    raw: pl.DataFrame,
    columns: Iterable[str],
    quantiles: tuple[float, float] = DEFAULT_QUANTILES,
) -> dict[str, tuple[float, float]]:
    caps: dict[str, tuple[float, float]] = {}
    lower_q, upper_q = quantiles
    for column in columns:
        if column not in raw.columns:
            continue
        values = pl.col(column).cast(pl.Float64, strict=False)
        lower = _quantile_value(raw, values, lower_q)
        upper = _quantile_value(raw, values, upper_q)
        if lower is None or upper is None:
            continue
        if lower >= upper:
            continue
        caps[column] = (lower, upper)
    return caps


def compute_engine_ratio_caps(
    raw: pl.DataFrame,
    power_column: str = "engine_power",
    volume_column: str = "engine_volume",
    quantiles: tuple[float, float] = DEFAULT_QUANTILES,
) -> dict[str, tuple[float, float]]:
    if power_column not in raw.columns or volume_column not in raw.columns:
        return {}
    power = pl.col(power_column).cast(pl.Float64, strict=False)
    volume = pl.col(volume_column).cast(pl.Float64, strict=False)
    ratio = pl.when(volume > 0).then(power / volume).otherwise(None)
    lower = _quantile_value(raw, ratio, quantiles[0])
    upper = _quantile_value(raw, ratio, quantiles[1])
    if lower is None or upper is None:
        return {}
    if lower >= upper:
        return {}
    return {"engine_power_per_volume": (lower, upper)}


def apply_outlier_caps(raw: pl.DataFrame, caps: dict[str, tuple[float, float]] | None) -> pl.DataFrame:
    if not caps:
        return raw

    expressions: list[pl.Expr] = []
    for column in raw.columns:
        if column in caps:
            lower, upper = caps[column]
            expr = pl.col(column).cast(pl.Float64, strict=False).clip(lower, upper).alias(column)
        else:
            expr = pl.col(column)
        expressions.append(expr)
    return raw.select(expressions)

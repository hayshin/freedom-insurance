from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan))


def add_premium_features(frame: pd.DataFrame) -> None:
    if "premium" not in frame:
        return

    cols = ["premium"]
    if "n_drivers" in frame:
        cols.append("n_drivers")
    if "n_cars" in frame:
        cols.append("n_cars")
    if "premium_wo_term" in frame:
        cols.append("premium_wo_term")

    data = pl.from_dict({col: frame[col].to_list() for col in cols})
    premium = pl.col("premium").cast(pl.Float64, strict=False)
    expressions = [
        premium.clip(0, None).log1p().alias("premium_log1p"),
        (premium == 0).cast(pl.Int8).alias("premium_is_zero"),
        premium.is_null().cast(pl.Int8).alias("premium_is_missing"),
    ]

    if "n_drivers" in frame:
        n_drivers = pl.col("n_drivers").cast(pl.Float64, strict=False)
        premium_per_driver = pl.when(n_drivers != 0).then(premium / n_drivers).otherwise(None).fill_null(0.0)
        expressions.extend(
            [
                premium_per_driver.alias("premium_per_driver"),
                premium_per_driver.clip(0, None).log1p().alias("premium_per_driver_log1p"),
            ]
        )
    if "n_cars" in frame:
        n_cars = pl.col("n_cars").cast(pl.Float64, strict=False)
        premium_per_car = pl.when(n_cars != 0).then(premium / n_cars).otherwise(None).fill_null(0.0)
        expressions.extend(
            [
                premium_per_car.alias("premium_per_car"),
                premium_per_car.clip(0, None).log1p().alias("premium_per_car_log1p"),
            ]
        )

    if "premium_wo_term" in frame:
        premium_wo_term = pl.col("premium_wo_term").cast(pl.Float64, strict=False)
        ratio = pl.when(premium != 0).then(premium_wo_term / premium).otherwise(None).fill_null(0.0)
        expressions.extend(
            [
                premium_wo_term.clip(0, None).log1p().alias("premium_wo_term_log1p"),
                (premium_wo_term == 0).cast(pl.Int8).alias("premium_wo_term_is_zero"),
                premium_wo_term.is_null().cast(pl.Int8).alias("premium_wo_term_is_missing"),
                ratio.alias("premium_wo_term_ratio"),
                (1.0 - ratio).clip(0.0, 1.0).alias("premium_return_ratio"),
            ]
        )

    features = data.select(expressions)
    for col, values in features.to_dict(as_series=False).items():
        frame[col] = values

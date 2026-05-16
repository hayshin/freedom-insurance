from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl


REFERENCE_YEAR = 2022


def parse_is_seven_year_car(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    normalized = str(value).strip().lower()
    if not normalized:
        return np.nan
    if "до 7" in normalized or "меньше 7" in normalized:
        return 1.0
    if "свыше 7" in normalized or "больше 7" in normalized:
        return 0.0
    return np.nan


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def add_car_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    required = {"contract_number", "car_year"}
    if not required.issubset(raw.columns):
        return

    car_year = pl.col("car_year").cast(pl.Float64, strict=False)
    age = pl.when(car_year.is_between(1900, REFERENCE_YEAR)).then(REFERENCE_YEAR - car_year).otherwise(None)
    is_seven = pl.when(age.is_not_null()).then(age <= 7).otherwise(None)

    expressions = [pl.col("contract_number"), age.alias("car_age_years"), is_seven.cast(pl.Float64).alias("is_seven_year_car")]
    if "car_age" in raw.columns:
        normalized_age = pl.col("car_age").cast(pl.String, strict=False).str.to_lowercase()
        fallback = (
            pl.when(normalized_age.str.contains("до 7|меньше 7", literal=False))
            .then(1.0)
            .when(normalized_age.str.contains("свыше 7|больше 7", literal=False))
            .then(0.0)
            .otherwise(None)
        )
        expressions[-1] = pl.coalesce(is_seven.cast(pl.Float64), fallback).alias("is_seven_year_car")

    car = raw.select(expressions)
    features = car.group_by("contract_number", maintain_order=True).agg(
        [
            pl.col("car_age_years").mean().alias("car_age"),
            pl.col("car_age_years").min().alias("car_age_min"),
            pl.col("car_age_years").max().alias("car_age_max"),
            pl.col("car_age_years").std().alias("car_age_std"),
            pl.col("car_age_years").drop_nulls().n_unique().cast(pl.Int32).alias("car_age_nunique"),
            pl.col("is_seven_year_car").max().fill_null(0).cast(pl.Int8).alias("is_seven_year_car"),
            pl.col("is_seven_year_car").min().fill_null(0).cast(pl.Int8).alias("is_all_seven_year_car"),
            pl.col("is_seven_year_car").mean().fill_null(0.0).alias("seven_year_car_share"),
        ]
    )
    indexed = _to_indexed_pandas(features, frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]

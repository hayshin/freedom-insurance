from __future__ import annotations

import numpy as np
import pandas as pd


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


def add_car_features(raw: pd.DataFrame, frame: pd.DataFrame) -> None:
    required = {"contract_number", "car_year"}
    if not required.issubset(raw.columns):
        return

    car = raw[["contract_number", "car_year"]].copy()
    car_year = pd.to_numeric(car["car_year"], errors="coerce")
    car["car_age_years"] = (REFERENCE_YEAR - car_year).where(car_year.between(1900, REFERENCE_YEAR))
    car["is_seven_year_car"] = car["car_age_years"].le(7).where(car["car_age_years"].notna())

    if "car_age" in raw:
        fallback = raw["car_age"].map(parse_is_seven_year_car)
        car["is_seven_year_car"] = car["is_seven_year_car"].fillna(fallback)

    grouped = car.groupby("contract_number", sort=False)
    car_age = grouped["car_age_years"].agg(["min", "max", "mean", "std"])
    car_age.columns = [f"car_age_{suffix}" for suffix in car_age.columns]
    frame["car_age"] = car_age["car_age_mean"]
    frame["car_age_min"] = car_age["car_age_min"]
    frame["car_age_max"] = car_age["car_age_max"]
    frame["car_age_std"] = car_age["car_age_std"]
    frame["car_age_nunique"] = grouped["car_age_years"].nunique(dropna=True).astype("int32")

    seven_year = grouped["is_seven_year_car"].agg(["max", "min", "mean"])
    frame["is_seven_year_car"] = seven_year["max"].fillna(0).astype("int8")
    frame["is_all_seven_year_car"] = seven_year["min"].fillna(0).astype("int8")
    frame["seven_year_car_share"] = seven_year["mean"].fillna(0.0)

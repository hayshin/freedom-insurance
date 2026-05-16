from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.divide(denominator.replace(0, np.nan))


def add_premium_features(frame: pd.DataFrame) -> None:
    if "premium" not in frame:
        return

    premium = pd.to_numeric(frame["premium"], errors="coerce").astype(float)
    frame["premium_log1p"] = np.log1p(premium.clip(lower=0))
    frame["premium_is_zero"] = premium.eq(0).astype("int8")
    frame["premium_is_missing"] = premium.isna().astype("int8")

    if "n_drivers" in frame:
        frame["premium_per_driver"] = safe_divide(premium, frame["n_drivers"].astype(float)).fillna(0.0)
        frame["premium_per_driver_log1p"] = np.log1p(frame["premium_per_driver"].clip(lower=0))
    if "n_cars" in frame:
        frame["premium_per_car"] = safe_divide(premium, frame["n_cars"].astype(float)).fillna(0.0)
        frame["premium_per_car_log1p"] = np.log1p(frame["premium_per_car"].clip(lower=0))

    if "premium_wo_term" not in frame:
        return

    premium_wo_term = pd.to_numeric(frame["premium_wo_term"], errors="coerce").astype(float)
    frame["premium_wo_term_log1p"] = np.log1p(premium_wo_term.clip(lower=0))
    frame["premium_wo_term_is_zero"] = premium_wo_term.eq(0).astype("int8")
    frame["premium_wo_term_is_missing"] = premium_wo_term.isna().astype("int8")
    frame["premium_wo_term_ratio"] = safe_divide(premium_wo_term, premium).fillna(0.0)
    frame["premium_return_ratio"] = (1.0 - frame["premium_wo_term_ratio"]).clip(lower=0.0, upper=1.0)

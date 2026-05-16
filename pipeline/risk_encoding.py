from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ID_RISK_COLUMNS = [
    "driver_iin_mode",
    "insurer_iin_mode",
    "car_number_mode",
]
BASE_RISK_COLUMNS = [
    "region_macro_mode",
    "region_climate_mode",
    "mark_clean_mode",
    "model_clean_mode",
    "mark_model_pair",
    "vehicle_type_name_mode",
    "car_age",
    "operation_month",
    "operation_quarter",
    "ownerkato_short_mode",
]
BINNED_RISK_COLUMNS = [
    "bonus_malus_coef_max",
    "bonus_malus_coef_mean",
    "driver_experience_year_min",
    "driver_experience_year_mean",
    "engine_power_mean",
    "engine_volume_mean",
    "score_row_mean_mean",
    "score_row_min_min",
    "score_row_max_max",
    "premium_log1p",
    "premium_per_driver_log1p",
]
COMBO_RISK_COLUMNS = [
    ("region_macro_mode", "vehicle_type_name_mode"),
    ("region_macro_mode", "bonus_malus_coef_max_bin"),
    ("vehicle_type_name_mode", "bonus_malus_coef_max_bin"),
    ("mark_clean_mode", "vehicle_type_name_mode"),
    ("car_age", "vehicle_type_name_mode"),
    ("score_row_mean_mean_bin", "bonus_malus_coef_max_bin"),
]


@dataclass
class RiskEncodingInfo:
    prior: float
    smoothing: float
    n_splits: int
    encoded_columns: list[str]


def _string_values(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str)


def _fit_smoothed_stats(keys: pd.Series, target: pd.Series, prior: float, smoothing: float) -> pd.DataFrame:
    stats = pd.DataFrame({"key": _string_values(keys), "target": target.astype(float)})
    grouped = stats.groupby("key", observed=True)["target"].agg(["sum", "count"])
    grouped["rate"] = (grouped["sum"] + prior * smoothing) / (grouped["count"] + smoothing)
    grouped["log_count"] = np.log1p(grouped["count"].astype(float))
    return grouped[["rate", "log_count"]]


def _apply_stats(keys: pd.Series, stats: pd.DataFrame, prior: float) -> tuple[pd.Series, pd.Series]:
    mapped = _string_values(keys).map(stats["rate"]).fillna(prior).astype(float)
    counts = _string_values(keys).map(stats["log_count"]).fillna(0.0).astype(float)
    return mapped, counts


def _add_quantile_bins(train: pd.DataFrame, frames: list[pd.DataFrame], columns: list[str]) -> list[str]:
    added: list[str] = []
    for col in columns:
        if col not in train.columns:
            continue
        train_values = pd.to_numeric(train[col], errors="coerce")
        non_null = train_values.dropna()
        if non_null.nunique() < 5:
            continue
        _, edges = pd.qcut(non_null, q=8, duplicates="drop", retbins=True)
        if len(edges) < 3:
            continue
        edges = np.unique(edges)
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_col = f"{col}_bin"
        for frame in frames:
            values = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else pd.Series(np.nan, index=frame.index)
            frame[bin_col] = pd.cut(values, bins=edges, labels=False, include_lowest=True).astype("float").fillna(-1).astype(int)
        added.append(bin_col)
    return added


def _add_combo_columns(frames: list[pd.DataFrame], combos: list[tuple[str, str]]) -> list[str]:
    added: list[str] = []
    for left, right in combos:
        if not all(left in frame.columns and right in frame.columns for frame in frames):
            continue
        combo_col = f"{left}__x__{right}"
        for frame in frames:
            frame[combo_col] = _string_values(frame[left]) + "|" + _string_values(frame[right])
        added.append(combo_col)
    return added


def add_oof_risk_encoding_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_state: int,
    n_splits: int = 5,
    smoothing: float = 80.0,
    include_id_columns: bool = False,
) -> RiskEncodingInfo:
    if "is_claim" not in train.columns:
        raise ValueError("Risk encoding requires is_claim in train frame.")

    frames = [train, valid, test]
    bin_columns = _add_quantile_bins(train, frames, BINNED_RISK_COLUMNS)
    combo_columns = _add_combo_columns(frames, COMBO_RISK_COLUMNS)
    base_columns = [*BASE_RISK_COLUMNS]
    if include_id_columns:
        base_columns = [*ID_RISK_COLUMNS, *base_columns]
    candidate_columns = [
        col
        for col in [*base_columns, *bin_columns, *combo_columns]
        if col in train.columns and train[col].nunique(dropna=False) > 1
    ]

    prior = float(train["is_claim"].mean())
    y = train["is_claim"].astype(int)
    actual_splits = min(max(2, n_splits), int(y.value_counts().min()))
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    encoded_columns: list[str] = []

    for col in candidate_columns:
        rate_col = f"risk_{col}_claim_rate_oof"
        count_col = f"risk_{col}_log_count"
        train[rate_col] = prior
        train[count_col] = 0.0

        for fold_train_idx, fold_valid_idx in splitter.split(train, y):
            fold_stats = _fit_smoothed_stats(train.iloc[fold_train_idx][col], y.iloc[fold_train_idx], prior, smoothing)
            rates, counts = _apply_stats(train.iloc[fold_valid_idx][col], fold_stats, prior)
            train.iloc[fold_valid_idx, train.columns.get_loc(rate_col)] = rates.to_numpy()
            train.iloc[fold_valid_idx, train.columns.get_loc(count_col)] = counts.to_numpy()

        full_stats = _fit_smoothed_stats(train[col], y, prior, smoothing)
        for frame in [valid, test]:
            rates, counts = _apply_stats(frame[col], full_stats, prior)
            frame[rate_col] = rates.to_numpy()
            frame[count_col] = counts.to_numpy()

        train[f"risk_{col}_claim_lift_oof"] = train[rate_col] / max(prior, 1e-9)
        valid[f"risk_{col}_claim_lift_oof"] = valid[rate_col] / max(prior, 1e-9)
        test[f"risk_{col}_claim_lift_oof"] = test[rate_col] / max(prior, 1e-9)
        encoded_columns.extend([rate_col, count_col, f"risk_{col}_claim_lift_oof"])

    return RiskEncodingInfo(
        prior=prior,
        smoothing=float(smoothing),
        n_splits=int(actual_splits),
        encoded_columns=encoded_columns,
    )

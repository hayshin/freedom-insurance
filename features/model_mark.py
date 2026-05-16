from __future__ import annotations

import re

import pandas as pd


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


def add_model_mark_features(raw: pd.DataFrame, frame: pd.DataFrame) -> None:
    required = {"contract_number", "mark", "model"}
    if not required.issubset(raw.columns):
        return

    vehicles = raw[["contract_number", "mark", "model"]].copy()
    vehicles["mark_clean"] = vehicles["mark"].map(normalize_vehicle_text)
    vehicles["model_clean"] = vehicles["model"].map(normalize_vehicle_text)
    vehicles["mark_model_pair"] = [
        make_mark_model_pair(mark, model)
        for mark, model in zip(vehicles["mark_clean"], vehicles["model_clean"], strict=False)
    ]

    grouped = vehicles.groupby("contract_number", sort=False)
    frame["mark_clean_mode"] = grouped["mark_clean"].agg(mode_or_missing)
    frame["mark_clean_nunique"] = grouped["mark_clean"].nunique(dropna=True).astype("int32")
    frame["model_clean_mode"] = grouped["model_clean"].agg(mode_or_missing)
    frame["model_clean_nunique"] = grouped["model_clean"].nunique(dropna=True).astype("int32")
    frame["mark_model_pair"] = grouped["mark_model_pair"].agg(mode_or_missing)
    frame["mark_model_pair_nunique"] = grouped["mark_model_pair"].nunique(dropna=True).astype("int32")
    frame["is_multi_mark"] = (frame["mark_clean_nunique"] > 1).astype("int8")
    frame["is_multi_model"] = (frame["model_clean_nunique"] > 1).astype("int8")
    frame["is_multi_mark_model_pair"] = (frame["mark_model_pair_nunique"] > 1).astype("int8")

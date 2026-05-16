from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import HIGH_CARDINALITY_COLUMNS, LEAKAGE_COLUMNS


def is_categorical_feature(series: pd.Series, column_name: str) -> bool:
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or column_name in HIGH_CARDINALITY_COLUMNS
    )


def build_feature_lists(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_columns = [
        col
        for col in frame.columns
        if col not in LEAKAGE_COLUMNS
        and col not in {"loss_ratio"}
        and not col.startswith("claim_amount")
        and not col.startswith("claim_cnt")
        and not col.startswith("is_claim")
    ]
    categorical = [col for col in feature_columns if is_categorical_feature(frame[col], col)]
    numeric = [col for col in feature_columns if col not in categorical]
    leaked = sorted(set(feature_columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"Leakage columns in features: {leaked}")
    return feature_columns, numeric, categorical


def apply_rare_categories(
    train: pd.DataFrame,
    other_frames: Iterable[pd.DataFrame],
    categorical_columns: list[str],
    min_count: int,
) -> dict[str, list[str]]:
    vocab: dict[str, list[str]] = {}
    for col in categorical_columns:
        train[col] = train[col].fillna("__MISSING__").astype(str)
        counts = train[col].value_counts(dropna=False)
        keep = set(counts[counts >= min_count].index.astype(str))
        vocab[col] = sorted(keep)
        train[col] = train[col].where(train[col].isin(keep), "__RARE__")
        for frame in other_frames:
            frame[col] = frame[col].fillna("__MISSING__").astype(str)
            frame[col] = frame[col].where(frame[col].isin(keep), "__RARE__")
    return vocab


def align_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_in_test = [col for col in train.columns if col not in test.columns]
    missing_in_train = [col for col in test.columns if col not in train.columns]
    for col in missing_in_test:
        test[col] = np.nan
    for col in missing_in_train:
        train[col] = np.nan
    return train, test[train.columns]


def split_train_valid(
    contracts: pd.DataFrame, valid_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "operation_month" in contracts:
        cutoff = contracts["operation_month"].quantile(1.0 - valid_size)
        train = contracts[contracts["operation_month"] <= cutoff].copy()
        valid = contracts[contracts["operation_month"] > cutoff].copy()
        if len(valid) > 0 and valid["is_claim"].sum() > 0 and len(train) > len(valid):
            return train, valid

    train_idx, valid_idx = train_test_split(
        contracts.index,
        test_size=valid_size,
        random_state=random_state,
        stratify=contracts["is_claim"],
    )
    return contracts.loc[train_idx].copy(), contracts.loc[valid_idx].copy()

from __future__ import annotations

import re

import numpy as np
import pandas as pd


SCORE_GROUP_RE = re.compile(r"^SCORE_(\d+)_")


def score_group(column_name: str) -> str | None:
    match = SCORE_GROUP_RE.match(column_name)
    if match is None:
        return None
    return match.group(1)


def build_score_features(raw: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if "contract_number" not in raw:
        return pd.DataFrame(index=index)

    score_cols = [col for col in raw.columns if col.startswith("SCORE_")]
    if not score_cols:
        return pd.DataFrame(index=index)

    grouped_keys = raw["contract_number"]
    scores = raw[score_cols].apply(pd.to_numeric, errors="coerce")

    row_available = scores.notna().sum(axis=1)
    row_missing_rate = scores.isna().mean(axis=1)
    row_mean = scores.mean(axis=1)
    row_min = scores.min(axis=1)
    row_max = scores.max(axis=1)
    row_std = scores.std(axis=1)

    features: dict[str, pd.Series] = {
        "score_available_count_mean": row_available.groupby(grouped_keys, sort=False).mean(),
        "score_available_count_min": row_available.groupby(grouped_keys, sort=False).min().astype("int32"),
        "score_missing_rate_mean": row_missing_rate.groupby(grouped_keys, sort=False).mean(),
        "score_missing_rate_max": row_missing_rate.groupby(grouped_keys, sort=False).max(),
        "score_row_mean_mean": row_mean.groupby(grouped_keys, sort=False).mean(),
        "score_row_mean_min": row_mean.groupby(grouped_keys, sort=False).min(),
        "score_row_mean_max": row_mean.groupby(grouped_keys, sort=False).max(),
        "score_row_mean_std": row_mean.groupby(grouped_keys, sort=False).std(),
        "score_row_min_mean": row_min.groupby(grouped_keys, sort=False).mean(),
        "score_row_min_min": row_min.groupby(grouped_keys, sort=False).min(),
        "score_row_max_mean": row_max.groupby(grouped_keys, sort=False).mean(),
        "score_row_max_max": row_max.groupby(grouped_keys, sort=False).max(),
        "score_row_std_mean": row_std.groupby(grouped_keys, sort=False).mean(),
        "score_row_std_max": row_std.groupby(grouped_keys, sort=False).max(),
    }

    grouped_columns: dict[str, list[str]] = {}
    for col in score_cols:
        group = score_group(col)
        if group is not None:
            grouped_columns.setdefault(group, []).append(col)

    for group, cols in sorted(grouped_columns.items(), key=lambda item: int(item[0])):
        group_scores = scores[cols]
        group_row_mean = group_scores.mean(axis=1)
        group_missing_rate = group_scores.isna().mean(axis=1)
        prefix = f"score_g{group}"
        features[f"{prefix}_mean_mean"] = group_row_mean.groupby(grouped_keys, sort=False).mean()
        features[f"{prefix}_mean_min"] = group_row_mean.groupby(grouped_keys, sort=False).min()
        features[f"{prefix}_mean_max"] = group_row_mean.groupby(grouped_keys, sort=False).max()
        features[f"{prefix}_mean_std"] = group_row_mean.groupby(grouped_keys, sort=False).std()
        features[f"{prefix}_missing_rate_mean"] = group_missing_rate.groupby(grouped_keys, sort=False).mean()

    score_features = pd.DataFrame(features, index=index).replace([np.inf, -np.inf], np.nan)
    return score_features


def add_score_features(raw: pd.DataFrame, frame: pd.DataFrame) -> None:
    score_features = build_score_features(raw, frame.index)
    if score_features.empty:
        return
    frame[score_features.columns] = score_features

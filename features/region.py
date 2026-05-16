from __future__ import annotations

import pandas as pd
import polars as pl


MISSING_REGION = "MISSING"
OTHER_REGION = "other"
MULTI_REGION = "MULTI"

REGION_MACRO_MAP = {
    "03 Восточно-Казахстанская область": "east",
    "19 Абайская область": "east",
    "05 Карагандинская область": "central",
    "21 Улытауская область": "central",
    "13 Атырауская область": "west",
    "11 Западно-Казахстанская область": "west",
    "10 Актюбинская область": "west",
    "14 Мангистауская область": "west",
    "07 Акмолинская область": "north",
    "04 Костанайская область": "north",
    "06 Северо-Казахстанская область": "north",
    "08 Павлодарская область": "north",
    "16 Астана": "north",
    "01 Алматинская область": "south",
    "09 Жамбылская область": "south",
    "12 Кызылординская область": "south",
    "15 Алматы": "south",
    "17 Шымкент": "south",
    "18 Туркестанская область": "south",
    "20 Жетысуйская область": "south",
    "Временная регистрация": OTHER_REGION,
    "Временный въезд": OTHER_REGION,
    MULTI_REGION: MULTI_REGION,
}

REGION_CLIMATE_MAP = {
    "07 Акмолинская область": "cold_steppe",
    "04 Костанайская область": "cold_steppe",
    "06 Северо-Казахстанская область": "cold_steppe",
    "08 Павлодарская область": "cold_steppe",
    "03 Восточно-Казахстанская область": "mountain_cold",
    "19 Абайская область": "mountain_cold",
    "05 Карагандинская область": "dry_continental",
    "21 Улытауская область": "dry_continental",
    "13 Атырауская область": "arid_hot",
    "11 Западно-Казахстанская область": "arid_hot",
    "10 Актюбинская область": "arid_hot",
    "14 Мангистауская область": "arid_hot",
    "09 Жамбылская область": "hot_south",
    "12 Кызылординская область": "hot_south",
    "18 Туркестанская область": "hot_south",
    "01 Алматинская область": "foothill",
    "20 Жетысуйская область": "foothill",
    "15 Алматы": "metro_foothill",
    "16 Астана": "metro_cold_steppe",
    "17 Шымкент": "metro_hot",
    "Временная регистрация": OTHER_REGION,
    "Временный въезд": OTHER_REGION,
    MULTI_REGION: MULTI_REGION,
}


def normalize_region_name(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING_REGION
    normalized = str(value).strip()
    if not normalized:
        return MISSING_REGION
    return normalized


def map_region_macro(region_name: object) -> str:
    normalized = normalize_region_name(region_name)
    if normalized == MISSING_REGION:
        return MISSING_REGION
    return REGION_MACRO_MAP.get(normalized, OTHER_REGION)


def map_region_climate(region_name: object) -> str:
    normalized = normalize_region_name(region_name)
    if normalized == MISSING_REGION:
        return MISSING_REGION
    return REGION_CLIMATE_MAP.get(normalized, OTHER_REGION)


def mode_or_missing(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return MISSING_REGION
    mode = non_null.mode(dropna=True)
    if mode.empty:
        return str(non_null.iloc[0])
    return str(mode.iloc[0])


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def _normalize_region_expr(column: str) -> pl.Expr:
    normalized = pl.col(column).cast(pl.String, strict=False).str.strip_chars()
    return pl.when(normalized.is_null() | (normalized == "")).then(pl.lit(MISSING_REGION)).otherwise(normalized)


def _map_values_expr(source: pl.Expr, mapping: dict[str, str], alias: str) -> pl.Expr:
    mapped = pl.when(source == MISSING_REGION).then(pl.lit(MISSING_REGION))
    for key, value in mapping.items():
        mapped = mapped.when(source == key).then(pl.lit(value))
    return mapped.otherwise(pl.lit(OTHER_REGION)).alias(alias)


def _mode_or_missing_expr(column: str, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .drop_nulls()
        .mode()
        .first()
        .cast(pl.String)
        .fill_null(MISSING_REGION)
        .alias(alias)
    )


def add_region_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    if "contract_number" not in raw.columns or "region_name" not in raw.columns:
        return

    normalized = _normalize_region_expr("region_name")
    region = raw.select(
        [
            "contract_number",
            normalized.alias("region_name_normalized"),
            _map_values_expr(normalized, REGION_MACRO_MAP, "region_macro"),
            _map_values_expr(normalized, REGION_CLIMATE_MAP, "region_climate"),
        ]
    )

    features = region.group_by("contract_number", maintain_order=True).agg(
        [
            _mode_or_missing_expr("region_macro", "region_macro_mode"),
            pl.col("region_macro").drop_nulls().n_unique().cast(pl.Int32).alias("region_macro_nunique"),
            _mode_or_missing_expr("region_climate", "region_climate_mode"),
            pl.col("region_climate").drop_nulls().n_unique().cast(pl.Int32).alias("region_climate_nunique"),
        ]
    ).with_columns(
        [
            (pl.col("region_macro_nunique") > 1).cast(pl.Int8).alias("is_multi_region_macro"),
            (pl.col("region_climate_nunique") > 1).cast(pl.Int8).alias("is_multi_region_climate"),
        ]
    )
    indexed = _to_indexed_pandas(features, frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]

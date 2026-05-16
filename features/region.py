from __future__ import annotations

import pandas as pd


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


def add_region_features(raw: pd.DataFrame, frame: pd.DataFrame) -> None:
    if "contract_number" not in raw or "region_name" not in raw:
        return

    region = raw[["contract_number", "region_name"]].copy()
    region["region_name_normalized"] = region["region_name"].map(normalize_region_name)
    region["region_macro"] = region["region_name_normalized"].map(map_region_macro)
    region["region_climate"] = region["region_name_normalized"].map(map_region_climate)

    grouped = region.groupby("contract_number", sort=False)
    frame["region_macro_mode"] = grouped["region_macro"].agg(mode_or_missing)
    frame["region_macro_nunique"] = grouped["region_macro"].nunique(dropna=True).astype("int32")
    frame["region_climate_mode"] = grouped["region_climate"].agg(mode_or_missing)
    frame["region_climate_nunique"] = grouped["region_climate"].nunique(dropna=True).astype("int32")
    frame["is_multi_region_macro"] = (frame["region_macro_nunique"] > 1).astype("int8")
    frame["is_multi_region_climate"] = (frame["region_climate_nunique"] > 1).astype("int8")

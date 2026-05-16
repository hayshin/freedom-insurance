from __future__ import annotations

import pandas as pd
import polars as pl


BONUS_MALUS_COEF_MAP = {
    "M": 2.45,
    "0": 2.30,
    "1": 1.55,
    "2": 1.40,
    "3": 1.00,
    "4": 0.95,
    "5": 0.90,
    "6": 0.85,
    "7": 0.80,
    "8": 0.75,
    "9": 0.70,
    "10": 0.65,
    "11": 0.60,
    "12": 0.55,
    "13": 0.50,
}

NOVICE_AGE_EXPERIENCE_IDS = [2, 4]
YOUNG_AGE_EXPERIENCE_IDS = [2, 3]
YOUNG_NOVICE_AGE_EXPERIENCE_IDS = [2]


def _to_indexed_pandas(features: pl.DataFrame, index: pd.Index) -> pd.DataFrame:
    if features.is_empty():
        return pd.DataFrame(index=index)
    return pd.DataFrame(features.to_dict(as_series=False)).set_index("contract_number").reindex(index)


def _bonus_malus_class_expr() -> pl.Expr:
    normalized = pl.col("bonus_malus").cast(pl.String, strict=False).str.strip_chars().str.to_uppercase()
    return pl.when(normalized.is_null() | (normalized == "")).then(None).otherwise(normalized)


def _bonus_malus_coef_expr(bonus_malus_class: pl.Expr) -> pl.Expr:
    mapped = pl.when(bonus_malus_class == "M").then(pl.lit(BONUS_MALUS_COEF_MAP["M"]))
    for key, value in BONUS_MALUS_COEF_MAP.items():
        if key == "M":
            continue
        mapped = mapped.when(bonus_malus_class == key).then(pl.lit(value))
    return mapped.otherwise(None)


def _valid_experience_expr() -> pl.Expr:
    experience = pl.col("experience_year").cast(pl.Float64, strict=False)
    return pl.when(experience.is_between(0, 80)).then(experience).otherwise(None)


def _age_experience_id_expr() -> pl.Expr:
    return pl.col("age_experience_id").cast(pl.Int64, strict=False)


def _flag_from_age_experience_or_experience(
    age_experience_id: pl.Expr,
    matching_ids: list[int],
    fallback: pl.Expr | None = None,
) -> pl.Expr:
    known_ids = [2, 3, 4, 5]
    flag = (
        pl.when(age_experience_id.is_in(matching_ids))
        .then(pl.lit(1.0))
        .when(age_experience_id.is_in(known_ids))
        .then(pl.lit(0.0))
    )
    if fallback is not None:
        flag = flag.when(fallback.is_not_null()).then(fallback.cast(pl.Float64))
    return flag.otherwise(None)


def add_driver_features(raw: pl.DataFrame, frame: pd.DataFrame) -> None:
    if "contract_number" not in raw.columns:
        return

    driver_columns = {"bonus_malus", "experience_year", "age_experience_id"}
    if not driver_columns.intersection(raw.columns):
        return

    bonus_malus_class = _bonus_malus_class_expr() if "bonus_malus" in raw.columns else pl.lit(None, dtype=pl.String)
    bonus_malus_coef = _bonus_malus_coef_expr(bonus_malus_class)
    bonus_malus_missing = bonus_malus_class.is_null().cast(pl.Float64)
    bonus_malus_unknown = (bonus_malus_class.is_not_null() & bonus_malus_coef.is_null()).cast(pl.Float64)

    experience_year = _valid_experience_expr() if "experience_year" in raw.columns else pl.lit(None, dtype=pl.Float64)
    age_experience_id = (
        _age_experience_id_expr() if "age_experience_id" in raw.columns else pl.lit(None, dtype=pl.Int64)
    )
    novice = _flag_from_age_experience_or_experience(
        age_experience_id,
        NOVICE_AGE_EXPERIENCE_IDS,
        fallback=experience_year < 2,
    )
    young = _flag_from_age_experience_or_experience(age_experience_id, YOUNG_AGE_EXPERIENCE_IDS)
    young_novice = _flag_from_age_experience_or_experience(age_experience_id, YOUNG_NOVICE_AGE_EXPERIENCE_IDS)

    rows = raw.select(
        [
            "contract_number",
            bonus_malus_coef.alias("_bonus_malus_coef"),
            bonus_malus_missing.alias("_bonus_malus_missing"),
            bonus_malus_unknown.alias("_bonus_malus_unknown"),
            experience_year.alias("_driver_experience_year"),
            age_experience_id.alias("_age_experience_id"),
            novice.alias("_is_novice_driver"),
            young.alias("_is_young_driver"),
            young_novice.alias("_is_young_novice_driver"),
        ]
    )
    features = rows.group_by("contract_number", maintain_order=True).agg(
        [
            pl.col("_bonus_malus_coef").min().alias("bonus_malus_coef_min"),
            pl.col("_bonus_malus_coef").max().alias("bonus_malus_coef_max"),
            pl.col("_bonus_malus_coef").mean().alias("bonus_malus_coef_mean"),
            pl.col("_bonus_malus_coef").std().alias("bonus_malus_coef_std"),
            pl.col("_bonus_malus_coef").drop_nulls().n_unique().cast(pl.Int32).alias("bonus_malus_coef_nunique"),
            (pl.col("_bonus_malus_coef") > 1.0).max().fill_null(False).cast(pl.Int8).alias("has_malus_class"),
            (pl.col("_bonus_malus_coef") <= 0.65).max().fill_null(False).cast(pl.Int8).alias(
                "has_super_bonus_class"
            ),
            pl.col("_bonus_malus_missing").mean().fill_null(0.0).alias("bonus_malus_missing_share"),
            pl.col("_bonus_malus_unknown").mean().fill_null(0.0).alias("bonus_malus_unknown_share"),
            pl.col("_driver_experience_year").min().alias("driver_experience_year_min"),
            pl.col("_driver_experience_year").max().alias("driver_experience_year_max"),
            pl.col("_driver_experience_year").mean().alias("driver_experience_year_mean"),
            pl.col("_driver_experience_year").std().alias("driver_experience_year_std"),
            pl.col("_is_novice_driver").max().fill_null(0).cast(pl.Int8).alias("has_novice_driver"),
            pl.col("_is_novice_driver").min().fill_null(0).cast(pl.Int8).alias("all_drivers_novice"),
            pl.col("_is_novice_driver").mean().fill_null(0.0).alias("novice_driver_share"),
            pl.col("_is_young_driver").max().fill_null(0).cast(pl.Int8).alias("has_young_driver"),
            pl.col("_is_young_driver").mean().fill_null(0.0).alias("young_driver_share"),
            pl.col("_is_young_novice_driver").max().fill_null(0).cast(pl.Int8).alias(
                "has_young_novice_driver"
            ),
            pl.col("_age_experience_id").drop_nulls().n_unique().cast(pl.Int32).alias("age_experience_id_nunique"),
        ]
    ).with_columns(
        (pl.col("age_experience_id_nunique") > 1).cast(pl.Int8).alias("is_multi_age_experience_group")
    )

    indexed = _to_indexed_pandas(features, frame.index)
    for col in indexed.columns:
        frame[col] = indexed[col]

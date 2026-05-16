from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMNS = {"claim_amount", "claim_cnt", "is_claim"}
RAW_ID_COLUMNS = {"unique_id", "contract_number", "insurer_iin", "driver_iin", "car_number"}
ID_STAT_HELPER_COLUMNS = {
    "insurer_iin_mode",
    "driver_iin_mode",
    "car_number_mode",
}
FINANCIAL_METRIC_ONLY_COLUMNS = {
    "premium_wo_term",
    "premium_wo_term_log1p",
    "premium_wo_term_is_zero",
    "premium_wo_term_is_missing",
    "premium_wo_term_ratio",
    "premium_return_ratio",
}
DATE_COLUMNS = {"operation_date"}
PREPROCESSED_SOURCE_COLUMNS = {
    "mark",
    "model",
    "car_age",
    "car_year",
    "bonus_malus",
    "age_experience_id",
    "age_experience_name",
    "experience_year",
    "engine_volume",
    "engine_power",
}
LEAKAGE_COLUMNS = TARGET_COLUMNS | RAW_ID_COLUMNS | ID_STAT_HELPER_COLUMNS | FINANCIAL_METRIC_ONLY_COLUMNS
HIGH_CARDINALITY_COLUMNS = {
    "mark_clean_mode",
    "model_clean_mode",
    "mark_model_pair",
    "ownerkato",
    "ownerkato_short",
}
OUTLIER_CAP_COLUMNS = {
    "engine_volume",
    "engine_power",
}
TARGET_LOSS_RATIO = 0.70
N_PIPELINE_STAGES = 12


@dataclass
class PricingCalibration:
    scale: float
    threshold: float
    floor_ratio: float
    validation_loss_ratio: float
    group_keep_or_decrease_loss_ratio: float | None
    group_increase_loss_ratio: float | None
    keep_or_decrease_share: float
    group_loss_ratio_gap: float | None


@dataclass
class SeverityCalibration:
    intercept: float
    slope: float
    positive_count: int
    raw_rmse_positive: float
    raw_r2_positive: float

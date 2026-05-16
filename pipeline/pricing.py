from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pipeline.config import TARGET_LOSS_RATIO, PricingCalibration


def portfolio_loss_ratio(claim_amount: pd.Series | np.ndarray, premium: pd.Series | np.ndarray) -> float:
    premium_sum = float(np.sum(premium))
    if premium_sum <= 0:
        return math.nan
    return float(np.sum(claim_amount) / premium_sum)


def calibrate_pricing(valid: pd.DataFrame, expected_claim: np.ndarray) -> tuple[np.ndarray, PricingCalibration]:
    base_premium = valid["premium"].astype(float).to_numpy()
    actual_claim = valid["claim_amount"].astype(float).to_numpy()
    risk_ratio = expected_claim / np.maximum(base_premium, 1.0)

    best: tuple[float, float, float, float, float, float | None, float | None, float, float | None] | None = None
    scales = np.linspace(0.05, 4.00, 140)
    floor_ratios = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0])
    thresholds = np.unique(np.quantile(risk_ratio, np.r_[np.linspace(0.0, 0.97, 61), 0.98, 0.99, 0.995]))

    for scale in scales:
        raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * scale, 0.0, base_premium * 3.0)
        for floor_ratio in floor_ratios:
            minimum_allowed = base_premium * floor_ratio
            increase_candidate = np.maximum(raw_new_premium, minimum_allowed)
            keep_candidate = np.maximum(np.minimum(raw_new_premium, base_premium), minimum_allowed)
            for threshold in thresholds:
                proposed = np.where(risk_ratio >= threshold, increase_candidate, keep_candidate)
                proposed = np.clip(proposed, 0.0, base_premium * 3.0)
                lr_total = portfolio_loss_ratio(actual_claim, proposed)
                increased = proposed > base_premium
                keep_share = float((~increased).mean())
                lr_keep = portfolio_loss_ratio(actual_claim[~increased], proposed[~increased]) if (~increased).any() else None
                lr_inc = portfolio_loss_ratio(actual_claim[increased], proposed[increased]) if increased.any() else None

                group_gap = None
                if lr_keep is not None and lr_inc is not None:
                    group_gap = abs(lr_keep - TARGET_LOSS_RATIO) + abs(lr_inc - TARGET_LOSS_RATIO)
                    group_penalty = group_gap
                else:
                    group_penalty = 2.0

                total_penalty = abs(lr_total - TARGET_LOSS_RATIO)
                outside_target_band = max(total_penalty - 0.02, 0.0)
                objective = 2.0 * total_penalty + 1.0 * group_penalty + 5.0 * outside_target_band - 0.06 * keep_share

                if best is None or objective < best[0]:
                    best = (objective, scale, threshold, floor_ratio, lr_total, lr_keep, lr_inc, keep_share, group_gap)

    if best is None:
        raise RuntimeError("Could not calibrate pricing.")

    _, scale, threshold, floor_ratio, lr_total, lr_keep, lr_inc, keep_share, group_gap = best
    raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * scale, 0.0, base_premium * 3.0)
    minimum_allowed = base_premium * floor_ratio
    increase_candidate = np.maximum(raw_new_premium, minimum_allowed)
    keep_candidate = np.maximum(np.minimum(raw_new_premium, base_premium), minimum_allowed)
    calibrated = np.where(risk_ratio >= threshold, increase_candidate, keep_candidate)
    calibrated = np.clip(calibrated, 0.0, base_premium * 3.0)

    calibration = PricingCalibration(
        scale=float(scale),
        threshold=float(threshold),
        floor_ratio=float(floor_ratio),
        validation_loss_ratio=float(lr_total),
        group_keep_or_decrease_loss_ratio=None if lr_keep is None else float(lr_keep),
        group_increase_loss_ratio=None if lr_inc is None else float(lr_inc),
        keep_or_decrease_share=float(keep_share),
        group_loss_ratio_gap=None if group_gap is None else float(group_gap),
    )
    return calibrated, calibration


def apply_pricing(frame: pd.DataFrame, expected_claim: np.ndarray, calibration: PricingCalibration) -> np.ndarray:
    base_premium = frame["premium"].astype(float).to_numpy()
    risk_ratio = expected_claim / np.maximum(base_premium, 1.0)
    raw_new_premium = np.clip((expected_claim / TARGET_LOSS_RATIO) * calibration.scale, 0.0, base_premium * 3.0)
    minimum_allowed = base_premium * calibration.floor_ratio
    increase_candidate = np.maximum(raw_new_premium, minimum_allowed)
    keep_candidate = np.maximum(np.minimum(raw_new_premium, base_premium), minimum_allowed)
    proposed = np.where(
        risk_ratio >= calibration.threshold,
        increase_candidate,
        keep_candidate,
    )
    return np.clip(proposed, 0.0, base_premium * 3.0)

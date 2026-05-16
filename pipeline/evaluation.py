from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

from pipeline.config import TARGET_LOSS_RATIO
from pipeline.pricing import portfolio_loss_ratio


def best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def evaluate(
    valid: pd.DataFrame,
    claim_probability: np.ndarray,
    severity_pred: np.ndarray,
    expected_claim: np.ndarray,
    new_premium: np.ndarray,
    f1_threshold: float,
) -> dict:
    actual_claim = valid["claim_amount"].astype(float)
    is_claim = valid["is_claim"].astype(int)
    y_pred = claim_probability >= f1_threshold
    positive = actual_claim > 0
    base_premium = valid["premium"].astype(float).to_numpy()
    increased = new_premium > base_premium
    decreased = new_premium < base_premium
    kept = new_premium == base_premium
    keep_or_decrease = ~increased
    keep_or_decrease_count = int(keep_or_decrease.sum())
    increase_count = int(increased.sum())
    decrease_count = int(decreased.sum())
    keep_count = int(kept.sum())
    keep_lr = (
        portfolio_loss_ratio(
            actual_claim[keep_or_decrease],
            new_premium[keep_or_decrease],
        )
        if keep_or_decrease.any()
        else None
    )
    increase_lr = (
        portfolio_loss_ratio(
            actual_claim[increased],
            new_premium[increased],
        )
        if increased.any()
        else None
    )
    group_loss_ratio_gap = (
        abs(keep_lr - TARGET_LOSS_RATIO) + abs(increase_lr - TARGET_LOSS_RATIO)
        if keep_lr is not None and increase_lr is not None
        else None
    )

    metrics = {
        "frequency": {
            "roc_auc": float(roc_auc_score(is_claim, claim_probability)),
            "gini": float(2 * roc_auc_score(is_claim, claim_probability) - 1),
            "pr_auc": float(average_precision_score(is_claim, claim_probability)),
            "f1_threshold": float(f1_threshold),
            "f1": float(f1_score(is_claim, y_pred)),
            "claim_rate": float(is_claim.mean()),
        },
        "severity": {
            "n_positive_valid": int(positive.sum()),
            "mae_positive": None,
            "rmse_positive": None,
            "r2_positive": None,
        },
        "business": {
            "baseline_loss_ratio_premium": portfolio_loss_ratio(actual_claim, valid["premium"]),
            "baseline_loss_ratio_premium_wo_term": portfolio_loss_ratio(actual_claim, valid["premium_wo_term"]),
            "post_pricing_loss_ratio": portfolio_loss_ratio(actual_claim, new_premium),
            "mean_expected_claim": float(np.mean(expected_claim)),
            "mean_new_premium": float(np.mean(new_premium)),
            "increase_share": float(increased.mean()),
            "keep_share": float(kept.mean()),
            "decrease_share": float(decreased.mean()),
            "keep_or_decrease_share": float(keep_or_decrease.mean()),
            "keep_or_decrease_count": keep_or_decrease_count,
            "increase_count": increase_count,
            "keep_count": keep_count,
            "decrease_count": decrease_count,
            "keep_or_decrease_loss_ratio": keep_lr,
            "increase_loss_ratio": increase_lr,
            "group_loss_ratio_gap": group_loss_ratio_gap,
            "min_new_to_old_premium_ratio": float(np.min(new_premium / np.maximum(base_premium, 1.0))),
            "max_new_to_old_premium_ratio": float(np.max(new_premium / np.maximum(base_premium, 1.0))),
            "mean_uplift_for_increased": (
                float(np.mean((new_premium[increased] / np.maximum(base_premium[increased], 1.0)) - 1.0))
                if increased.any()
                else None
            ),
        },
    }
    if positive.any():
        metrics["severity"]["mae_positive"] = float(mean_absolute_error(actual_claim[positive], severity_pred[positive]))
        metrics["severity"]["rmse_positive"] = float(
            mean_squared_error(actual_claim[positive], severity_pred[positive]) ** 0.5
        )
        metrics["severity"]["r2_positive"] = float(r2_score(actual_claim[positive], severity_pred[positive]))
    return metrics

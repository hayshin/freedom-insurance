from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pipeline.config import SeverityCalibration
from pipeline.features import is_categorical_feature
from pipeline.logging import log_progress

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None
    HAS_CATBOOST = False


@dataclass
class SeverityModel:
    estimator: object
    backend: str
    target: str
    calibration: SeverityCalibration | None = None


def _positive_regression_metrics(actual: pd.Series | np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    actual_array = np.asarray(actual, dtype=float)
    pred_array = np.asarray(pred, dtype=float)
    residuals = pred_array - actual_array
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.mean(residuals**2) ** 0.5)
    variance = float(np.var(actual_array))
    r2 = float(1.0 - (np.mean(residuals**2) / variance)) if variance > 0 else float("-inf")
    return mae, rmse, r2


def fit_severity_calibration(pred: np.ndarray, actual_amount: pd.Series | np.ndarray) -> SeverityCalibration | None:
    actual = np.asarray(actual_amount, dtype=float)
    pred = np.asarray(pred, dtype=float)
    positive = actual > 0
    if int(positive.sum()) < 20:
        return None

    actual_positive = actual[positive]
    pred_positive = pred[positive]
    _, raw_rmse, raw_r2 = _positive_regression_metrics(actual_positive, pred_positive)

    pred_variance = float(np.var(pred_positive))
    if pred_variance <= 0:
        slope = 0.0
    else:
        covariance = float(np.mean((pred_positive - pred_positive.mean()) * (actual_positive - actual_positive.mean())))
        slope = covariance / pred_variance
    slope = float(np.clip(slope, 0.0, 2.0))
    intercept = float(actual_positive.mean() - slope * pred_positive.mean())
    return SeverityCalibration(
        intercept=intercept,
        slope=slope,
        positive_count=int(positive.sum()),
        raw_rmse_positive=raw_rmse,
        raw_r2_positive=raw_r2,
    )


def apply_severity_calibration(pred: np.ndarray, calibration: SeverityCalibration | None) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if calibration is None:
        return pred.clip(min=0)
    return (calibration.intercept + calibration.slope * pred).clip(min=0)


def make_lgbm_progress_callback(label: str, total_iterations: int, period: int, enabled: bool):
    period = max(1, period)

    def _callback(env) -> None:
        iteration = env.iteration + 1
        if iteration == 1 or iteration % period == 0 or iteration >= total_iterations:
            pct = 100.0 * iteration / max(total_iterations, 1)
            log_progress(f"{label}: iteration {iteration}/{total_iterations} ({pct:.1f}%)", enabled=enabled)

    _callback.order = 10
    return _callback


def make_lgbm_classifier(random_state: int):
    return LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=48,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_lgbm_regressor(random_state: int):
    return LGBMRegressor(
        objective="regression",
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_catboost_classifier(random_state: int, progress_period: int, show_progress: bool):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1400,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=10.0,
        random_seed=random_state,
        auto_class_weights="SqrtBalanced",
        random_strength=0.5,
        od_type="Iter",
        od_wait=80,
        use_best_model=True,
        allow_writing_files=False,
        thread_count=-1,
        verbose=max(1, progress_period) if show_progress else False,
    )


def make_catboost_regressor(
    random_state: int,
    progress_period: int,
    show_progress: bool,
    *,
    params: dict | None = None,
):
    base_params = {
        "loss_function": "RMSE",
        "iterations": 700,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 6.0,
        "random_seed": random_state,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": max(1, progress_period) if show_progress else False,
    }
    if params:
        base_params.update(params)
    return CatBoostRegressor(**base_params)


def _sample_catboost_severity_params(rng: np.random.Generator) -> dict:
    bootstrap_type = rng.choice(["Bayesian", "Bernoulli"])
    params = {
        "iterations": int(rng.integers(400, 1201)),
        "learning_rate": float(rng.uniform(0.01, 0.08)),
        "depth": int(rng.integers(5, 9)),
        "l2_leaf_reg": float(rng.uniform(2.0, 10.0)),
        "min_data_in_leaf": int(rng.integers(10, 81)),
        "random_strength": float(rng.uniform(0.0, 1.2)),
        "bootstrap_type": bootstrap_type,
    }
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = float(rng.uniform(0.0, 1.2))
    else:
        params["subsample"] = float(rng.uniform(0.6, 0.95))
    return params


def make_severity_target(train_x: pd.DataFrame, amount: pd.Series, target: str) -> pd.Series:
    amount = amount.astype(float)
    if target == "claim_per_premium":
        premium = train_x["premium"].astype(float).clip(lower=1.0)
        return np.log1p(amount / premium)
    if target == "amount":
        return np.log1p(amount)
    raise ValueError(f"Unsupported severity target: {target}")


def invert_severity_prediction(frame: pd.DataFrame, raw_pred: np.ndarray, target: str) -> np.ndarray:
    pred = np.expm1(raw_pred).clip(min=0)
    if target == "claim_per_premium":
        premium = frame["premium"].astype(float).clip(lower=1.0).to_numpy()
        return pred * premium
    if target == "amount":
        return pred
    raise ValueError(f"Unsupported severity target: {target}")


def tune_catboost_severity(
    train_x: pd.DataFrame,
    train_amount: pd.Series,
    valid_x: pd.DataFrame,
    valid_amount: pd.Series,
    categorical: list[str],
    random_state: int,
    trials: int,
    max_seconds: int,
    progress_period: int,
    show_progress: bool,
    r2_weight: float,
    objective: str,
    severity_target: str,
    calibrate_severity: bool,
):
    if not HAS_CATBOOST:
        raise ImportError("CatBoost tuning requested, but catboost is not installed.")

    trials = max(1, int(trials))
    max_seconds = max(1, int(max_seconds))
    rng = np.random.default_rng(random_state)

    train_prepared = prepare_catboost_frame(train_x, categorical)
    valid_prepared = prepare_catboost_frame(valid_x, categorical)
    train_y = make_severity_target(train_x, train_amount, severity_target)
    valid_amount = valid_amount.astype(float)
    valid_positive = valid_amount > 0
    if valid_positive.any():
        eval_x = valid_prepared.loc[valid_positive]
        eval_y = make_severity_target(valid_x.loc[valid_positive], valid_amount[valid_positive], severity_target)
    else:
        eval_x = valid_prepared
        eval_y = make_severity_target(valid_x, valid_amount.clip(lower=0), severity_target)

    best_score = float("inf")
    best_rmse = float("inf")
    best_mae = float("inf")
    best_r2 = float("-inf")
    best_params: dict | None = None
    best_model = None
    best_pred = None
    best_calibration = None
    best_raw_rmse = float("inf")
    best_raw_r2 = float("-inf")
    started_at = time.time()
    last_log = started_at
    trial = 0

    for trial in range(1, trials + 1):
        if time.time() - started_at >= max_seconds:
            break

        params = _sample_catboost_severity_params(rng)
        params["od_type"] = "Iter"
        params["od_wait"] = 50
        params["use_best_model"] = True
        params["eval_metric"] = "RMSE"
        params["verbose"] = False

        model = make_catboost_regressor(
            random_state,
            progress_period,
            show_progress,
            params=params,
        )
        model.fit(
            train_prepared,
            train_y,
            cat_features=categorical,
            eval_set=(eval_x, eval_y),
        )
        raw_pred_valid = invert_severity_prediction(valid_x, model.predict(valid_prepared), severity_target)
        calibration = fit_severity_calibration(raw_pred_valid, valid_amount) if calibrate_severity else None
        pred_valid = apply_severity_calibration(raw_pred_valid, calibration)
        if valid_positive.any():
            actual = valid_amount[valid_positive]
            raw_mae, raw_rmse, raw_r2 = _positive_regression_metrics(actual, raw_pred_valid[valid_positive])
            mae, rmse, r2 = _positive_regression_metrics(actual, pred_valid[valid_positive])
        else:
            raw_rmse = float("inf")
            raw_r2 = float("-inf")
            mae = float("inf")
            rmse = float("inf")
            r2 = float("-inf")

        if objective == "rmse_r2":
            score = rmse + r2_weight * (1.0 - r2)
        else:
            score = rmse

        if score < best_score or (score == best_score and mae < best_mae):
            best_score = score
            best_rmse = rmse
            best_mae = mae
            best_r2 = r2
            best_params = params
            best_model = SeverityModel(model, "catboost", severity_target, calibration)
            best_pred = pred_valid
            best_calibration = calibration
            best_raw_rmse = raw_rmse
            best_raw_r2 = raw_r2

        if show_progress and (time.time() - last_log >= max(1, progress_period)):
            log_progress(
                (
                    f"Severity tuning: trial {trial}/{trials}, "
                    f"best_score={best_score:.2f}, best_r2={best_r2:.4f}"
                ),
                enabled=show_progress,
            )
            last_log = time.time()

    if best_model is None or best_pred is None:
        raise RuntimeError("Severity tuning failed to produce a model.")

    tuning_info = {
        "trials_requested": int(trials),
        "trials_completed": int(trial),
        "time_seconds": float(time.time() - started_at),
        "objective": objective,
        "r2_weight": float(r2_weight),
        "best_score": float(best_score),
        "best_rmse_positive": float(best_rmse),
        "best_mae_positive": float(best_mae),
        "best_r2_positive": float(best_r2),
        "best_raw_rmse_positive": float(best_raw_rmse),
        "best_raw_r2_positive": float(best_raw_r2),
        "severity_target": severity_target,
        "severity_calibration": None if best_calibration is None else asdict(best_calibration),
        "best_params": best_params,
    }
    return best_model, best_pred, tuning_info


def make_sklearn_model(model_type: str, numeric: list[str], categorical: list[str], random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    if model_type == "classifier":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def prepare_lgbm_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for col in categorical:
        prepared[col] = prepared[col].fillna("__MISSING__").astype("category")
    missed_categorical = [
        col for col in prepared.columns if col not in categorical and is_categorical_feature(prepared[col], col)
    ]
    if missed_categorical:
        raise ValueError(
            "LightGBM input contains string/category columns not declared as categorical: "
            f"{missed_categorical}"
        )
    return prepared


def prepare_catboost_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for col in categorical:
        prepared[col] = prepared[col].fillna("__MISSING__").astype(str)
    missed_categorical = [
        col for col in prepared.columns if col not in categorical and is_categorical_feature(prepared[col], col)
    ]
    if missed_categorical:
        raise ValueError(
            "CatBoost input contains string/category columns not declared as categorical: "
            f"{missed_categorical}"
        )
    return prepared


def resolve_model_backend(requested_backend: str, force_sklearn: bool) -> str:
    if force_sklearn:
        return "sklearn"
    if requested_backend == "auto":
        if HAS_CATBOOST:
            return "catboost"
        if HAS_LIGHTGBM:
            return "lightgbm"
        return "sklearn"
    if requested_backend == "catboost" and not HAS_CATBOOST:
        raise ImportError("CatBoost backend requested, but catboost is not installed.")
    if requested_backend == "lightgbm" and not HAS_LIGHTGBM:
        raise ImportError("LightGBM backend requested, but lightgbm is not installed.")
    return requested_backend


def train_frequency_model(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    categorical: list[str],
    numeric: list[str],
    random_state: int,
    backend: str,
    progress_period: int,
    show_progress: bool,
):
    if backend == "catboost":
        model = make_catboost_classifier(random_state, progress_period, show_progress)
        model.fit(
            prepare_catboost_frame(train_x, categorical),
            train_y,
            cat_features=categorical,
            eval_set=(prepare_catboost_frame(valid_x, categorical), valid_y),
        )
        valid_pred = model.predict_proba(prepare_catboost_frame(valid_x, categorical))[:, 1]
        return model, valid_pred, "catboost"

    if backend == "lightgbm":
        model = make_lgbm_classifier(random_state)
        model.fit(
            prepare_lgbm_frame(train_x, categorical),
            train_y,
            categorical_feature=categorical,
            callbacks=[
                make_lgbm_progress_callback(
                    "Frequency model",
                    total_iterations=model.n_estimators,
                    period=progress_period,
                    enabled=show_progress,
                )
            ],
        )
        valid_pred = model.predict_proba(prepare_lgbm_frame(valid_x, categorical))[:, 1]
        return model, valid_pred, "lightgbm"

    model = make_sklearn_model("classifier", numeric, categorical, random_state)
    model.fit(train_x, train_y)
    valid_pred = model.predict_proba(valid_x)[:, 1]
    return model, valid_pred, "sklearn_hist_gradient_boosting"


def train_severity_model(
    train_x: pd.DataFrame,
    train_amount: pd.Series,
    valid_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    random_state: int,
    backend: str,
    progress_period: int,
    show_progress: bool,
    severity_target: str = "claim_per_premium",
    calibrate_severity: bool = True,
    valid_amount: pd.Series | None = None,
):
    y = make_severity_target(train_x, train_amount, severity_target)
    if backend == "catboost":
        model = make_catboost_regressor(random_state, progress_period, show_progress)
        model.fit(
            prepare_catboost_frame(train_x, categorical),
            y,
            cat_features=categorical,
        )
        raw_valid_pred = invert_severity_prediction(
            valid_x,
            model.predict(prepare_catboost_frame(valid_x, categorical)),
            severity_target,
        )
        calibration = (
            fit_severity_calibration(raw_valid_pred, valid_amount)
            if calibrate_severity and valid_amount is not None
            else None
        )
        valid_pred = apply_severity_calibration(raw_valid_pred, calibration)
        return SeverityModel(model, "catboost", severity_target, calibration), valid_pred, "catboost"

    if backend == "lightgbm":
        model = make_lgbm_regressor(random_state)
        model.fit(
            prepare_lgbm_frame(train_x, categorical),
            y,
            categorical_feature=categorical,
            callbacks=[
                make_lgbm_progress_callback(
                    "Severity model",
                    total_iterations=model.n_estimators,
                    period=progress_period,
                    enabled=show_progress,
                )
            ],
        )
        raw_valid_pred = invert_severity_prediction(
            valid_x,
            model.predict(prepare_lgbm_frame(valid_x, categorical)),
            severity_target,
        )
        calibration = (
            fit_severity_calibration(raw_valid_pred, valid_amount)
            if calibrate_severity and valid_amount is not None
            else None
        )
        valid_pred = apply_severity_calibration(raw_valid_pred, calibration)
        return SeverityModel(model, "lightgbm", severity_target, calibration), valid_pred, "lightgbm"

    model = make_sklearn_model("regressor", numeric, categorical, random_state)
    model.fit(train_x, y)
    raw_valid_pred = invert_severity_prediction(valid_x, model.predict(valid_x), severity_target)
    calibration = (
        fit_severity_calibration(raw_valid_pred, valid_amount)
        if calibrate_severity and valid_amount is not None
        else None
    )
    valid_pred = apply_severity_calibration(raw_valid_pred, calibration)
    return SeverityModel(model, "sklearn_hist_gradient_boosting", severity_target, calibration), valid_pred, "sklearn_hist_gradient_boosting"


def predict_frequency(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if backend == "catboost":
        return model.predict_proba(prepare_catboost_frame(x, categorical))[:, 1]
    if backend == "lightgbm":
        return model.predict_proba(prepare_lgbm_frame(x, categorical))[:, 1]
    return model.predict_proba(x)[:, 1]


def predict_severity(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if isinstance(model, SeverityModel):
        raw_backend = model.backend
        estimator = model.estimator
        target = model.target
        calibration = model.calibration
    else:
        raw_backend = backend
        estimator = model
        target = "amount"
        calibration = None

    if raw_backend == "catboost":
        raw_pred = estimator.predict(prepare_catboost_frame(x, categorical))
    elif raw_backend == "lightgbm":
        raw_pred = estimator.predict(prepare_lgbm_frame(x, categorical))
    else:
        raw_pred = estimator.predict(x)
    amount_pred = invert_severity_prediction(x, raw_pred, target)
    return apply_severity_calibration(amount_pred, calibration)

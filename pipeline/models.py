from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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
        iterations=900,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=random_state,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        thread_count=-1,
        verbose=max(1, progress_period) if show_progress else False,
    )


def make_catboost_regressor(random_state: int, progress_period: int, show_progress: bool):
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=700,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=random_state,
        allow_writing_files=False,
        thread_count=-1,
        verbose=max(1, progress_period) if show_progress else False,
    )


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
):
    y = np.log1p(train_amount)
    if backend == "catboost":
        model = make_catboost_regressor(random_state, progress_period, show_progress)
        model.fit(
            prepare_catboost_frame(train_x, categorical),
            y,
            cat_features=categorical,
        )
        valid_pred = np.expm1(model.predict(prepare_catboost_frame(valid_x, categorical))).clip(min=0)
        return model, valid_pred, "catboost"

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
        valid_pred = np.expm1(model.predict(prepare_lgbm_frame(valid_x, categorical))).clip(min=0)
        return model, valid_pred, "lightgbm"

    model = make_sklearn_model("regressor", numeric, categorical, random_state)
    model.fit(train_x, y)
    valid_pred = np.expm1(model.predict(valid_x)).clip(min=0)
    return model, valid_pred, "sklearn_hist_gradient_boosting"


def predict_frequency(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if backend == "catboost":
        return model.predict_proba(prepare_catboost_frame(x, categorical))[:, 1]
    if backend == "lightgbm":
        return model.predict_proba(prepare_lgbm_frame(x, categorical))[:, 1]
    return model.predict_proba(x)[:, 1]


def predict_severity(model, x: pd.DataFrame, categorical: list[str], backend: str) -> np.ndarray:
    if backend == "catboost":
        pred = model.predict(prepare_catboost_frame(x, categorical))
    elif backend == "lightgbm":
        pred = model.predict(prepare_lgbm_frame(x, categorical))
    else:
        pred = model.predict(x)
    return np.expm1(pred).clip(min=0)

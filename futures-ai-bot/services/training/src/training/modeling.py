"""Baseline model training utilities and ensemble scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

BASE_FEATURE_COLUMNS = [
    "ret_1",
    "log_ret_1",
    "rolling_vol",
    "atr",
    "dist_from_vwap",
    "candle_body_ratio",
    "close_location",
    "breakout_pressure",
    "breakdown_pressure",
    "compression_expansion",
    "mean_reversion_z",
    "trend_slope_proxy",
    "trend_strength",
    "minute_of_day",
    "day_of_week",
    "range_pct",
    "volume_z",
    "rel_es_vs_nq",
    "rel_cl_vs_gc",
    "spread_ret_es_nq",
    "spread_ret_cl_gc",
]

REGIME_FEATURE_COLUMNS = ["rolling_vol", "atr", "compression_expansion", "range_pct", "volume_z"]


@dataclass(slots=True)
class TrainingOutput:
    """Container for trained models and evaluation metrics."""

    model: CalibratedClassifierCV
    regime_model: KMeans
    feature_columns: list[str]
    regime_feature_columns: list[str]
    metrics: dict[str, float]


def train_baseline_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    random_state: int = 7,
) -> TrainingOutput:
    """Train a calibrated probability classifier and a regime clustering model."""

    train_clean = _clean_for_model(train_df, [*BASE_FEATURE_COLUMNS, "next_n_bar_direction"])
    val_clean = _clean_for_model(validation_df, [*BASE_FEATURE_COLUMNS, "next_n_bar_direction"])

    x_train = train_clean[BASE_FEATURE_COLUMNS].astype(float).to_numpy()
    y_train = train_clean["next_n_bar_direction"].astype(int).to_numpy()
    x_val = val_clean[BASE_FEATURE_COLUMNS].astype(float).to_numpy()
    y_val = val_clean["next_n_bar_direction"].astype(int).to_numpy()

    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=200,
        random_state=random_state,
        min_samples_leaf=30,
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(x_train, y_train)

    regime_train = train_clean[REGIME_FEATURE_COLUMNS].astype(float).to_numpy()
    regime_model = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    regime_model.fit(regime_train)

    prob = model.predict_proba(x_val)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "brier": float(brier_score_loss(y_val, prob)),
    }
    if len(np.unique(y_val)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_val, prob))
    else:
        metrics["roc_auc"] = 0.5

    return TrainingOutput(
        model=model,
        regime_model=regime_model,
        feature_columns=BASE_FEATURE_COLUMNS,
        regime_feature_columns=REGIME_FEATURE_COLUMNS,
        metrics=metrics,
    )


def predict_probability(model: CalibratedClassifierCV, frame: pd.DataFrame) -> np.ndarray:
    """Return calibrated long-direction probabilities."""

    clean = frame[BASE_FEATURE_COLUMNS].astype(float).fillna(0.0)
    return np.asarray(model.predict_proba(clean.to_numpy())[:, 1], dtype=float)


def predict_regime(model: KMeans, frame: pd.DataFrame) -> np.ndarray:
    """Assign volatility/trend regime cluster id."""

    clean = frame[REGIME_FEATURE_COLUMNS].astype(float).fillna(0.0)
    return np.asarray(model.predict(clean.to_numpy()), dtype=int)


def ensemble_signal_score(
    frame: pd.DataFrame, probability: np.ndarray, regime: np.ndarray
) -> np.ndarray:
    """Combine directional models with handcrafted short-horizon factors."""

    trend = np.tanh(frame["trend_slope_proxy"].fillna(0.0).to_numpy())
    breakout = np.tanh(
        (
            frame["breakout_pressure"].fillna(0.0) - frame["breakdown_pressure"].fillna(0.0)
        ).to_numpy()
    )
    mean_rev = -np.tanh(frame["mean_reversion_z"].fillna(0.0).to_numpy())
    vol = np.tanh(frame["rolling_vol"].fillna(0.0).to_numpy() * 10.0)

    regime_bias_map = {0: -0.15, 1: 0.0, 2: 0.1, 3: 0.15}
    regime_bias = np.array([regime_bias_map.get(int(r), 0.0) for r in regime], dtype=float)
    score = (
        (0.45 * (probability - 0.5) * 2.0)
        + (0.20 * trend)
        + (0.15 * breakout)
        + (0.10 * mean_rev)
        + (0.05 * vol)
        + (0.05 * regime_bias)
    )
    return np.asarray(np.clip(score, -1.0, 1.0), dtype=float)


def _clean_for_model(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out:
            out[col] = 0.0
    label_col = columns[-1]
    feature_cols = columns[:-1]
    out = out[columns].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[label_col])
    out[feature_cols] = out[feature_cols].fillna(0.0)
    if out.empty:
        raise ValueError("No rows remain after cleaning feature frame for modeling")
    return out


def to_serializable_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize metric values for metadata serialization."""

    return {key: float(value) for key, value in metrics.items()}

"""Controlled retraining workflow with drift checks and promotion gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PromotionThresholds:
    """Thresholds for controlled model promotion."""

    max_feature_drift_psi: float = 0.25
    min_shadow_auc: float = 0.52
    min_auc_delta: float = 0.0


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute PSI drift metric between two numeric distributions."""

    ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    cur = current.replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(ref, quantiles))
    if len(breakpoints) < 3:
        return 0.0

    ref_bins = pd.cut(ref, bins=breakpoints, include_lowest=True)
    cur_bins = pd.cut(cur, bins=breakpoints, include_lowest=True)
    ref_dist = ref_bins.value_counts(normalize=True).sort_index()
    cur_dist = cur_bins.value_counts(normalize=True).sort_index()
    aligned = ref_dist.to_frame("ref").join(cur_dist.to_frame("cur"), how="outer").fillna(1e-6)
    aligned = aligned.clip(lower=1e-6)
    psi = ((aligned["cur"] - aligned["ref"]) * np.log(aligned["cur"] / aligned["ref"])).sum()
    return float(psi)


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Compute feature drift map keyed by feature name."""

    drift: dict[str, float] = {}
    for col in features:
        if col not in reference_df or col not in current_df:
            continue
        drift[col] = population_stability_index(reference_df[col], current_df[col])
    return drift


def evaluate_promotion_gate(
    shadow_metrics: dict[str, float],
    production_metrics: dict[str, float] | None,
    drift_metrics: dict[str, float],
    thresholds: PromotionThresholds | None = None,
) -> tuple[bool, dict[str, str]]:
    """Apply promotion gate checks and return decision plus diagnostics."""

    cfg = thresholds or PromotionThresholds()
    diagnostics: dict[str, str] = {}

    max_drift = max(drift_metrics.values(), default=0.0)
    diagnostics["max_drift_psi"] = f"{max_drift:.4f}"
    if max_drift > cfg.max_feature_drift_psi:
        diagnostics["drift_gate"] = "failed"
        return False, diagnostics
    diagnostics["drift_gate"] = "passed"

    shadow_auc = shadow_metrics.get("roc_auc", 0.0)
    diagnostics["shadow_auc"] = f"{shadow_auc:.4f}"
    if shadow_auc < cfg.min_shadow_auc:
        diagnostics["quality_gate"] = "failed"
        return False, diagnostics

    if production_metrics is not None:
        prod_auc = production_metrics.get("roc_auc", 0.0)
        delta = shadow_auc - prod_auc
        diagnostics["production_auc"] = f"{prod_auc:.4f}"
        diagnostics["auc_delta"] = f"{delta:.4f}"
        if delta < cfg.min_auc_delta:
            diagnostics["quality_gate"] = "failed"
            return False, diagnostics

    diagnostics["quality_gate"] = "passed"
    return True, diagnostics

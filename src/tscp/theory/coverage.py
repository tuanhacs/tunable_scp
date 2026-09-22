from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..methods.tscp import TsCPClassification, TsCPRegression
from ..methods.ecp import ecp_classification_alpha
from ..quantiles import minimum_alpha_for_rank, sorted_conformal_quantiles, validate_epsilon


@dataclass(frozen=True)
class CoverageEstimate:
    alpha_hat: float
    delta_hat: float
    old_proxy: float
    corrected_bound: float
    alpha_terms: np.ndarray
    delta_terms: np.ndarray


def _result(alpha_terms: list[float], delta_terms: list[float]) -> CoverageEstimate:
    alpha = np.asarray(alpha_terms, dtype=float)
    delta = np.asarray(delta_terms, dtype=float)
    alpha_hat = float(alpha.mean())
    delta_hat = float(delta.mean())
    return CoverageEstimate(alpha_hat, delta_hat, 1.0 - alpha_hat, 1.0 - alpha_hat - delta_hat, alpha, delta)


def estimate_ecp_regression_alpha_loo(
    calibration_scores: np.ndarray,
    scales: np.ndarray,
    budgets: np.ndarray,
    epsilon: float,
    return_fallbacks: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-observation truncated-eCP adaptive levels after LOO deletion."""
    scores = np.asarray(calibration_scores, dtype=float)
    scales = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    budgets = np.asarray(budgets, dtype=float)
    n = len(scores)
    if n < 2:
        raise ValueError("eCP LOO estimation requires at least two calibration scores.")
    epsilon = validate_epsilon(epsilon)
    if np.any(budgets <= 0):
        raise ValueError("Regression eCP requires positive size budgets.")
    loo_totals = float(scores.sum()) - scores
    # The LOO calibration size is n-1, so the eCP denominator is alpha*n-1.
    required_alphas = (1.0 + 2.0 * scales * loo_totals / budgets) / n
    upper_alpha = 1.0 - epsilon
    max_denominator = upper_alpha * n - 1.0
    fallback_mask = np.full(n, max_denominator <= 0, dtype=bool)
    if max_denominator > 0:
        fallback_mask = 2.0 * scales * loo_totals / max_denominator > budgets
    alphas = np.minimum(np.maximum(epsilon, required_alphas), upper_alpha)
    for _ in range(64):
        denominators = alphas * n - 1.0
        feasible = (denominators > 0) & (
            2.0 * scales * loo_totals / denominators <= budgets
        )
        if np.all(feasible | fallback_mask):
            break
        alphas = np.where(feasible | fallback_mask, alphas, np.minimum(np.nextafter(alphas, 1.0), upper_alpha))
    denominators = alphas * n - 1.0
    feasible = (denominators > 0) & (2.0 * scales * loo_totals / denominators <= budgets)
    alphas = np.where(feasible, alphas, upper_alpha)
    if return_fallbacks:
        return alphas, fallback_mask
    return alphas


def estimate_ecp_classification_alpha_loo(
    true_scores: np.ndarray,
    candidate_scores: np.ndarray,
    budgets: np.ndarray,
    epsilon: float,
    return_fallbacks: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Per-observation truncated-eCP adaptive levels after LOO deletion."""
    true_scores = np.asarray(true_scores, dtype=float)
    candidates = np.asarray(candidate_scores, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
    n = len(true_scores)
    if n < 2:
        raise ValueError("eCP LOO estimation requires at least two calibration scores.")
    if len(candidates) != n:
        raise ValueError("Candidate-score rows must align with true-label scores.")
    total = float(true_scores.sum())
    alphas = np.asarray([
        ecp_classification_alpha(candidates[i], total - true_scores[i], n - 1,
                                 budgets[i], epsilon)
        for i in range(n)
    ])
    if not return_fallbacks:
        return alphas
    fallback_mask = np.empty(n, dtype=bool)
    for i in range(n):
        denominator = (total - true_scores[i] + candidates[i]) / n
        e_values = candidates[i] / np.maximum(denominator, 1e-12)
        fallback_mask[i] = np.count_nonzero(alphas[i] * e_values < 1.0) > np.floor(budgets[i])
    return alphas, fallback_mask


def estimate_classification_coverage(
    method: TsCPClassification,
    d1_label_scores: np.ndarray,
    d1_true_labels: np.ndarray,
    d1_budgets: np.ndarray,
) -> CoverageEstimate:
    """Algorithm-2 LOO estimates of E[alpha_delta] and E[Delta_delta,n]."""
    all_scores = np.asarray(d1_label_scores, dtype=float)
    labels = np.asarray(d1_true_labels, dtype=int)
    budgets = np.asarray(d1_budgets, dtype=float)
    if len(all_scores) != len(method.scores_d1):
        raise ValueError("D1 candidate scores must align with D1 true-label scores.")
    n = len(method.scores_d1)
    if n < 2:
        raise ValueError("LOO coverage estimation requires at least two D1 scores.")
    allowed = np.floor(budgets - method.delta).astype(int)
    if np.any(allowed < 0):
        raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
    candidate_count = all_scores.shape[1]
    clipped = np.minimum(allowed, candidate_count - 1)
    cutoffs = np.sort(all_scores, axis=1)[np.arange(n), clipped]
    max_ranks = np.searchsorted(method.ordered_d1, cutoffs, side="left")
    max_ranks -= (method.scores_d1 < cutoffs).astype(int)
    # A budget admitting every label also admits the infinity quantile.
    max_ranks[allowed >= candidate_count] = n
    alpha = minimum_alpha_for_rank(max_ranks, n - 1, method.epsilon)
    q2 = sorted_conformal_quantiles(method.ordered_d2, alpha)
    misses = (all_scores[np.arange(len(all_scores)), labels] > q2).astype(float)
    alpha_terms = alpha.tolist()
    delta_terms = (misses - alpha).tolist()
    return _result(alpha_terms, delta_terms)


def estimate_regression_coverage(
    method: TsCPRegression,
    d1_predictions: np.ndarray,
    d1_scales: np.ndarray,
    d1_targets: np.ndarray,
    d1_budgets: np.ndarray,
) -> CoverageEstimate:
    predictions = np.asarray(d1_predictions, dtype=float)
    scales = np.asarray(d1_scales, dtype=float)
    targets = np.asarray(d1_targets, dtype=float)
    budgets = np.asarray(d1_budgets, dtype=float)
    if len(predictions) != len(method.scores_d1):
        raise ValueError("D1 observations must align with D1 scores.")
    true_scores = np.abs(targets - predictions) / np.maximum(scales, 1e-12)
    n = len(method.scores_d1)
    if n < 2:
        raise ValueError("LOO coverage estimation requires at least two D1 scores.")
    targets = budgets - method.delta
    if np.any(targets < 0):
        raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
    thresholds = targets / (2.0 * np.maximum(scales, 1e-12))
    max_ranks = np.searchsorted(method.ordered_d1, thresholds, side="right")
    max_ranks -= (method.scores_d1 <= thresholds).astype(int)
    alpha = minimum_alpha_for_rank(max_ranks, n - 1, method.epsilon)
    q2 = sorted_conformal_quantiles(method.ordered_d2, alpha)
    misses = (true_scores > q2).astype(float)
    alpha_terms = alpha.tolist()
    delta_terms = (misses - alpha).tolist()
    return _result(alpha_terms, delta_terms)

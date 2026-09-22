from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the ceil((n+1)(1-alpha))-th conformal order statistic."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one calibration score is required.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha))))
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 1/2).")
    return epsilon


def conformal_ranks(sample_size: int, alphas: np.ndarray | float) -> np.ndarray:
    """Finite-sample quantile ranks, including the infinity rank m+1."""
    values = np.asarray(alphas, dtype=float)
    return np.ceil((sample_size + 1) * (1.0 - values)).astype(int)


def minimum_alpha_for_rank(max_ranks: np.ndarray | int, sample_size: int, epsilon: float) -> np.ndarray:
    """Smallest alpha in [epsilon, 1-epsilon] with quantile rank <= max_rank."""
    epsilon = validate_epsilon(epsilon)
    ranks = np.asarray(max_ranks, dtype=int)
    lower_rank = int(conformal_ranks(sample_size, epsilon))
    upper_rank = int(conformal_ranks(sample_size, 1.0 - epsilon))
    if np.any(ranks < upper_rank):
        raise ValueError("No alpha in [epsilon, 1-epsilon] satisfies the size budget.")
    result = np.maximum(epsilon, (sample_size + 1 - ranks) / (sample_size + 1))
    result = np.where(ranks >= lower_rank, epsilon, result)
    # Guard against roundoff at an empirical rank breakpoint.
    for _ in range(4):
        wrong = conformal_ranks(sample_size, result) > ranks
        if not np.any(wrong):
            break
        result = np.where(wrong, np.nextafter(result, 1.0), result)
    if np.any(conformal_ranks(sample_size, result) > ranks) or np.any(result > 1.0 - epsilon):
        raise ValueError("No representable alpha in [epsilon, 1-epsilon] satisfies the size budget.")
    return result


def sorted_conformal_quantiles(ordered_scores: np.ndarray, alphas: np.ndarray | float) -> np.ndarray:
    """Look up conformal quantiles from scores already sorted in ascending order."""
    ordered = np.asarray(ordered_scores, dtype=float).reshape(-1)
    ranks = conformal_ranks(len(ordered), alphas)
    clipped = np.clip(ranks - 1, 0, len(ordered) - 1)
    return np.where(ranks > len(ordered), np.inf, ordered[clipped])

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..quantiles import minimum_alpha_for_rank, sorted_conformal_quantiles, validate_epsilon


@dataclass
class TsCPClassification:
    scores_d1: np.ndarray
    scores_d2: np.ndarray
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        self.scores_d1 = np.asarray(self.scores_d1, dtype=float)
        self.scores_d2 = np.asarray(self.scores_d2, dtype=float)
        self.epsilon = validate_epsilon(self.epsilon)
        if self.delta < 0:
            raise ValueError("delta must be nonnegative.")
        self.ordered_d1 = np.sort(self.scores_d1)
        self.ordered_d2 = np.sort(self.scores_d2)

    def choose_alpha(self, label_scores: np.ndarray, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        candidates = np.asarray(label_scores, dtype=float).reshape(-1)
        allowed = int(np.floor(target))
        if allowed >= len(candidates):
            return self.epsilon
        ordered = self.ordered_d1 if scores_d1 is None else np.sort(np.asarray(scores_d1, dtype=float))
        # q_a(alpha) must be strictly below the (allowed+1)-st score.
        cutoff = np.partition(candidates, allowed)[allowed]
        max_rank = int(np.searchsorted(ordered, cutoff, side="left"))
        return float(minimum_alpha_for_rank(max_rank, len(ordered), self.epsilon))

    def predict_one(self, label_scores: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
        alpha = self.choose_alpha(label_scores, budget)
        q2 = float(sorted_conformal_quantiles(self.ordered_d2, alpha))
        return np.asarray(label_scores) <= q2, alpha


@dataclass
class TsCPRegression:
    scores_d1: np.ndarray
    scores_d2: np.ndarray
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        self.scores_d1 = np.asarray(self.scores_d1, dtype=float)
        self.scores_d2 = np.asarray(self.scores_d2, dtype=float)
        self.epsilon = validate_epsilon(self.epsilon)
        if self.delta < 0:
            raise ValueError("delta must be nonnegative.")
        self.ordered_d1 = np.sort(self.scores_d1)
        self.ordered_d2 = np.sort(self.scores_d2)

    def choose_alpha(self, scale: float, budget: float, scores_d1: np.ndarray | None = None) -> float:
        target = float(budget) - self.delta
        if target < 0:
            raise ValueError("The theoretical construction requires S(x)-delta >= 0.")
        ordered = self.ordered_d1 if scores_d1 is None else np.sort(np.asarray(scores_d1, dtype=float))
        threshold = target / (2.0 * max(float(scale), 1e-12))
        max_rank = int(np.searchsorted(ordered, threshold, side="right"))
        return float(minimum_alpha_for_rank(max_rank, len(ordered), self.epsilon))

    def predict_one(self, prediction: float, scale: float, budget: float) -> tuple[tuple[float, float], float]:
        alpha = self.choose_alpha(scale, budget)
        q2 = float(sorted_conformal_quantiles(self.ordered_d2, alpha))
        radius = max(float(scale), 1e-12) * q2
        return (float(prediction - radius), float(prediction + radius)), alpha

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..quantiles import validate_epsilon


def ecp_classification_alpha(scores: np.ndarray, calibration_sum: float,
                             calibration_size: int, budget: float, epsilon: float) -> float:
    """Find the first exact e-value inclusion breakpoint meeting the budget."""
    epsilon = validate_epsilon(epsilon)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    allowed = int(np.floor(float(budget)))
    if allowed < 0:
        raise ValueError("The prediction-set budget must be nonnegative.")
    if allowed >= len(scores):
        return epsilon
    denominator = (calibration_sum + scores) / (calibration_size + 1)
    e_values = scores / np.maximum(denominator, 1e-12)
    breakpoints = np.divide(1.0, e_values, out=np.full_like(e_values, np.inf), where=e_values > 0)
    threshold = float(np.partition(breakpoints, len(scores) - allowed - 1)[len(scores) - allowed - 1])
    alpha = max(epsilon, threshold)
    upper_alpha = 1.0 - epsilon
    if alpha > upper_alpha:
        # No feasible level exists: return the smallest set attainable in the
        # allowed alpha range, even though it may exceed the requested budget.
        return upper_alpha
    # Inclusion is strict: e_y < 1/alpha, so a label is excluded at its
    # breakpoint.  Adjust only if rounding places us on the wrong side.
    for _ in range(4):
        if np.count_nonzero(alpha * e_values < 1.0) <= allowed:
            return alpha
        alpha = float(np.nextafter(alpha, 1.0))
    return upper_alpha


def ecp_regression_alpha(calibration_sum: float, calibration_size: int,
                         scale: float, budget: float, epsilon: float) -> float:
    """Solve the eCP interval-length inequality directly for alpha."""
    epsilon = validate_epsilon(epsilon)
    if budget <= 0:
        raise ValueError("Regression eCP requires a positive size budget.")
    scale = max(float(scale), 1e-12)
    total = float(calibration_sum)
    numerator = 2.0 * scale * total
    upper_alpha = 1.0 - epsilon
    upper_denominator = upper_alpha * (calibration_size + 1) - 1.0
    minimum_size = (
        numerator / upper_denominator if upper_denominator > 0 else float("inf")
    )
    required_alpha = (1.0 + numerator / budget) / (calibration_size + 1)
    if minimum_size > budget:
        # Markov's bound can be too loose for the requested size.  As in the
        # original grid implementation, use the smallest attainable eCP set.
        return float(upper_alpha)
    alpha = max(epsilon, required_alpha)
    # The analytic breakpoint can round down by a few ULPs; move only as far
    # as needed to satisfy the same floating-point length check as predict_one.
    for _ in range(64):
        denominator = alpha * (calibration_size + 1) - 1.0
        if denominator > 0 and numerator / denominator <= budget:
            if alpha <= upper_alpha:
                return float(alpha)
            break
        alpha = float(np.nextafter(alpha, 1.0))
        if alpha > upper_alpha:
            break
    if upper_denominator > 0 and numerator / upper_denominator <= budget:
        # Only reachable when the exact breakpoint is too close to the upper
        # endpoint for the finite-ULP correction above.
        return float(upper_alpha)
    return float(upper_alpha)


@dataclass
class ECPClassification:
    calibration_scores: np.ndarray
    epsilon: float

    def __post_init__(self) -> None:
        self.calibration_scores = np.asarray(self.calibration_scores, dtype=float)
        self.epsilon = validate_epsilon(self.epsilon)

    def predict_one(self, label_scores: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
        scores = np.asarray(label_scores, dtype=float)
        denominator = (self.calibration_scores.sum() + scores) / (len(self.calibration_scores) + 1)
        e_values = scores / np.maximum(denominator, 1e-12)
        alpha = ecp_classification_alpha(scores, float(self.calibration_scores.sum()),
                                         len(self.calibration_scores), budget, self.epsilon)
        return alpha * e_values < 1.0, alpha


@dataclass
class ECPRegression:
    calibration_scores: np.ndarray
    epsilon: float

    def __post_init__(self) -> None:
        self.calibration_scores = np.asarray(self.calibration_scores, dtype=float)
        self.epsilon = validate_epsilon(self.epsilon)

    def predict_one(self, prediction: float, scale: float, budget: float) -> tuple[tuple[float, float], float]:
        total = self.calibration_scores.sum()
        n = len(self.calibration_scores)
        chosen_alpha = ecp_regression_alpha(float(total), n, scale, budget, self.epsilon)
        radius_score = total / (chosen_alpha * (n + 1) - 1.0)
        radius = max(float(scale), 1e-12) * radius_score
        return (float(prediction - radius), float(prediction + radius)), chosen_alpha

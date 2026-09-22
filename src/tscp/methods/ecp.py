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
    if alpha > 1.0 - epsilon:
        raise ValueError("No alpha in [epsilon, 1-epsilon] satisfies the size budget.")
    # Inclusion is strict: e_y < 1/alpha, so a label is excluded at its
    # breakpoint.  Adjust only if rounding places us on the wrong side.
    for _ in range(4):
        if np.count_nonzero(alpha * e_values < 1.0) <= allowed:
            return alpha
        alpha = float(np.nextafter(alpha, 1.0))
    raise ValueError("No representable alpha satisfies the eCP size budget.")


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
        raise ValueError(
            "No alpha in [epsilon, 1-epsilon] satisfies the eCP size budget: "
            f"reason={'requires_alpha_above_one' if required_alpha > 1.0 else 'epsilon_upper_boundary'}, "
            f"alpha_required={required_alpha:.17g}, alpha_max={upper_alpha:.17g}, "
            f"size_at_alpha_max={minimum_size:.17g}, budget={budget:.17g}, "
            f"scale={scale:.17g}, calibration_score_sum={total:.17g}, "
            f"calibration_size={calibration_size}."
        )
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
    raise ValueError(
        "eCP regression alpha search failed a numerical boundary check: "
        f"alpha_required={required_alpha:.17g}, alpha_max={upper_alpha:.17g}, "
        f"size_at_alpha_max={minimum_size:.17g}, budget={budget:.17g}, "
        f"scale={scale:.17g}, calibration_score_sum={total:.17g}, "
        f"calibration_size={calibration_size}."
    )


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

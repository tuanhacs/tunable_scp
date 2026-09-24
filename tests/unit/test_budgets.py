import numpy as np
import pytest

from tscp.budgets import BudgetSpec, evaluate_budget, regression_uncertainty


@pytest.mark.parametrize("kind", ["linear", "quadratic", "exponential"])
def test_adaptive_budgets_are_bounded_and_monotone(kind):
    spec = BudgetSpec(kind=kind, minimum=2.0, maximum=8.0, beta=2.0)
    values = evaluate_budget(spec, np.array([0.0, 0.25, 0.5, 1.0]))
    assert values[0] == 2.0
    assert values[-1] == pytest.approx(8.0)
    assert np.all(np.diff(values) >= 0)


def test_classification_budget_is_integer_valued():
    spec = BudgetSpec(kind="linear", minimum=1, maximum=5)
    values = evaluate_budget(spec, np.array([0.1, 0.6]), classification=True)
    assert np.all(values == np.ceil(values))


def test_log_sigma_budget_normalizes_after_log_transform():
    reference = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    values = np.array([2.0, 4.0, 8.0])
    sigma = regression_uncertainty(values, reference, "sigma")
    log_sigma = regression_uncertainty(values, reference, "log_sigma")
    assert np.all(np.diff(log_sigma) > 0)
    assert not np.allclose(sigma, log_sigma)


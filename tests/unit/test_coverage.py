import numpy as np
import pytest

from tscp.methods.ecp import ECPClassification, ECPRegression
from tscp.methods.tscp import TsCPClassification, TsCPRegression
from tscp.quantiles import conformal_quantile, minimum_alpha_for_rank
from tscp.theory.coverage import (
    estimate_classification_coverage,
    estimate_ecp_classification_alpha_loo,
    estimate_ecp_regression_alpha_loo,
    estimate_regression_coverage,
)


def test_rank_selection_uses_exact_empirical_breakpoint():
    alpha = float(minimum_alpha_for_rank(2, 4, 1e-10))
    assert alpha == pytest.approx(0.6)
    assert conformal_quantile(np.arange(1.0, 5.0), alpha) == 2.0
    assert conformal_quantile(np.arange(1.0, 5.0), alpha - 1e-3) == 3.0


def test_ecp_regression_loo_matches_explicit_point_deletion():
    scores = np.array([0.2, 0.5, 0.8, 1.1])
    scales = np.array([0.8, 1.0, 1.2, 1.4])
    budgets = np.array([2.0, 2.0, 2.5, 3.0])
    epsilon = 1e-10
    expected = np.array([
        ECPRegression(np.delete(scores, i), epsilon).predict_one(0.0, scales[i], budgets[i])[1]
        for i in range(len(scores))
    ])
    actual = estimate_ecp_regression_alpha_loo(scores, scales, budgets, epsilon)
    np.testing.assert_allclose(actual, expected)


def test_ecp_regression_loo_falls_back_and_reports_mask():
    scores = np.ones(4)
    scales = np.array([10.0, 1.0, 1.0, 1.0])
    budgets = np.ones(4)
    alphas, fallback_mask = estimate_ecp_regression_alpha_loo(
        scores, scales, budgets, 1e-10, return_fallbacks=True,
    )
    assert fallback_mask[0]
    assert alphas[0] == pytest.approx(1.0 - 1e-10)
    np.testing.assert_allclose(alphas, [
        ECPRegression(np.delete(scores, i), 1e-10).predict_one(0.0, scales[i], budgets[i])[1]
        for i in range(len(scores))
    ])


def test_ecp_classification_loo_matches_explicit_point_deletion():
    candidates = np.array([
        [0.1, 0.7, 0.9], [0.6, 0.2, 0.8], [0.7, 0.9, 0.3], [0.4, 0.5, 0.8],
    ])
    labels = np.array([0, 1, 2, 0])
    true_scores = candidates[np.arange(len(labels)), labels]
    budgets = np.array([2, 2, 1, 2])
    epsilon = 1e-10
    expected = np.array([
        ECPClassification(np.delete(true_scores, i), epsilon).predict_one(candidates[i], budgets[i])[1]
        for i in range(len(true_scores))
    ])
    actual = estimate_ecp_classification_alpha_loo(true_scores, candidates, budgets, epsilon)
    np.testing.assert_allclose(actual, expected)


def test_tscp_regression_selects_exact_rank_and_loo_matches_deletion():
    d1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    method = TsCPRegression(d1, d1 + 0.02, 1e-10, 0.0)
    assert method.choose_alpha(1.0, 0.65) == pytest.approx(0.5)
    scales = np.ones(len(d1))
    budgets = np.full(len(d1), 0.65)
    estimate = estimate_regression_coverage(method, np.zeros(len(d1)), scales, d1, budgets)
    expected = np.array([
        method.choose_alpha(scales[i], budgets[i], np.delete(d1, i))
        for i in range(len(d1))
    ])
    np.testing.assert_allclose(estimate.alpha_terms, expected)


def test_tscp_classification_loo_matches_deletion_with_ties():
    d1 = np.array([0.15, 0.25, 0.25, 0.45, 0.55])
    candidates = np.array([
        [0.1, 0.4, 0.8], [0.2, 0.5, 0.7], [0.1, 0.5, 0.8],
        [0.2, 0.3, 0.9], [0.15, 0.4, 0.7],
    ])
    budgets = np.full(len(d1), 1.0)
    method = TsCPClassification(d1, d1 + 0.01, 1e-10, 0.0)
    estimate = estimate_classification_coverage(
        method, candidates, np.zeros(len(d1), dtype=int), budgets,
    )
    expected = np.array([
        method.choose_alpha(candidates[i], budgets[i], np.delete(d1, i))
        for i in range(len(d1))
    ])
    np.testing.assert_allclose(estimate.alpha_terms, expected)


def test_ecp_regression_uses_closed_form_not_grid():
    method = ECPRegression(np.array([0.2, 0.3, 0.4, 0.5]), 1e-10)
    interval, alpha = method.predict_one(0.0, 1.0, 2.0)
    assert alpha == pytest.approx((1.0 + 1.4) / 5.0)
    assert interval[1] - interval[0] <= 2.0 + 1e-12


def test_ecp_regression_large_budget_is_feasible():
    method = ECPRegression(np.full(500, 1.0), 1e-10)
    interval, alpha = method.predict_one(0.0, 7.563318567907845, 500.0)
    assert 0.0 < alpha < 1.0
    assert interval[1] - interval[0] <= 500.0


def test_ecp_regression_infeasibility_falls_back_to_upper_alpha():
    method = ECPRegression(np.full(499, 496.4618698511901 / 499), 1e-10)
    interval, alpha = method.predict_one(0.0, 7.563318567907845, 15.0)
    assert alpha == pytest.approx(1.0 - 1e-10)
    assert interval[1] - interval[0] > 15.0


def test_ecp_classification_infeasibility_falls_back_to_upper_alpha():
    method = ECPClassification(np.array([0.1, 0.1, 0.1]), 1e-10)
    prediction_set, alpha = method.predict_one(np.array([0.1, 0.1, 0.1]), 0.0)
    assert alpha == pytest.approx(1.0 - 1e-10)
    assert prediction_set.sum() > 0


def test_ecp_classification_uses_exact_evalue_breakpoint():
    method = ECPClassification(np.array([0.1, 0.1, 0.1]), 1e-10)
    prediction_set, alpha = method.predict_one(np.array([0.2, 0.3, 0.5]), 1.0)
    assert alpha == pytest.approx(0.5)
    assert prediction_set.sum() == 1


def test_infeasible_budget_raises_instead_of_silent_fallback():
    method = TsCPRegression(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.1, 0.0)
    with pytest.raises(ValueError, match="No alpha"):
        method.choose_alpha(1.0, 0.1)


def test_corrected_classification_bound_uses_delta_hat():
    method = TsCPClassification(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.15, 0.25, 0.35, 0.45]), 1e-10, 0.0)
    label_scores = np.array([[0.1, 0.8], [0.2, 0.7], [0.3, 0.6], [0.4, 0.5]])
    estimate = estimate_classification_coverage(method, label_scores, np.array([0, 0, 0, 0]), np.full(4, 2.0))
    assert estimate.corrected_bound == pytest.approx(1.0 - estimate.alpha_hat - estimate.delta_hat)
    assert len(estimate.delta_terms) == 4


def test_corrected_regression_bound_uses_delta_hat():
    method = TsCPRegression(np.array([0.2, 0.4, 0.6]), np.array([0.3, 0.5, 0.7]), 1e-10, 0.0)
    estimate = estimate_regression_coverage(method, np.zeros(3), np.ones(3), np.array([0.2, 0.4, 0.6]), np.full(3, 4.0))
    assert estimate.corrected_bound == pytest.approx(1.0 - estimate.alpha_hat - estimate.delta_hat)

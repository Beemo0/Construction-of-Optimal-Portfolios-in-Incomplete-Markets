"""
Tests for optiport.continuous.black_scholes.

The central test (``test_merton_equals_dual_pathwise``) checks that
Merton's dynamic-programming solution and the continuous-time duality
solution of Chapter 4 coincide *exactly* (up to floating-point
tolerance), pathwise, in the complete Black-Scholes market -- the
primary correctness test flagged in Chapters 4, 6 and 7 (Experiment
1). The remaining tests validate the budget constraint and the
closed-form value function against Monte Carlo estimates.
"""

import numpy as np
import pytest

from optiport.continuous.black_scholes import BlackScholesMarket
from optiport.utility import CRRAUtility

MARKET = BlackScholesMarket(mu=0.10, sigma=0.20, r=0.02)
T = 1.0
X0 = 100.0


@pytest.mark.parametrize("alpha", [-3.0, -1.0, -0.5, 0.5, 0.9])
def test_merton_equals_dual_pathwise(alpha):
    utility = CRRAUtility(alpha)
    W_T = np.linspace(-2.0, 2.0, 41)  # deterministic grid: an algebraic
    # identity should hold at every point, not just on average.
    x_merton = MARKET.merton_terminal_wealth(X0, utility, T, W_T)
    x_dual = MARKET.dual_terminal_wealth(X0, utility, T, W_T)
    assert np.allclose(x_merton, x_dual, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.5])
def test_budget_constraint_holds_in_expectation(alpha):
    utility = CRRAUtility(alpha)
    rng = np.random.default_rng(42)
    n = 2_000_000
    W_T = rng.normal(0.0, np.sqrt(T), n)
    H_T = MARKET.state_price_density(W_T, T)
    x_merton = MARKET.merton_terminal_wealth(X0, utility, T, W_T)
    budget = np.mean(H_T * x_merton)
    # Monte Carlo tolerance: relative error within ~1% for 2M paths.
    assert abs(budget - X0) / X0 < 0.02


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.5])
def test_value_function_matches_monte_carlo(alpha):
    utility = CRRAUtility(alpha)
    rng = np.random.default_rng(7)
    n = 2_000_000
    W_T = rng.normal(0.0, np.sqrt(T), n)
    x_merton = MARKET.merton_terminal_wealth(X0, utility, T, W_T)
    mc_value = np.mean(utility.U(x_merton))
    analytic_value = MARKET.merton_value_function(X0, utility, T)
    assert abs(mc_value - analytic_value) / abs(analytic_value) < 0.02


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.5])
def test_expected_utility_constant_mix_matches_merton_at_optimum(alpha):
    utility = CRRAUtility(alpha)
    pi_star = MARKET.merton_optimal_pi(utility)
    analytic_via_merton = MARKET.merton_value_function(X0, utility, T)
    analytic_via_general_formula = MARKET.expected_utility_constant_mix(X0, utility, T, pi_star)
    assert np.isclose(analytic_via_merton, analytic_via_general_formula, rtol=1e-10)


@pytest.mark.parametrize("pi", [0.0, 0.3, 0.8, 1.5])
def test_expected_utility_constant_mix_matches_monte_carlo(pi):
    alpha = -1.0
    utility = CRRAUtility(alpha)
    rng = np.random.default_rng(99)
    n = 2_000_000
    W_T = rng.normal(0.0, np.sqrt(T), n)
    drift = MARKET.r + pi * (MARKET.mu - MARKET.r) - 0.5 * pi**2 * MARKET.sigma**2
    X_T = X0 * np.exp(drift * T + pi * MARKET.sigma * W_T)
    mc_value = np.mean(utility.U(X_T))
    analytic_value = MARKET.expected_utility_constant_mix(X0, utility, T, pi)
    assert abs(mc_value - analytic_value) / abs(analytic_value) < 0.02


def test_log_utility_does_not_raise_zero_division():
    """Regression test: alpha=0 (log utility) triggered a
    ZeroDivisionError in both merton_value_function and
    expected_utility_constant_mix, since both originally divided by
    alpha unconditionally. Both now branch to the correct additive
    (log) formula instead.
    """
    utility = CRRAUtility(0.0)
    v = MARKET.merton_value_function(X0, utility, T)
    pi_star = MARKET.merton_optimal_pi(utility)
    eu = MARKET.expected_utility_constant_mix(X0, utility, T, pi_star)
    assert np.isfinite(v) and np.isfinite(eu)
    assert np.isclose(v, eu, atol=1e-10)


def test_forgetting_the_discount_factor_breaks_the_identity():
    """Regression test for the bug found during development: using
    Z_T = dQ/dP directly, instead of the state-price density
    H_T = exp(-r*T) * Z_T, in the budget constraint produces a
    terminal wealth that is a *constant multiple* of the correct one,
    not the correct one itself -- and the discrepancy is visible even
    though both quantities look plausible in isolation.
    """
    alpha = -1.0
    utility = CRRAUtility(alpha)
    theta = MARKET.theta
    W_T = np.linspace(-1.0, 1.0, 5)

    x_merton = MARKET.merton_terminal_wealth(X0, utility, T, W_T)

    # Deliberately wrong calibration: omit exp(-r*T) in the moment used
    # to calibrate y (i.e. treat H_T as if it were just Z_T).
    p = alpha / (alpha - 1.0)
    E_Zp_wrong = np.exp(0.5 * p * (p - 1.0) * theta**2 * T)  # missing exp(-r*T*p)
    y_wrong = (X0 / E_Zp_wrong) ** (alpha - 1.0)
    Z_T = np.exp(-theta * W_T - 0.5 * theta**2 * T)
    x_dual_wrong = utility.I(y_wrong * Z_T)

    assert not np.allclose(x_merton, x_dual_wrong, rtol=1e-6)

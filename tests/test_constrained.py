"""
Tests for optiport.constraints.constrained_black_scholes.

The central test (``test_constrained_equals_dual_pathwise``) checks
the pathwise coincidence between the constrained investor's actual
terminal wealth (constant-mix pi_c under the real market) and the
constrained duality first-order condition of Chapter 5, across all
three qualitative cases: upper bound binding (leverage cap), lower
bound binding (no-short-sale), and the constraint not binding at all.

A second test is a deliberate regression test for a bug found during
development: omitting the extra discount rate delta(nu*) -- using only
the shifted market price of risk, as if the constrained problem were
just an unconstrained problem in a market with a different mu -- looks
locally plausible (the resulting "dual" wealth is a smooth, correctly-
shaped function of W_T) but is off by a constant multiplicative factor,
exactly as forgetting exp(-rT) was in the unconstrained module.
"""

import numpy as np
import pytest

from optiport.continuous.black_scholes import BlackScholesMarket
from optiport.constraints.constrained_black_scholes import ConstrainedBlackScholesMarket
from optiport.constraints.support_function import support_function
from optiport.utility import CRRAUtility

MARKET = BlackScholesMarket(mu=0.10, sigma=0.20, r=0.02)
T, X0 = 1.0, 100.0


@pytest.mark.parametrize(
    "a,b,label",
    [
        (-np.inf, 0.5, "upper bound (leverage cap) binds"),
        (0.0, np.inf, "lower bound (no-short-sale) binds"),
        (0.0, 2.0, "constraint not binding"),
    ],
)
@pytest.mark.parametrize("alpha", [-2.0, -1.0, 0.5])
def test_constrained_equals_dual_pathwise(a, b, label, alpha):
    utility = CRRAUtility(alpha)
    cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=a, b=b)
    W_T = np.linspace(-2.0, 2.0, 41)
    x_real = cmarket.real_constrained_terminal_wealth(X0, utility, T, W_T)
    x_dual = cmarket.dual_terminal_wealth(X0, utility, T, W_T)
    assert np.allclose(x_real, x_dual, rtol=1e-8, atol=1e-6), label


def test_no_short_sale_binds_when_mu_below_r():
    """With mu < r, the unconstrained Merton proportion is negative
    (short the risky asset), so a no-short-sale constraint [0, inf)
    must bind, and the constrained optimum is exactly 0.
    """
    low_mu_market = BlackScholesMarket(mu=0.00, sigma=0.20, r=0.02)
    utility = CRRAUtility(-1.0)
    cmarket = ConstrainedBlackScholesMarket(market=low_mu_market, a=0.0, b=np.inf)
    assert cmarket.is_binding(utility)
    assert np.isclose(cmarket.constrained_optimal_pi(utility), 0.0)


def test_constraint_not_binding_gives_zero_shadow_price_and_zero_loss():
    utility = CRRAUtility(-1.0)
    cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=-10.0, b=10.0)
    assert not cmarket.is_binding(utility)
    nu_star, penalty = cmarket.shadow_price_and_penalty(utility)
    assert np.isclose(nu_star, 0.0, atol=1e-12)
    assert np.isclose(penalty, 0.0, atol=1e-12)
    assert np.isclose(cmarket.utility_loss(X0, utility, T), 0.0, atol=1e-10)


@pytest.mark.parametrize("alpha", [-2.0, -1.0, 0.5])
def test_utility_loss_is_nonnegative_when_binding(alpha):
    utility = CRRAUtility(alpha)
    cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=-np.inf, b=0.5)
    assert cmarket.is_binding(utility)
    loss = cmarket.utility_loss(X0, utility, T)
    assert loss > 0.0


def test_utility_loss_increases_as_leverage_cap_tightens():
    utility = CRRAUtility(-1.0)
    pi_star = MARKET.merton_optimal_pi(utility)
    caps = [pi_star * f for f in (1.5, 1.0, 0.75, 0.5, 0.25)]
    losses = []
    for cap in caps:
        cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=-np.inf, b=cap)
        losses.append(cmarket.utility_loss(X0, utility, T))
    # losses should be non-decreasing as the cap tightens (cap decreasing)
    assert all(losses[i] <= losses[i + 1] + 1e-12 for i in range(len(losses) - 1))
    assert np.isclose(losses[0], 0.0, atol=1e-10)  # cap above pi_star: not binding


def test_forgetting_the_support_function_penalty_breaks_the_identity():
    """Regression test: shifting only the market price of risk
    (equivalent to shifting mu), without the extra discount rate
    delta(nu*), reproduces the correct *shape* of the constrained
    dual wealth as a function of W_T but not its exact level.
    """
    alpha = -1.0
    utility = CRRAUtility(alpha)
    cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=-np.inf, b=0.5)
    W_T = np.linspace(-1.0, 1.0, 9)

    x_real = cmarket.real_constrained_terminal_wealth(X0, utility, T, W_T)

    aux_market, c = cmarket.auxiliary_market_and_extra_discount(utility)
    assert c > 0.0  # the penalty is strictly positive whenever the constraint binds

    # Deliberately wrong: use the auxiliary market's own (unconstrained)
    # duality formula, omitting the extra discount rate c.
    x_dual_wrong = aux_market.dual_terminal_wealth(X0, utility, T, W_T)
    x_dual_correct = cmarket.dual_terminal_wealth(X0, utility, T, W_T)

    assert np.allclose(x_dual_correct, x_real, rtol=1e-8)
    assert not np.allclose(x_dual_wrong, x_real, rtol=1e-6)
    ratio = x_dual_wrong / x_real
    assert np.allclose(ratio, ratio[0], atol=1e-8)  # constant, but wrong, offset

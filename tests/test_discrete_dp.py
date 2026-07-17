"""
Tests for optiport.discrete.dynamic_programming.

These tests validate the backward-induction solver against two exact
theoretical facts (the correct content of Experiment 2, Chapter 7):

1. Under i.i.d. returns (degenerate single-state regime market),
   *every* CRRA utility (log or power) is myopic: the dynamically
   optimal proportion coincides with the one-period myopic policy at
   every date (Samuelson's 1969 discrete-time myopic-policy theorem).

2. Under a regime-switching market whose transitions are correlated
   with the realised branch, the logarithmic investor remains myopic
   at every date and in every regime (Merton's classical zero-hedging-
   demand result for log utility), while a non-log CRRA investor's
   dynamically optimal policy strictly differs from the myopic one at
   dates away from maturity, reflecting a genuine intertemporal
   hedging demand.
"""

import numpy as np
import pytest

from optiport.discrete.regime_market import (
    RegimeSwitchingBinomialMarket,
    iid_market_as_regime_switching,
)
from optiport.discrete.dynamic_programming import (
    solve_backward_induction,
    myopic_policy,
)
from optiport.utility import CRRAUtility


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.0, 0.5])
def test_iid_market_is_always_myopic(alpha):
    market = iid_market_as_regime_switching(
        n_periods=8, u=1.08, d=0.95, p_up=0.55
    )
    utility = CRRAUtility(alpha)
    result = solve_backward_induction(market, utility)
    pi_m = myopic_policy(market, utility)
    # Every date's dynamically optimal pi must equal the myopic one,
    # for every CRRA parameter, since returns are i.i.d. (Samuelson).
    assert np.allclose(result.pi[:, 0], pi_m[0], atol=1e-6)


def _correlated_regime_market(n_periods: int) -> RegimeSwitchingBinomialMarket:
    """Two-regime market ('bull' = 0, 'bear' = 1) whose transition
    depends on the realised branch: an up-move makes staying in /
    moving to the bull regime more likely, and conversely for a
    down-move. This branch-correlation is what generates a non-zero
    hedging demand for non-log CRRA utilities.
    """
    return RegimeSwitchingBinomialMarket(
        n_periods=n_periods,
        u=np.array([1.12, 1.04]),
        d=np.array([0.93, 0.97]),
        p_up=np.array([0.60, 0.40]),
        trans_up=np.array([[0.85, 0.15], [0.55, 0.45]]),
        trans_down=np.array([[0.45, 0.55], [0.15, 0.85]]),
    )


def test_log_utility_is_myopic_under_correlated_regimes():
    market = _correlated_regime_market(n_periods=10)
    utility = CRRAUtility(0.0)
    result = solve_backward_induction(market, utility)
    pi_m = myopic_policy(market, utility)
    for s in range(market.n_states):
        assert np.allclose(result.pi[:, s], pi_m[s], atol=1e-6), (
            "Log utility must be myopic at every date and regime, "
            "even under branch-correlated regime switching."
        )


@pytest.mark.parametrize("alpha", [-3.0, -1.0, 0.5])
def test_non_log_crra_has_hedging_demand_under_correlated_regimes(alpha):
    market = _correlated_regime_market(n_periods=10)
    utility = CRRAUtility(alpha)
    result = solve_backward_induction(market, utility)
    pi_m = myopic_policy(market, utility)
    # Away from maturity, the dynamically optimal policy must differ
    # from the myopic one in at least one regime: a genuine
    # intertemporal hedging demand.
    early_gap = np.max(np.abs(result.pi[0, :] - pi_m))
    assert early_gap > 1e-4, (
        "Non-log CRRA under correlated regime-switching should exhibit "
        "a strictly non-zero hedging demand away from maturity."
    )
    # The very last decision date (t = T-1) has no continuation value
    # left to hedge, so it must coincide exactly with the myopic policy.
    assert np.allclose(result.pi[-1, :], pi_m, atol=1e-6)


def test_hedging_demand_vanishes_as_alpha_to_zero():
    market = _correlated_regime_market(n_periods=10)
    gaps = []
    for alpha in [-0.5, -0.1, -0.01]:
        utility = CRRAUtility(alpha)
        result = solve_backward_induction(market, utility)
        pi_m = myopic_policy(market, utility)
        gaps.append(np.max(np.abs(result.pi[0, :] - pi_m)))
    # The hedging-demand gap should shrink monotonically as alpha -> 0
    # (utility -> logarithmic), consistent with log utility's exactly
    # zero hedging demand established above.
    assert gaps[0] > gaps[1] > gaps[2]

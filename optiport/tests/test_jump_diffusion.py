"""
Tests for optiport.continuous.jump_diffusion.

Validates, by Monte Carlo (importance-sampling style, using the
P-simulated primitives and the explicit Radon-Nikodym derivative):

1. the martingale condition E^P[Z_T * S_T] = S0 * exp(r*T) for several
   values of beta in the Esscher sub-family (Chapter 4's fundamental
   theorem of asset pricing, applied pathwise);
2. the closed-form relative entropy D(Q_beta || P) against its Monte
   Carlo estimate E^P[Z_T * log Z_T];
3. that ``minimal_entropy_beta`` finds a genuine interior minimum of
   the entropy, strictly better than beta = 0 and other test points.
"""

import numpy as np
import pytest

from optiport.continuous.jump_diffusion import MertonJumpDiffusionMarket

MARKET = MertonJumpDiffusionMarket(
    mu=0.10, sigma=0.15, r=0.02, lam=0.8, jump_mean=-0.05, jump_std=0.10
)
T = 1.0
S0 = 100.0
N_PATHS = 3_000_000


@pytest.mark.parametrize("beta", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_martingale_condition_holds(beta):
    prim = MARKET.simulate_primitives(N_PATHS, T, seed=1)
    S_T = MARKET.terminal_price(S0, T, prim)
    Z_T = MARKET.radon_nikodym_Qbeta(T, beta, prim)
    budget = np.mean(Z_T * S_T)
    target = S0 * np.exp(MARKET.r * T)
    assert abs(budget - target) / target < 0.02


@pytest.mark.parametrize("beta", [-1.0, 0.0, 1.5])
def test_radon_nikodym_has_unit_mean(beta):
    # A sanity check that Q_beta is a genuine probability measure:
    # E^P[dQ_beta/dP] = 1.
    prim = MARKET.simulate_primitives(N_PATHS, T, seed=2)
    Z_T = MARKET.radon_nikodym_Qbeta(T, beta, prim)
    assert abs(np.mean(Z_T) - 1.0) < 0.01


@pytest.mark.parametrize("beta", [-1.0, 0.5, 1.5])
def test_relative_entropy_matches_monte_carlo(beta):
    prim = MARKET.simulate_primitives(N_PATHS, T, seed=3)
    Z_T = MARKET.radon_nikodym_Qbeta(T, beta, prim)
    mc_entropy = np.mean(Z_T * np.log(Z_T))
    analytic_entropy = MARKET.relative_entropy_closed_form(T, beta)
    # Relative entropy involves a Z*log(Z) integrand with heavier tails
    # than Z alone, hence the looser (but still tight) tolerance.
    assert abs(mc_entropy - analytic_entropy) < 0.05 * max(abs(analytic_entropy), 1.0)


def test_minimal_entropy_is_a_genuine_interior_minimum():
    beta_star = MARKET.minimal_entropy_beta(T)
    entropy_star = MARKET.relative_entropy_closed_form(T, beta_star)
    for beta_other in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        entropy_other = MARKET.relative_entropy_closed_form(T, beta_other)
        assert entropy_star <= entropy_other + 1e-9
    # not stuck at a search bound:
    assert -9.9 < beta_star < 9.9


def test_minimal_entropy_measure_differs_from_beta_zero():
    beta_star = MARKET.minimal_entropy_beta(T)
    # For a market with a genuinely asymmetric jump distribution
    # (jump_mean != 0), the minimal entropy measure should not
    # coincide with the "naive" beta=0 choice.
    assert abs(beta_star - 0.0) > 1e-3

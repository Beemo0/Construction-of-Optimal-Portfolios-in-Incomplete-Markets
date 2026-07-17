"""
Constrained portfolio optimisation in the Black-Scholes model
(Chapter 5, Section on the no-short-sales / leverage example;
Chapter 7, Experiment 4).

For a single risky asset and a box constraint pi in [a, b] on the
proportion of wealth invested in it, the constrained CRRA problem has
an explicit solution:

  - the constrained optimal proportion is the projection of Merton's
    unconstrained proportion onto [a, b];
  - the corresponding dual optimizer is the equivalent martingale
    measure of an *auxiliary* Black-Scholes market, whose market price
    of risk is shifted by nu*/sigma and whose discount rate is
    shifted by delta(nu*) (Chapter 5's support-function penalty) --
    both nu* and its shift are pinned down analytically by requiring
    that the constrained proportion be the auxiliary market's own
    unconstrained Merton optimum.

A subtlety was found and is deliberately kept here as a documented
regression test (see ``tests/test_constrained.py``): shifting only the
market price of risk (equivalently, only ``mu``) reproduces the
*shape* of the constrained wealth process as a function of ``W_T`` but
not its exact level -- the extra discount rate ``delta(nu*)`` is
required for the pathwise identity to hold exactly, exactly as the
discount factor ``exp(-rT)`` was required (and initially forgotten) in
the unconstrained Black-Scholes module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optiport.continuous.black_scholes import BlackScholesMarket
from optiport.constraints.support_function import (
    project_onto_box,
    shadow_price,
    support_function,
)
from optiport.utility import CRRAUtility


@dataclass(frozen=True)
class ConstrainedBlackScholesMarket:
    """A Black-Scholes market together with a box constraint
    pi in [a, b] on the risky-asset proportion (a = -inf and/or
    b = +inf allowed).
    """

    market: BlackScholesMarket
    a: float
    b: float

    def constrained_optimal_pi(self, utility: CRRAUtility) -> float:
        pi_star = self.market.merton_optimal_pi(utility)
        return project_onto_box(pi_star, self.a, self.b)

    def is_binding(self, utility: CRRAUtility, tol: float = 1e-10) -> bool:
        pi_star = self.market.merton_optimal_pi(utility)
        pi_c = self.constrained_optimal_pi(utility)
        return abs(pi_star - pi_c) > tol

    def shadow_price_and_penalty(self, utility: CRRAUtility) -> tuple[float, float]:
        """Return (nu*, delta(nu*)), the shadow price of the
        constraint and its support-function penalty (Chapter 5).
        """
        pi_star = self.market.merton_optimal_pi(utility)
        pi_c = self.constrained_optimal_pi(utility)
        R = utility.relative_risk_aversion
        nu_star = shadow_price(pi_star, pi_c, self.market.sigma, R)
        penalty = support_function(nu_star, self.a, self.b)
        return nu_star, penalty

    def auxiliary_market_and_extra_discount(
        self, utility: CRRAUtility
    ) -> tuple[BlackScholesMarket, float]:
        """The auxiliary Black-Scholes market (shifted market price of
        risk) and the extra discount rate c = delta(nu*), such that
        the constrained proportion pi_c is the auxiliary market's own
        Merton optimum, and the auxiliary state-price density
        H'_t = exp(-(r+c)*t) * dQ_aux/dP correctly prices the
        constrained investor's terminal wealth (see module docstring).
        """
        nu_star, c = self.shadow_price_and_penalty(utility)
        theta_prime = self.market.theta + nu_star / self.market.sigma
        mu_aux = self.market.r + self.market.sigma * theta_prime
        aux_market = BlackScholesMarket(mu=mu_aux, sigma=self.market.sigma, r=self.market.r)
        return aux_market, c

    def real_constrained_terminal_wealth(
        self, x0: float, utility: CRRAUtility, T: float, W_T: np.ndarray
    ) -> np.ndarray:
        """The *actual* terminal wealth of the constrained investor,
        trading the constant proportion pi_c under the REAL market
        dynamics (mu, sigma, r) -- not the auxiliary ones.
        """
        pi_c = self.constrained_optimal_pi(utility)
        drift = (
            self.market.r
            + pi_c * (self.market.mu - self.market.r)
            - 0.5 * pi_c**2 * self.market.sigma**2
        )
        return x0 * np.exp(drift * T + pi_c * self.market.sigma * W_T)

    def dual_terminal_wealth(
        self, x0: float, utility: CRRAUtility, T: float, W_T: np.ndarray
    ) -> np.ndarray:
        """The duality first-order condition X_T_hat = I(y * H'_T),
        H'_T built from the auxiliary market and the extra discount
        rate c = delta(nu*) (Chapter 5), which must coincide *exactly*
        with ``real_constrained_terminal_wealth`` -- the pathwise
        validation of Experiment 4.
        """
        aux_market, c = self.auxiliary_market_and_extra_discount(utility)
        alpha = utility.alpha
        theta_prime = aux_market.theta
        p = alpha / (alpha - 1.0)
        E_Hp = np.exp(-(self.market.r + c) * T * p) * np.exp(
            0.5 * p * (p - 1.0) * theta_prime**2 * T
        )
        y = (x0 / E_Hp) ** (alpha - 1.0)
        H_T = np.exp(-(self.market.r + c) * T) * np.exp(
            -theta_prime * W_T - 0.5 * theta_prime**2 * T
        )
        return utility.I(y * H_T)

    def utility_loss(self, x0: float, utility: CRRAUtility, T: float) -> float:
        """Certainty-equivalent relative utility loss of the
        constrained strategy relative to the unconstrained Merton
        strategy, 1 - CE_constrained / CE_unconstrained, where the
        certainty equivalent CE = U^{-1}(E[U(X_T)]) is used instead of
        the raw expected-utility ratio, since the latter is not
        sign-robust for alpha < 0 (U(x) = x^alpha/alpha is then
        negative-valued, and a naive ratio of expected utilities flips
        sign relative to the intended "fraction of wealth lost").
        Zero exactly when the constraint does not bind.
        """
        alpha = utility.alpha
        pi_star = self.market.merton_optimal_pi(utility)
        pi_c = self.constrained_optimal_pi(utility)
        u_unconstrained = self.market.expected_utility_constant_mix(x0, utility, T, pi_star)
        u_constrained = self.market.expected_utility_constant_mix(x0, utility, T, pi_c)
        if np.isclose(alpha, 0.0):
            ce_unconstrained = np.exp(u_unconstrained)
            ce_constrained = np.exp(u_constrained)
        else:
            ce_unconstrained = (alpha * u_unconstrained) ** (1.0 / alpha)
            ce_constrained = (alpha * u_constrained) ** (1.0 / alpha)
        return 1.0 - ce_constrained / ce_unconstrained

"""
Direct numerical optimisation over the constraint set, as an
algorithmic cross-check of the closed-form projected-Merton solution
of ``constrained_black_scholes.py`` (Chapter 6: numerical algorithms,
Experiment 4).

Rather than relying on the analytic fact that the constrained optimum
is the projection of Merton's proportion onto [a, b], this module
finds the optimal constant proportion by direct 1-D numerical
optimisation of the certainty-equivalent wealth over the box [a, b],
using only ``BlackScholesMarket.expected_utility_constant_mix``
(the closed-form expected utility of an *arbitrary* constant-mix
strategy) as a black box. Agreement between this numerical search and
the closed-form projection is a genuine cross-check: the two
computations share no code path.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from optiport.continuous.black_scholes import BlackScholesMarket
from optiport.utility import CRRAUtility


def certainty_equivalent(
    market: BlackScholesMarket, utility: CRRAUtility, T: float, x0: float, pi: float
) -> float:
    """Certainty-equivalent wealth CE(pi) = U^{-1}(E[U(X_T^pi)]) of the
    constant-mix strategy pi, a strictly increasing transform of the
    expected utility and therefore a numerically better-conditioned
    (and sign-robust, for alpha < 0) objective to optimise directly.
    """
    alpha = utility.alpha
    eu = market.expected_utility_constant_mix(x0, utility, T, pi)
    if np.isclose(alpha, 0.0):
        return float(np.exp(eu))
    return float((alpha * eu) ** (1.0 / alpha))


def numerical_constrained_optimum(
    market: BlackScholesMarket,
    utility: CRRAUtility,
    T: float,
    x0: float,
    a: float,
    b: float,
) -> float:
    """Directly maximise the certainty-equivalent wealth over pi in
    [a, b] by bounded 1-D scalar optimisation (Brent's method), with
    no reference to the closed-form projected-Merton solution.
    """
    lo = a if np.isfinite(a) else -50.0
    hi = b if np.isfinite(b) else 50.0
    res = minimize_scalar(
        lambda pi: -certainty_equivalent(market, utility, T, x0, pi),
        bounds=(lo, hi), method="bounded", options={"xatol": 1e-10},
    )
    return res.x

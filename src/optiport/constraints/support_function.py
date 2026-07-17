"""
Support function of interval (box) portfolio constraints (Chapter 5).

For a single risky asset, the constraint set K = [a, b] (with
a = -infinity and/or b = +infinity allowed, recovering a one-sided or
no constraint) has support function

    delta(nu) = sup_{pi in [a,b]} (-pi * nu)

which takes the explicit form:

    - a, b both finite:  delta(nu) = max(-a*nu, -b*nu)
    - a = -infinity:     delta(nu) = -b*nu if nu <= 0, else +infinity
    - b = +infinity:     delta(nu) = -a*nu if nu >= 0, else +infinity
    - a = -infinity and b = +infinity (no constraint): delta(nu) = 0
      if nu = 0, else +infinity.

This module also provides the constrained proportion (the projection
of the unconstrained Merton proportion onto [a,b]) and the shadow
price nu* that makes it optimal in the auxiliary market of Chapter 5,
used by ``constraints.constrained_black_scholes``.
"""

from __future__ import annotations

import numpy as np


def support_function(nu: float, a: float, b: float) -> float:
    """delta(nu) = sup_{pi in [a,b]} (-pi*nu), for a possibly one- or
    two-sided infinite interval [a, b].
    """
    if a == -np.inf and b == np.inf:
        return 0.0 if nu == 0.0 else np.inf
    if a == -np.inf:
        return -b * nu if nu <= 0.0 else np.inf
    if b == np.inf:
        return -a * nu if nu >= 0.0 else np.inf
    return max(-a * nu, -b * nu)


def project_onto_box(pi_unconstrained: float, a: float, b: float) -> float:
    """Projection of the unconstrained proportion onto [a, b]."""
    return float(np.clip(pi_unconstrained, a, b))


def shadow_price(
    pi_unconstrained: float, pi_constrained: float, sigma: float, relative_risk_aversion: float
) -> float:
    """The shadow price nu* such that the constrained proportion
    pi_constrained is the *unconstrained* Merton optimum of the
    auxiliary market with market price of risk theta + nu*/sigma
    (Chapter 5, auxiliary-market construction). For CRRA utility with
    relative_risk_aversion R = 1 - alpha,
        nu* = (pi_constrained - pi_unconstrained) * sigma^2 * R.
    nu* = 0 exactly when the constraint does not bind.
    """
    return (pi_constrained - pi_unconstrained) * sigma**2 * relative_risk_aversion

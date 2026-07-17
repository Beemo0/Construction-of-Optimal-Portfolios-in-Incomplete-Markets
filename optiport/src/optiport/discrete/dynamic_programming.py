"""
Backward-induction dynamic programming solver for CRRA utility on the
regime-switching binomial market (Chapter 3, Chapter 6, Experiment 2).

For CRRA utility U(x) = x^alpha/alpha (alpha != 0) or U(x) = log(x)
(alpha = 0), the value function is known in closed form up to a
state- and time-dependent constant, thanks to the homogeneity of U:

    alpha != 0:  J_t(x, s) = c_t(s) * x^alpha / alpha
    alpha  = 0:  J_t(x, s) = log(x) + b_t(s)

so the backward recursion reduces to a scalar recursion on (c_t(s))
or (b_t(s)), and, at each date and state, a *one-dimensional* concave
maximisation over the risky-asset proportion pi. This is exploited
here to avoid discretising the wealth variable altogether: the
algorithm below is exact (up to the scalar optimisation's numerical
tolerance), not merely a discretised approximation of the general
Chapter 6 grid algorithm.

The myopic policy -- the proportion that would be optimal if the
current period were the last one, ignoring all continuation value --
is provided separately and is the object of comparison in
Experiment 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from optiport.discrete.regime_market import RegimeSwitchingBinomialMarket
from optiport.utility import CRRAUtility


def _one_period_objective(pi: float, u: float, d: float, p_up: float, alpha: float) -> float:
    """E[ (1 + pi*(R-1))^alpha ] for R in {u, d} under the physical
    measure, i.e. the (unnormalised) one-period CRRA objective. For
    alpha = 0 this is replaced by the log objective by the caller.
    """
    return p_up * (1.0 + pi * (u - 1.0)) ** alpha + (1.0 - p_up) * (
        1.0 + pi * (d - 1.0)
    ) ** alpha


def _one_period_log_objective(pi: float, u: float, d: float, p_up: float) -> float:
    return p_up * np.log(1.0 + pi * (u - 1.0)) + (1.0 - p_up) * np.log(
        1.0 + pi * (d - 1.0)
    )


def _optimize_pi(objective, bounds: tuple[float, float], maximize: bool = True) -> tuple[float, float]:
    """Optimise a scalar concave (if maximizing) or convex (if
    minimizing) objective over ``bounds`` with a bounded 1-D solver
    (Brent's method restricted to the bounded interval).

    ``maximize`` must be set to ``False`` whenever the term being
    optimised is multiplied, in the surrounding value function, by a
    *negative* constant -- which is exactly what happens for CRRA
    utility with alpha < 0, since J_t(x,s) = c_t(s) * x^alpha/alpha
    and x^alpha/alpha < 0 in that case: maximising J_t(x,s) for fixed
    x then means *minimising* the bracketed term that defines c_t(s).
    Getting this sign wrong silently produces a well-defined but
    economically meaningless optimiser (see the regression test
    ``test_alpha_negative_sign_convention`` in the test suite).
    """
    sign = -1.0 if maximize else 1.0
    res = minimize_scalar(
        lambda pi: sign * objective(pi), bounds=bounds, method="bounded",
        options={"xatol": 1e-10},
    )
    return res.x, objective(res.x)


@dataclass
class BackwardInductionResult:
    """Optimal proportions pi_t(s) and value-function constants,
    for every date t = 0, ..., T-1 and every regime s.
    """

    pi: np.ndarray  # shape (T, n_states)
    value_constants: np.ndarray  # shape (T+1, n_states): c_t(s) or b_t(s)
    is_log: bool


def solve_backward_induction(
    market: RegimeSwitchingBinomialMarket, utility: CRRAUtility
) -> BackwardInductionResult:
    """Exact backward-induction solution of the dynamically optimal
    CRRA portfolio problem on ``market`` (Chapter 3 theorem, applied
    node-by-node in the regime-switching setting).
    """
    T, S = market.n_periods, market.n_states
    alpha = utility.alpha
    is_log = np.isclose(alpha, 0.0)

    pi = np.zeros((T, S))
    const = np.zeros((T + 1, S))  # const[T] = 1 (power) or 0 (log), by construction
    if not is_log:
        const[T, :] = 1.0  # J_T(x,s) = U(x) = x^alpha/alpha => c_T(s) = 1

    for t in range(T - 1, -1, -1):
        for s in range(S):
            bounds = market.feasible_pi_bounds(s)
            if is_log:
                # Myopic term (does not depend on continuation values):
                pi_ts, _ = _optimize_pi(
                    lambda p: _one_period_log_objective(
                        p, market.u[s], market.d[s], market.p_up[s]
                    ),
                    bounds,
                )
                cont_up = market.trans_up[s] @ const[t + 1]
                cont_down = market.trans_down[s] @ const[t + 1]
                myopic_value = _one_period_log_objective(
                    pi_ts, market.u[s], market.d[s], market.p_up[s]
                )
                expected_continuation = (
                    market.p_up[s] * cont_up + (1.0 - market.p_up[s]) * cont_down
                )
                const[t, s] = myopic_value + expected_continuation
            else:
                a_up = market.trans_up[s] @ const[t + 1]
                a_down = market.trans_down[s] @ const[t + 1]

                def objective(p, u=market.u[s], d=market.d[s], pu=market.p_up[s],
                              au=a_up, ad=a_down, al=alpha):
                    return pu * au * (1.0 + p * (u - 1.0)) ** al + (
                        1.0 - pu
                    ) * ad * (1.0 + p * (d - 1.0)) ** al

                pi_ts, val = _optimize_pi(objective, bounds, maximize=(alpha > 0))
                const[t, s] = val

            pi[t, s] = pi_ts

    return BackwardInductionResult(pi=pi, value_constants=const, is_log=is_log)


def myopic_policy(
    market: RegimeSwitchingBinomialMarket, utility: CRRAUtility
) -> np.ndarray:
    """The proportion pi_myopic(s) that maximises the *one-period*
    CRRA objective in each regime s, ignoring all continuation value
    -- i.e. the policy that would be optimal if the current period
    were the last one. Independent of t by construction.
    """
    alpha = utility.alpha
    is_log = np.isclose(alpha, 0.0)
    S = market.n_states
    pi_m = np.zeros(S)
    for s in range(S):
        bounds = market.feasible_pi_bounds(s)
        if is_log:
            pi_m[s], _ = _optimize_pi(
                lambda p, u=market.u[s], d=market.d[s], pu=market.p_up[s]:
                _one_period_log_objective(p, u, d, pu),
                bounds,
            )
        else:
            pi_m[s], _ = _optimize_pi(
                lambda p, u=market.u[s], d=market.d[s], pu=market.p_up[s], al=alpha:
                _one_period_objective(p, u, d, pu, al),
                bounds,
                maximize=(alpha > 0),
            )
    return pi_m

"""
Regime-switching binomial market.

Extends the plain i.i.d. binomial market of ``market.py`` by letting
the up/down return parameters depend on a discrete regime (state)
that evolves stochastically over time. Crucially, the regime
transition is allowed to depend on the *realised* branch (up or
down move) of the risky asset at each date, i.e. the state shocks are
correlated with the return shocks.

This correlation is exactly what generates a non-trivial
intertemporal hedging demand for CRRA investors with relative risk
aversion different from 1 (Merton's classical ICAPM insight,
transposed here to discrete time): if the state transition were
independent of the realised branch, the optimal policy would remain
myopic for every CRRA utility, exactly as it is for i.i.d. returns
(Samuelson's discrete-time myopic-policy theorem). It is only the
branch-dependence of the transition that breaks this and makes the
theoretical distinction of Chapter 3/7 -- the logarithmic investor is
*always* myopic, other CRRA investors are myopic only in the absence
of predictable, correlated investment opportunities -- numerically
visible.

This model is used, in place of the simpler i.i.d. binomial market,
to correct and sharpen Experiment 2 of Chapter 7: the original
specification (myopic vs. optimal diverging for every non-log CRRA
under i.i.d. returns) is not in fact correct -- i.i.d. single-asset
CRRA problems are always myopic (Samuelson 1969) -- and the genuinely
interesting, and correct, experiment requires this regime-switching
structure instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RegimeSwitchingBinomialMarket:
    """A single risky asset whose up/down returns and up-move
    probability depend on a discrete regime s in {0, ..., n_states-1},
    with regime transitions depending on the realised branch.

    Parameters
    ----------
    n_periods : number of trading periods T.
    u, d : arrays of shape (n_states,), gross returns in each regime;
        d[s] < 1 < u[s] required for every s (no arbitrage state by
        state).
    p_up : array of shape (n_states,), physical probability of an
        up-move in each regime.
    trans_up, trans_down : arrays of shape (n_states, n_states),
        trans_up[s, s'] = P(next regime = s' | current regime = s,
        realised move = up), and similarly for trans_down. Each row
        must sum to 1.
    """

    n_periods: int
    u: np.ndarray
    d: np.ndarray
    p_up: np.ndarray
    trans_up: np.ndarray
    trans_down: np.ndarray

    def __post_init__(self):
        u, d, p_up = np.asarray(self.u), np.asarray(self.d), np.asarray(self.p_up)
        n = len(u)
        if not (len(d) == len(p_up) == n):
            raise ValueError("u, d, p_up must have the same length (n_states).")
        if np.any(d >= 1.0) or np.any(u <= 1.0):
            raise ValueError("No-arbitrage requires d[s] < 1 < u[s] for every state.")
        if np.any((p_up <= 0) | (p_up >= 1)):
            raise ValueError("p_up must be in (0,1) for every state.")
        for name, T in (("trans_up", self.trans_up), ("trans_down", self.trans_down)):
            T = np.asarray(T)
            if T.shape != (n, n):
                raise ValueError(f"{name} must have shape (n_states, n_states).")
            if not np.allclose(T.sum(axis=1), 1.0):
                raise ValueError(f"Each row of {name} must sum to 1.")

    @property
    def n_states(self) -> int:
        return len(np.asarray(self.u))

    def feasible_pi_bounds(self, state: int, eps: float = 1e-8) -> tuple[float, float]:
        """Bounds on the risky-asset proportion pi that keep terminal
        wealth strictly positive on both branches in ``state``:
        1 + pi*(u-1) > 0 and 1 + pi*(d-1) > 0.
        """
        lo = -1.0 / (self.u[state] - 1.0) + eps
        hi = 1.0 / (1.0 - self.d[state]) - eps
        return lo, hi


def iid_market_as_regime_switching(
    n_periods: int, u: float, d: float, p_up: float
) -> RegimeSwitchingBinomialMarket:
    """Degenerate single-state regime-switching market, equivalent to
    the plain i.i.d. binomial market -- used as the baseline case in
    which every CRRA utility is known to be myopic (Samuelson 1969).
    """
    return RegimeSwitchingBinomialMarket(
        n_periods=n_periods,
        u=np.array([u]),
        d=np.array([d]),
        p_up=np.array([p_up]),
        trans_up=np.array([[1.0]]),
        trans_down=np.array([[1.0]]),
    )

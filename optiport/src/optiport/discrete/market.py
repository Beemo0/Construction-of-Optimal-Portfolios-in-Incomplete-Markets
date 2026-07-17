"""
Discrete-time, multi-period binomial market model.

Implements the market model of Chapter 3: a single risky asset over
T periods, one riskless asset taken as numeraire (so discounted
prices are used throughout), on a recombining binomial tree. The
market is arbitrage-free and, in general, incomplete only in the sense
that with more than one risky asset several equivalent martingale
measures may coexist; the single-asset binomial tree used here is
in fact complete (a single up/down factor pins down a unique
martingale measure), and is used primarily as the tractable test bed
for the dynamic-programming algorithm of Chapter 6 and for the
myopic-vs-optimal comparison of Experiment 2 (Chapter 7).

A genuinely incomplete (multi-asset or jump) discrete model is
implemented separately in optiport.continuous.jump_diffusion, reused
in its discretised form for Experiment 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinomialMarket:
    """Recombining binomial tree for a single discounted risky asset.

    Parameters
    ----------
    n_periods : number of trading periods T.
    u, d : gross up and down discounted returns per period, d < 1 < u
        required for absence of arbitrage (Chapter 3, FTAP).
    p_physical : physical (real-world) probability of an up-move,
        used only to simulate sample paths, never in the valuation
        or optimisation itself.
    """

    n_periods: int
    u: float
    d: float
    p_physical: float = 0.5

    def __post_init__(self):
        if not (self.d < 1.0 < self.u):
            raise ValueError(
                "No-arbitrage requires d < 1 < u (Chapter 3, discrete FTAP)."
            )
        if not (0.0 < self.p_physical < 1.0):
            raise ValueError("p_physical must be in (0,1).")

    @property
    def q_martingale(self) -> float:
        """Unique martingale-measure probability of an up-move,
        q = (1 - d) / (u - d), obtained by solving
        q*u + (1-q)*d = 1 (discounted price is a martingale under Q).
        """
        return (1.0 - self.d) / (self.u - self.d)

    def price_tree(self, s0: float = 1.0) -> list[np.ndarray]:
        """Return the discounted price tree as a list of arrays,
        tree[t] holding the 2**t... actually t+1 recombining node
        values S_t(k), k = 0..t (k = number of down-moves).
        """
        tree = []
        for t in range(self.n_periods + 1):
            k = np.arange(t + 1)
            tree.append(s0 * self.u ** (t - k) * self.d**k)
        return tree

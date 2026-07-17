"""
Utility functions for expected utility maximisation.

Implements the utility function classes used throughout the thesis
(Chapters 3-6): CRRA (power / logarithmic) and CARA (exponential)
utilities, together with the machinery required by the duality
theorems -- the marginal utility inverse I = (U')^{-1} and the convex
conjugate (Legendre-Fenchel transform)

    V_tilde(y) = sup_{x in dom(U)} ( U(x) - x*y ),   y > 0.

All three quantities (U, I, V_tilde) are implemented analytically
where closed forms are available (they are, for both CRRA and CARA),
so that no numerical inversion or optimisation is needed elsewhere in
the codebase when evaluating the first-order condition
    X_T_hat = I(y * dQ_hat/dP).

References
----------
Chapter 3 (discrete time), Chapter 4 (continuous time), Section on the
logarithmic / exponential special cases in both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class UtilityFunction(ABC):
    """Abstract base class for strictly increasing, strictly concave
    utility functions satisfying the Inada conditions on their domain,
    as required by Chapters 3-4 of the thesis.
    """

    #: domain of U: "positive" for dom(U) = (0, infinity) (Case 1),
    #: "real" for dom(U) = R (Case 2). See Chapter 4, Section on the
    #: two classes of utility functions.
    domain: str

    @abstractmethod
    def U(self, x: np.ndarray) -> np.ndarray:
        """Evaluate U(x)."""

    @abstractmethod
    def U_prime(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the marginal utility U'(x)."""

    @abstractmethod
    def I(self, y: np.ndarray) -> np.ndarray:  # noqa: E743 (matches thesis notation)
        """Inverse marginal utility I(y) = (U')^{-1}(y), y > 0."""

    @abstractmethod
    def V_tilde(self, y: np.ndarray) -> np.ndarray:
        """Convex conjugate V_tilde(y) = sup_x (U(x) - x*y)."""

    def check_first_order_condition(self, y: np.ndarray, atol: float = 1e-8) -> bool:
        """Sanity check: U'(I(y)) == y for all y > 0.

        This is the identity underlying the first-order condition
        X_T_hat = I(y dQ_hat/dP) used throughout Chapters 3-6; every
        concrete utility class is tested against it (see
        tests/test_utility.py).
        """
        y = np.asarray(y, dtype=float)
        return bool(np.allclose(self.U_prime(self.I(y)), y, atol=atol))


class CRRAUtility(UtilityFunction):
    """Power utility U(x) = x^alpha / alpha for alpha < 1, alpha != 0,
    with the logarithmic utility U(x) = log(x) as the limiting case
    alpha -> 0, requested explicitly via ``alpha=0.0``.

    dom(U) = (0, infinity) (Case 1 of Chapter 4). Relative risk
    aversion is constant and equal to ``1 - alpha``.
    """

    domain = "positive"

    def __init__(self, alpha: float):
        if alpha >= 1:
            raise ValueError("CRRA utility requires alpha < 1.")
        self.alpha = float(alpha)
        self._is_log = np.isclose(self.alpha, 0.0)

    def U(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0):
            raise ValueError("CRRA utility is only defined on (0, infinity).")
        if self._is_log:
            return np.log(x)
        return x**self.alpha / self.alpha

    def U_prime(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._is_log:
            return 1.0 / x
        return x ** (self.alpha - 1.0)

    def I(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("I(y) is only defined for y > 0.")
        if self._is_log:
            return 1.0 / y
        return y ** (1.0 / (self.alpha - 1.0))

    def V_tilde(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self._is_log:
            return -np.log(y) - 1.0
        # V_tilde(y) = U(I(y)) - y*I(y), the standard Legendre transform
        # identity, specialised analytically for the power case:
        # V_tilde(y) = (1 - alpha)/alpha * y^(alpha/(alpha-1))
        beta = self.alpha / (self.alpha - 1.0)
        return (1.0 - self.alpha) / self.alpha * y**beta

    @property
    def relative_risk_aversion(self) -> float:
        return 1.0 - self.alpha


class ExponentialUtility(UtilityFunction):
    """Exponential (CARA) utility U(x) = -exp(-gamma*x)/gamma, gamma > 0.

    dom(U) = R (Case 2 of Chapter 4). This is the utility function for
    which the dual optimizer is the minimal entropy martingale measure
    (Chapter 4, Section on the exponential case and minimal entropy).
    """

    domain = "real"

    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError("Risk-aversion parameter gamma must be positive.")
        self.gamma = float(gamma)

    def U(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return -np.exp(-self.gamma * x) / self.gamma

    def U_prime(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(-self.gamma * x)

    def I(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("I(y) is only defined for y > 0.")
        return -np.log(y) / self.gamma

    def V_tilde(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("V_tilde(y) is only defined for y > 0.")
        return y * (np.log(y) - 1.0) / self.gamma

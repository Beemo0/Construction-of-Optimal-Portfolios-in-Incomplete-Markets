"""Tests for optiport.utility.

Validates the analytical CRRA and exponential utility implementations
against:
  1. the defining first-order identity U'(I(y)) = y (Chapters 3-4);
  2. numerical concavity of U;
  3. the Legendre-transform identity V_tilde(y) = U(I(y)) - y*I(y),
     checking the closed-form V_tilde against its definition.
"""

import numpy as np
import pytest

from optiport.utility import CRRAUtility, ExponentialUtility

Y_GRID = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
X_GRID = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.0, 0.5, 0.9])
def test_crra_first_order_condition(alpha):
    u = CRRAUtility(alpha)
    assert u.check_first_order_condition(Y_GRID)


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.0, 0.5, 0.9])
def test_crra_legendre_transform_identity(alpha):
    u = CRRAUtility(alpha)
    x_star = u.I(Y_GRID)
    expected = u.U(x_star) - Y_GRID * x_star
    assert np.allclose(u.V_tilde(Y_GRID), expected, atol=1e-8)


@pytest.mark.parametrize("alpha", [-2.0, -0.5, 0.5, 0.9])
def test_crra_concavity(alpha):
    u = CRRAUtility(alpha)
    h = 1e-4
    second_diff = (u.U(X_GRID + h) - 2 * u.U(X_GRID) + u.U(X_GRID - h)) / h**2
    assert np.all(second_diff < 0)


def test_crra_rejects_alpha_geq_one():
    with pytest.raises(ValueError):
        CRRAUtility(1.0)


def test_crra_rejects_nonpositive_wealth():
    u = CRRAUtility(0.5)
    with pytest.raises(ValueError):
        u.U(np.array([-1.0, 1.0]))


def test_exponential_first_order_condition():
    u = ExponentialUtility(gamma=2.0)
    assert u.check_first_order_condition(Y_GRID)


def test_exponential_legendre_transform_identity():
    u = ExponentialUtility(gamma=2.0)
    x_star = u.I(Y_GRID)
    expected = u.U(x_star) - Y_GRID * x_star
    assert np.allclose(u.V_tilde(Y_GRID), expected, atol=1e-8)


def test_exponential_concavity():
    u = ExponentialUtility(gamma=1.5)
    h = 1e-4
    second_diff = (u.U(X_GRID + h) - 2 * u.U(X_GRID) + u.U(X_GRID - h)) / h**2
    assert np.all(second_diff < 0)

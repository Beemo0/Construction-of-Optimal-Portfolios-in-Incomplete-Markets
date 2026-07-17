import numpy as np
import pytest

from optiport.continuous.black_scholes import BlackScholesMarket
from optiport.constraints.constrained_black_scholes import ConstrainedBlackScholesMarket
from optiport.constraints.dual_gradient import numerical_constrained_optimum
from optiport.utility import CRRAUtility

MARKET = BlackScholesMarket(mu=0.10, sigma=0.20, r=0.02)
T, X0 = 1.0, 100.0


@pytest.mark.parametrize(
    "a,b",
    [(-np.inf, 0.5), (0.0, np.inf), (0.0, 2.0), (-10.0, 10.0)],
)
@pytest.mark.parametrize("alpha", [-2.0, -1.0, -0.5, 0.0, 0.5])
def test_numerical_optimum_matches_closed_form_projection(a, b, alpha):
    utility = CRRAUtility(alpha)
    cmarket = ConstrainedBlackScholesMarket(market=MARKET, a=a, b=b)
    pi_closed_form = cmarket.constrained_optimal_pi(utility)
    pi_numerical = numerical_constrained_optimum(MARKET, utility, T, X0, a, b)
    assert np.isclose(pi_closed_form, pi_numerical, atol=1e-4)

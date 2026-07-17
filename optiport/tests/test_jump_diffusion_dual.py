import numpy as np

from optiport.continuous.jump_diffusion import MertonJumpDiffusionMarket

MARKET = MertonJumpDiffusionMarket(
    mu=0.10, sigma=0.15, r=0.02, lam=0.8, jump_mean=-0.05, jump_std=0.10
)
T = 1.0


def test_dual_objective_minimised_at_minimal_entropy_beta():
    beta_star = MARKET.minimal_entropy_beta(T)
    prim = MARKET.simulate_primitives(2_000_000, T, seed=11)
    y, gamma = 1.0, 1.5
    grid = np.linspace(beta_star - 2.0, beta_star + 2.0, 9)
    objectives = [
        MARKET.dual_objective_exponential_mc(T, b, y, gamma, prim) for b in grid
    ]
    argmin_beta = grid[int(np.argmin(objectives))]
    # The grid minimiser should be within one grid step of beta_star.
    assert abs(argmin_beta - beta_star) <= (grid[1] - grid[0]) + 1e-6

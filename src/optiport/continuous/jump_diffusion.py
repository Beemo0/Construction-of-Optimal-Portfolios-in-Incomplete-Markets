"""
Incomplete jump-diffusion market and the minimal entropy martingale
measure (Chapter 4, Section on the exponential case and minimal
entropy; Chapter 7, Experiment 3).

Model (Merton 1976 jump-diffusion, under the physical measure P):

    S_t = S_0 * exp( (mu - sigma^2/2 - lambda*kappa)*t + sigma*W_t + J_t )

where W is a P-Brownian motion, J_t = sum_{i=1}^{N_t} Z_i is a
compound Poisson process with intensity lambda and i.i.d. jump sizes
Z_i ~ N(jump_mean, jump_std^2), and
kappa = E[e^Z] - 1 = exp(jump_mean + jump_std^2/2) - 1 is the
compensator that makes E[S_t] = S_0 * exp(mu*t).

This market has two independent sources of randomness (the Brownian
motion and the compound Poisson jumps) but only one tradable risky
asset: it is genuinely incomplete, and the family of equivalent
martingale measures M_e(S) is infinite-dimensional. Rather than the
full family, this module works with the tractable one-parameter
Esscher/Girsanov sub-family {Q_beta, beta in R}, obtained by
simultaneously shifting the Brownian motion by theta and exponentially
tilting the jump-size distribution by beta, with theta = theta(beta)
determined by the martingale condition. This sub-family is rich enough
to contain the minimal entropy martingale measure (found here by a
one-dimensional numerical search over beta) and to illustrate,
concretely, that an arbitrary member of M_e(S) is generally not the
utility-maximiser's dual optimizer identified by the duality theorem
of Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class MertonJumpDiffusionMarket:
    """Merton (1976) jump-diffusion market under the physical measure P."""

    mu: float
    sigma: float
    r: float
    lam: float          # jump intensity (physical)
    jump_mean: float     # mean of log-jump size Z ~ N(jump_mean, jump_std^2)
    jump_std: float

    def __post_init__(self):
        if self.sigma <= 0 or self.jump_std <= 0 or self.lam < 0:
            raise ValueError("sigma, jump_std must be positive and lam nonnegative.")

    @property
    def kappa(self) -> float:
        """P-compensator kappa = E[e^Z] - 1."""
        return np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0

    def moment_generating_jump(self, beta: float) -> float:
        """M(beta) = E^P[exp(beta*Z)] = exp(beta*jump_mean + 0.5*beta^2*jump_std^2)."""
        return np.exp(beta * self.jump_mean + 0.5 * beta**2 * self.jump_std**2)

    def tilted_jump_params(self, beta: float) -> tuple[float, float]:
        """Esscher-tilted jump intensity and mean under Q_beta:
        lambda_Q(beta) = lambda * M(beta),
        jump_mean_Q(beta) = jump_mean + beta*jump_std^2
        (the tilted jump-size variance is unchanged, a standard property
        of exponential tilting of a Gaussian).
        """
        lam_Q = self.lam * self.moment_generating_jump(beta)
        mean_Q = self.jump_mean + beta * self.jump_std**2
        return lam_Q, mean_Q

    def theta_from_beta(self, beta: float) -> float:
        """Solve the martingale condition
        mu - lambda*kappa + sigma*theta + lambda_Q(beta)*kappa_Q(beta) = r
        for theta, given beta (Section above / Chapter 4).
        """
        lam_Q, mean_Q = self.tilted_jump_params(beta)
        kappa_Q = np.exp(mean_Q + 0.5 * self.jump_std**2) - 1.0
        return (self.r - self.mu + self.lam * self.kappa - lam_Q * kappa_Q) / self.sigma

    def simulate_primitives(
        self, n_paths: int, T: float, seed: int | None = None
    ) -> dict:
        """Simulate the terminal value of the driving P-primitives:
        W_T (Brownian), N_T (jump count), J_T (sum of P-jump sizes) --
        sufficient statistics for everything else in this module, since
        the model is a Levy process (independent, stationary increments).
        """
        rng = np.random.default_rng(seed)
        W_T = rng.normal(0.0, np.sqrt(T), n_paths)
        N_T = rng.poisson(self.lam * T, n_paths)
        # sum of N_T i.i.d. N(jump_mean, jump_std^2) draws, vectorised:
        max_n = max(N_T.max(), 1)
        all_jumps = rng.normal(self.jump_mean, self.jump_std, size=(n_paths, max_n))
        mask = np.arange(max_n)[None, :] < N_T[:, None]
        J_T = np.sum(all_jumps * mask, axis=1)
        return {"W_T": W_T, "N_T": N_T, "J_T": J_T}

    def terminal_price(self, S0: float, T: float, primitives: dict) -> np.ndarray:
        """S_T under P, from simulated primitives."""
        drift = self.mu - 0.5 * self.sigma**2 - self.lam * self.kappa
        return S0 * np.exp(
            drift * T + self.sigma * primitives["W_T"] + primitives["J_T"]
        )

    def radon_nikodym_Qbeta(self, T: float, beta: float, primitives: dict) -> np.ndarray:
        """dQ_beta/dP, as an explicit function of the P-primitives
        (W_T, J_T), for the Esscher/Girsanov sub-family defined above.
        """
        theta = self.theta_from_beta(beta)
        M_beta = self.moment_generating_jump(beta)
        W_T, J_T = primitives["W_T"], primitives["J_T"]
        girsanov = np.exp(theta * W_T - 0.5 * theta**2 * T)
        esscher = np.exp(beta * J_T - self.lam * T * (M_beta - 1.0))
        return girsanov * esscher

    def relative_entropy_closed_form(self, T: float, beta: float) -> float:
        """D(Q_beta || P) = E^{Q_beta}[log dQ_beta/dP], in closed form
        (Chapter 4, exponential/entropy section, specialised to this
        one-parameter sub-family).
        """
        theta = self.theta_from_beta(beta)
        lam_Q, mean_Q = self.tilted_jump_params(beta)
        M_beta = self.moment_generating_jump(beta)
        # E^Q[W_T] = theta*T ; E^Q[J_T] = lambda_Q(beta) * mean_Q(beta) * T
        return (
            0.5 * theta**2 * T
            + beta * lam_Q * mean_Q * T
            - self.lam * T * (M_beta - 1.0)
        )

    def minimal_entropy_beta(self, T: float, bounds: tuple[float, float] = (-10.0, 10.0)) -> float:
        """Numerically minimise the closed-form relative entropy over
        beta, within the Esscher sub-family: the minimal entropy
        martingale measure restricted to this tractable curve of M_e(S).
        """
        res = minimize_scalar(
            lambda b: self.relative_entropy_closed_form(T, b),
            bounds=bounds, method="bounded", options={"xatol": 1e-10},
        )
        return res.x

    def dual_objective_exponential_mc(
        self, T: float, beta: float, y: float, gamma: float, primitives: dict
    ) -> float:
        """Monte Carlo estimate of the exponential-utility dual
        objective E^P[V_tilde(y * H_beta)], H_beta = exp(-r*T) * Z_beta,
        V_tilde(z) = z*(log z - 1)/gamma (Chapter 4). For any *fixed*
        y, this objective is, up to a beta-independent affine
        transform, exactly proportional to the relative entropy
        D(Q_beta||P): minimising it over beta therefore recovers the
        same beta* as ``minimal_entropy_beta``, which is precisely the
        content of the exponential-utility duality result of Chapter 4
        (the dual optimizer is the minimal entropy martingale measure).
        This method provides an independent Monte Carlo check of that
        analytic equivalence.
        """
        Z_T = self.radon_nikodym_Qbeta(T, beta, primitives)
        H_T = np.exp(-self.r * T) * Z_T
        z = y * H_T
        V_tilde = z * (np.log(z) - 1.0) / gamma
        return float(np.mean(V_tilde))

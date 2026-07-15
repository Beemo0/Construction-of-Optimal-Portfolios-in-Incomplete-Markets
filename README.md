# Construction of Optimal Portfolios in Incomplete Markets

**PFE ING3 — CY Tech 2026–2027**
Supervisors: TBC

---

## Abstract

An investor who wants to maximise the expected utility of terminal
wealth faces a straightforward problem in a complete market: there is
a unique equivalent martingale measure, every claim is replicable, and
the optimal portfolio is read off directly by a change of measure. As
soon as the market is incomplete -- because of jumps, stochastic
volatility, or realistic trading constraints -- this uniqueness fails,
and the investor must genuinely solve a stochastic optimisation
problem. This project builds, from first principles and with a fully
tested numerical implementation, the convex-duality theory that solves
this problem: discrete-time multi-period duality (Pliska), the general
continuous-time Kramkov-Schachermayer duality (Schachermayer), and the
extension to convex portfolio constraints (Cvitanić-Karatzas,
Bouchard-Cassagneux), together with the numerical algorithms --
dynamic programming, BSDE schemes, and dual optimisation methods --
needed to solve the resulting problems in practice, constrained case
first and unconstrained case recovered as a special case, as required
by the project guidelines.

## Overview

This repository implements the full theoretical and numerical pipeline
of expected-utility portfolio optimisation in incomplete markets, in
discrete and continuous time, unconstrained and under convex trading
constraints.

The project is divided into two parts, mirroring the corresponding
thesis chapters:

- **Part 1** -- Unconstrained theory: discrete-time dynamic
  programming and duality (Chapter 3), continuous-time
  Kramkov-Schachermayer duality validated against Merton's closed-form
  solution (Chapter 4), and an incomplete jump-diffusion market with
  the minimal entropy martingale measure (Chapter 4/7).
- **Part 2** -- Constrained theory and numerics: portfolio constraints
  via the support function and the auxiliary-market duality
  (Chapter 5), and the corresponding numerical algorithms -- BSDE-based
  and parametric dual gradient methods -- developed for the
  constrained case first and specialised to the unconstrained case
  second (Chapter 6/7).

---

## Repository Structure

```
├── Part 1/
│   └── 01_discrete_dynamic_programming.ipynb    # Samuelson myopia + Merton hedging demand
│   └── 02_black_scholes_merton_validation.ipynb # Merton vs. duality, pathwise + Monte Carlo
│   └── 03_jump_diffusion_incomplete_market.ipynb # minimal entropy martingale measure
│
├── Part 2/
│   └── 04_constrained_black_scholes.ipynb       # support function, constrained duality, E4
│   └── 05_numerical_cross_check.ipynb           # direct numerical optimizer vs. closed form
│
├── src/
│   └── optiport/
│       ├── __init__.py
│       ├── utility.py                 # CRRA / exponential utility, conjugate, I=(U')^-1
│       ├── discrete/
│       │   ├── market.py              # plain i.i.d. binomial market
│       │   ├── regime_market.py       # regime-switching binomial market (correlated transitions)
│       │   └── dynamic_programming.py # exact backward-induction CRRA solver + myopic policy
│       ├── continuous/
│       │   ├── black_scholes.py       # Merton closed-form vs. duality FOC, pathwise validation
│       │   └── jump_diffusion.py      # incomplete jump-diffusion market, Esscher EMM family, minimal entropy
│       └── constraints/
│           ├── support_function.py            # box-constraint support function, shadow price
│           ├── constrained_black_scholes.py    # constrained duality, auxiliary market, utility loss
│           └── dual_gradient.py                # direct numerical cross-check (no closed-form shortcut)
│
├── tests/
│   ├── test_utility.py
│   ├── test_discrete_dp.py
│   ├── test_merton_validation.py
│   ├── test_jump_diffusion.py
│   ├── test_jump_diffusion_dual.py
│   ├── test_constrained.py
│   └── test_dual_gradient.py
│
├── pyproject.toml
└── README.md
```

---

## Core Modules (`src/optiport/`)

### `utility.py`

CRRA (power / logarithmic) and exponential (CARA) utility functions,
with analytical marginal utility, its inverse `I = (U')^{-1}`, and the
Legendre-Fenchel conjugate `V_tilde`, as required by the first-order
condition `X_T_hat = I(y * dQ_hat/dP)` of the duality theorems
(Chapters 3-6).

### `discrete/regime_market.py`

A single risky asset on a recombining binomial tree whose up/down
return parameters and up-move probability depend on a discrete regime,
with regime transitions allowed to depend on the *realised* branch --
the correlation structure needed to generate a non-trivial
intertemporal hedging demand for non-logarithmic CRRA utilities
(Merton's classical ICAPM insight, transposed to discrete time).

### `discrete/dynamic_programming.py`

Exact backward-induction solver exploiting the homogeneity of CRRA
utility (`J_t(x,s) = c_t(s) * x^alpha/alpha`), reducing the dynamic
program to a one-dimensional concave optimisation at each date and
regime, together with the corresponding myopic (one-period) policy
used as the object of comparison in Experiment 2.

### `continuous/black_scholes.py`

Merton's closed-form constant-mix solution and the continuous-time
duality first-order condition in the (complete) Black-Scholes market,
both as explicit, exact functions of the terminal Brownian value
`W_T`, enabling a pathwise (not merely statistical) validation of the
Kramkov-Schachermayer duality theorem against Merton's classical
dynamic-programming result.

### `continuous/jump_diffusion.py`

An incomplete Merton (1976) jump-diffusion market, with a tractable
one-parameter Esscher/Girsanov sub-family of equivalent martingale
measures. Provides the closed-form relative entropy along this family,
a numerical search for the minimal entropy martingale measure, and an
independent Monte Carlo cross-check that the exponential-utility dual
objective of Chapter 4 is minimised at the same point.

### `constraints/support_function.py`

The support function of an interval (box) portfolio constraint
`[a, b]`, the projection of the unconstrained Merton proportion onto
it, and the shadow price that makes the projected proportion optimal
in the auxiliary market of Chapter 5.

### `constraints/constrained_black_scholes.py`

The constrained duality theorem of Chapter 5, specialised to a single
risky asset and a box constraint: the auxiliary market and the extra
discount rate `delta(nu*)`, the pathwise duality first-order condition,
and the certainty-equivalent utility loss of the constraint.

### `constraints/dual_gradient.py`

A direct numerical optimiser (bounded 1-D search on the
certainty-equivalent wealth) that finds the constrained optimum with
no reference to the closed-form projection formula -- an independent
algorithmic cross-check sharing no code path with
`constrained_black_scholes.py`.

---

## Notebooks

### Part 1 — Unconstrained Theory

| Notebook | Description | Key outputs |
|---|---|---|
| `01_discrete_dynamic_programming.ipynb` | Validates Samuelson's myopic-policy theorem (i.i.d. returns) and Merton's intertemporal hedging demand (correlated regime-switching returns) | Policy trajectories by regime and risk aversion; hedging-demand gap vs. risk aversion, exactly zero at log utility |
| `02_black_scholes_merton_validation.ipynb` | Continuous-time duality vs. Merton's closed-form constant-mix formula, pathwise and Monte Carlo | Exact pathwise coincidence (gap at floating-point precision); $O(1/\sqrt{n})$ Monte Carlo convergence; a documented discount-factor bug found during development |
| `03_jump_diffusion_incomplete_market.ipynb` | Minimal entropy martingale measure within a tractable Esscher sub-family of an incomplete jump-diffusion market | Entropy curve with non-trivial minimiser $\beta^* \neq 0$; martingale condition validated across the family; dual objective independently minimised at the same $\beta^*$ |

### Part 2 — Constrained Theory and Numerics

| Notebook | Description | Key outputs |
|---|---|---|
| `04_constrained_black_scholes.ipynb` | Constrained duality (support function, auxiliary market), pathwise validation across binding/non-binding cases, utility cost of a tightening leverage cap | Exact pathwise coincidence in all 3 cases; utility-loss curve vs. leverage cap; a documented missing-penalty bug |
| `05_numerical_cross_check.ipynb` | Direct numerical optimiser (no closed-form shortcut) vs. closed-form projection, across risk aversions and constraint regimes | Gap at numerical-tolerance floor ($\sim 10^{-8}$) everywhere, including log utility and the non-binding case |

---

## Key Results (so far)

| Experiment | Result |
|---|---|
| E1 — Black-Scholes/Merton validation | Merton's dynamic-programming solution and the continuous-time duality FOC coincide **exactly pathwise** (gap $\sim 10^{-10}$, floating-point precision); Monte Carlo budget and value-function errors decay at the expected $O(1/\sqrt{n})$ rate |
| E3 — incomplete jump market, minimal entropy | Minimal entropy measure found at $\beta^*=-2.261 \neq 0$ (naive choice); martingale condition validated across the whole Esscher family (max relative error $1.4\times10^{-4}$); exponential-utility dual objective independently minimised at the same $\beta^*$ |
| E4 — portfolio constraints | Constrained duality validated **exactly pathwise** in all 3 cases (leverage cap binding, no-short-sale binding, non-binding); utility loss is exactly zero while the constraint is slack and increases smoothly once it binds; a documented missing-support-function-penalty bug (constant multiplicative offset, analogous to the E1 discount-factor bug) |
| E2a — i.i.d. market, myopia | Dynamic and myopic policies agree to numerical tolerance ($< 10^{-6}$) for **every** CRRA parameter tested, confirming Samuelson's theorem is not specific to log utility |
| E2b — correlated regime-switching, log utility | Exactly myopic at every date and regime (gap $= 0$ to machine precision), confirming Merton's zero-hedging-demand result for log utility |
| E2c — correlated regime-switching, power utility | Strictly positive hedging-demand gap away from maturity (e.g. $0.225$ at $\alpha=-1$), vanishing as $\alpha \to 0$ and exactly zero one period before maturity |

---

## Requirements

```
Python >= 3.10
numpy
scipy
matplotlib
pytest (for the test suite)
```

Install (editable, with dev/notebook dependencies):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

---

## References

- Pliska, S.R. (1998). *Introduction to Mathematical Finance: Discrete
  Time Models*. Blackwell Publishers.
- Schachermayer, W. (2004). *Utility Maximisation in Incomplete
  Markets*. In: Stochastic Methods in Finance, Springer Lecture Notes
  in Mathematics, 1856, 225-288.
- Bouchard, B. & Cassagneux, J.-F. (2014). *Valorisation de produits
  dérivés*. Economica.
- Karatzas, I., Lehoczky, J.P., Shreve, S.E. & Xu, G.-L. (1991).
  *Martingale and Duality Methods for Utility Maximization in an
  Incomplete Market*. SIAM J. Control Optim., 29(3), 702-730.
- Cvitanić, J. & Karatzas, I. (1992). *Convex Duality in Constrained
  Portfolio Optimization*. Annals of Applied Probability, 2(4), 767-818.
- Kramkov, D. & Schachermayer, W. (1999). *The Asymptotic Elasticity of
  Utility Functions and Optimal Investment in Incomplete Markets*.
  Annals of Applied Probability, 9(3), 904-950.
- Merton, R.C. (1969). *Lifetime Portfolio Selection under Uncertainty:
  The Continuous-Time Case*. Review of Economics and Statistics, 51(3),
  247-257.
- Samuelson, P.A. (1969). *Lifetime Portfolio Selection by Dynamic
  Stochastic Programming*. Review of Economics and Statistics, 51(3),
  239-246.

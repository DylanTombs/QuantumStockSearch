# Quantum Portfolio Optimization with VQE
This project explores quantum variational algorithms for portfolio optimization using real stock return data.

## Research Goal
Can a quantum algorithm (VQE) outperform a classical random search (Monte Carlo) in identifying optimal asset allocations?

## Quantum Approach
Quantum Algorithm: Variational Quantum Eigensolver (VQE)

Objective: Maximize return with a custom Hamiltonian that encodes:

Expected returns (linear term)

Risk via variance (quadratic term)

Ansatz: EfficientSU2 / TwoLocal parameterized circuit

Optimizer: COBYLA (classical)

Backend: Qiskit Aer simulator (runs on any machine)

##  Classical Baseline
Monte Carlo simulation of 100+ random portfolios

Simulates portfolio growth across 250 trading days

Returns best classical portfolio based on average simulated return

## Research

We evaluated the performance of a quantum variational algorithm (VQE) for portfolio optimization, comparing it to a classical Monte Carlo simulation approach across multiple portfolio sizes (4, 8, 12, 16 assets), risk aversion levels (0.3, 0.5, 0.8), and circuit depths (reps = 2, 3).

In a focused test with 4 assets, the quantum method outperformed the classical approach in Sharpe ratio and cumulative return. However, in a larger, exhaustive grid search, the quantum approach did not consistently outperform the classical baseline. It did, however, remain competitive in the 4-asset case, especially as risk aversion increased — where quantum returns rose linearly, reflecting the structure of the Hamiltonian.

These results suggest that while current quantum methods (particularly VQE with TwoLocal ansatz and Aer simulation) are not yet superior at scale, they show promising potential in low-dimensional settings. The observed sensitivity to asset size, risk preference, and circuit complexity aligns with known limitations of NISQ-era quantum algorithms and offers a clear direction for future improvements.

1. Quantum advantage is size-sensitive.
For small portfolios (4 assets), VQE produced higher Sharpe ratios and better risk-adjusted returns than classical Monte Carlo methods.

This suggests that quantum encodes useful correlations or optimization paths that classical simulations, even with randomness, may not capture well in small search spaces.

2. Quantum doesn’t scale performance linearly.
As portfolio size increases (8, 12, 16), quantum performance stays stable or even improves in return—but it doesn’t outperform classical in Sharpe.

Classical approaches benefit more from dimensionality (more random portfolios = higher likelihood of finding a better one).

Quantum methods, constrained by ansatz structure, limited expressibility, and optimization noise, plateau.

3. Risk aversion impacts quantum return linearly.
That’s actually expected and a strong result.

Because your cost Hamiltonian’s return term is linear in the asset expectations, increasing risk aversion increases the penalty from the covariance, and you’re seeing the optimizer adjust accordingly. That’s a good sanity check and a valuable analytic insight.




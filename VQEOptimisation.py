import time
import numpy as np
import pandas as pd
import os
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Estimator as AerEstimator
from scipy.optimize import minimize

from plot_utils import (
    plot_return_distribution,
    plot_risk_return_scatter,
    plot_quantum_probabilities,
    plot_weight_comparison
)

class OptimizerResult:
    def __init__(self):
        self.x = None
        self.fun = None
        self.nfev = None

class CustomCOBYLA:
    def __init__(self, maxiter=100):
        self.maxiter = maxiter

    def minimize(self, fun, x0, jac=None):
        result = minimize(fun, x0, method='COBYLA', options={'maxiter': self.maxiter})
        opt_result = OptimizerResult()
        opt_result.x = result.x
        opt_result.fun = result.fun
        opt_result.nfev = result.nfev
        return opt_result

def interpret_expectation_as_weights(expectations):
    raw_weights = 0.5 * (1 - np.array(expectations))  # Z=+1 → 0, Z=−1 → 1
    norm_weights = raw_weights / np.sum(raw_weights)
    return norm_weights

def monteCarloPortfolios(numPortfolios, stockCsvPath, numSimulations=100, seed=42):
    np.random.seed(seed)
    df = pd.read_csv(stockCsvPath)

    means = df['MeanReturn'].values
    stds = df['StdReturn'].values
    numAssets = len(means)
    cov = np.diag(stds ** 2)

    portfolioResults = []
    for _ in range(numPortfolios):
        weights = np.random.dirichlet(np.ones(numAssets))
        returns = []
        for _ in range(numSimulations):
            cumulative_return = 1.0
            for _ in range(250):  
                ret = np.random.multivariate_normal(means, cov)
                daily_return = np.dot(weights, ret)
                cumulative_return *= (1 + daily_return)
            portfolioReturn = cumulative_return - 1
            returns.append(portfolioReturn)
        avgReturn = np.mean(returns)
        stdReturn = np.std(returns)
        portfolioResults.append((weights, avgReturn, stdReturn))

    return portfolioResults, df['Stock'].tolist()

def classicalMaxPortfolio(portfolios):
    return max(enumerate(portfolios), key=lambda x: x[1][1])  # (index, (weights, return, std))

# -----------------------------
# Quantum
# -----------------------------

def create_portfolio_hamiltonian(means, cov_matrix, risk_aversion=0.5):
    num_assets = len(means)

    return_terms = []
    for i in range(num_assets):
        pauli = ['I'] * num_assets
        pauli[i] = 'Z'
        coeff = -means[i] / 2
        return_terms.append(SparsePauliOp(''.join(pauli), coeff))

    risk_terms = []
    for i in range(num_assets):
        for j in range(num_assets):
            if cov_matrix[i, j] != 0:
                pauli = ['I'] * num_assets
                pauli[i] = 'Z'
                pauli[j] = 'Z'
                coeff = risk_aversion * cov_matrix[i, j] / 4
                risk_terms.append(SparsePauliOp(''.join(pauli), coeff))

    hamiltonian = sum(return_terms + risk_terms)
    return hamiltonian.simplify()

def run_vqe(hamiltonian, ansatz, estimator, optimizer, callback=None):
    def cost_function(params):
        bound_circuit = ansatz.assign_parameters(params)
        job = estimator.run(bound_circuit, hamiltonian)
        result = job.result()
        expval = result.values[0]
        if callback:
            callback(params, expval)
        return expval

    x0 = np.random.random(ansatz.num_parameters)
    result = optimizer.minimize(cost_function, x0)
    optimal_circuit = ansatz.assign_parameters(result.x)
    job = estimator.run(optimal_circuit, hamiltonian)
    final_result = job.result()

    class VQEResult:
        def __init__(self):
            self.eigenvalue = final_result.values[0]
            self.optimal_parameters = result.x
            self.optimal_circuit = optimal_circuit

    return VQEResult()

def run_vqe_portfolio_optimization_continuous(means, cov_matrix):

    hamiltonian = create_portfolio_hamiltonian(means, cov_matrix)
    num_assets = len(means)
    ansatz = TwoLocal(num_assets, 'ry', 'cz', reps=1, entanglement='linear')

    estimator = AerEstimator()
    optimizer = CustomCOBYLA(maxiter=30)

    print(" Running VQE on Aer simulator")
    result = run_vqe(hamiltonian, ansatz, estimator, optimizer)

    expectation_vals = []
    for i in range(num_assets):
        z_pauli = ['I'] * num_assets
        z_pauli[i] = 'Z'
        op = SparsePauliOp(''.join(z_pauli), coeffs=[1.0])
        val = estimator.run(result.optimal_circuit, op).result().values[0]
        expectation_vals.append(val)

    weights = interpret_expectation_as_weights(expectation_vals)

    print(f"\n Circuit depth: {result.optimal_circuit.depth()}")
    print(f" Number of parameters: {ansatz.num_parameters}")

    return weights, result

def sharpe_ratio(return_, risk, rf_rate=0.01):
    return (return_ - rf_rate) / risk if risk != 0 else 0

def main():
    start_time = time.time()
    numPortfolios = 5
    portfolios, tickers = monteCarloPortfolios(numPortfolios, "processed_stock_summary.csv")

    all_weights = [p[0] for p in portfolios]
    means = np.array([p[1] for p in portfolios])
    stds = np.array([p[2] for p in portfolios])
    cov_matrix = np.diag(stds**2)

    quantum_weights, result = run_vqe_portfolio_optimization_continuous(means, cov_matrix)

    closest_portfolio = min(enumerate(all_weights), key=lambda x: np.linalg.norm(x[1] - quantum_weights))
    q_index = closest_portfolio[0]
    q_weights, q_return, q_risk = portfolios[q_index]

    print("\n Quantum Optimal Portfolio:")
    for t, w in zip(tickers, np.round(q_weights, 3)):
        print(f"  {t}: {w}")
    print(f" Return: {q_return:.4f}")
    print(f" Risk: {q_risk:.4f}")
    print(f" Sharpe: {sharpe_ratio(q_return, q_risk):.4f}")

    bestClassicalIndex, (c_weights, c_return, c_risk) = classicalMaxPortfolio(portfolios)
    print("\n Classical Optimal Portfolio:")
    for t, w in zip(tickers, np.round(c_weights, 3)):
        print(f"  {t}: {w}")
    print(f" Return: {c_return:.4f}")
    print(f" Risk: {c_risk:.4f}")
    print(f" Sharpe: {sharpe_ratio(c_return, c_risk):.4f}")


    os.makedirs("results", exist_ok=True)

    # Extract classical metrics
    classical_returns = [p[1] for p in portfolios]
    classical_risks = [p[2] for p in portfolios]

    # Visualizations
    plot_return_distribution(classical_returns, q_return)
    plot_risk_return_scatter(classical_risks, classical_returns, q_risk, q_return)
    plot_quantum_probabilities(abs(Statevector(result.optimal_circuit).data)**2)
    plot_weight_comparison(q_weights, c_weights, tickers)

    print("\n📊 Plots saved to ./results/")

    print(f"\n Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()




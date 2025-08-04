import pandas as pd
import numpy as np
import os
import time
from VQEOptimisation import (
    monteCarloPortfolios,
    classicalMaxPortfolio,
    run_vqe_portfolio_optimization_continuous,
    sharpe_ratio
)
from plot_utils import (
    plot_return_distribution,
    plot_risk_return_scatter,
    plot_weight_comparison,
    plot_quantum_probabilities
)

def run_experiment(num_assets=5, num_portfolios=10, reps=1, risk_aversion=0.5, seed=42, exp_id="exp01"):
    print(f"\nRunning Experiment: {exp_id}")
    print(f"Assets: {num_assets}, Portfolios: {num_portfolios}, Risk Aversion: {risk_aversion}, Ansatz reps: {reps}")

    # Load top N assets
    df = pd.read_csv("processed_stock_summary.csv").head(num_assets)
    tickers = df['Stock'].tolist()

    # Generate portfolios
    portfolios, _ = monteCarloPortfolios(num_portfolios, "processed_stock_summary.csv", seed=seed)
    means = np.array([p[1] for p in portfolios])
    stds = np.array([p[2] for p in portfolios])
    cov_matrix = np.diag(stds**2)
    all_weights = [p[0] for p in portfolios]

    # Classical best
    _, (c_weights, c_return, c_risk) = classicalMaxPortfolio(portfolios)

    # Quantum optimization
    start = time.time()
    q_weights, result = run_vqe_portfolio_optimization_continuous(means, cov_matrix)
    q_runtime = time.time() - start

    # Closest simulated portfolio to VQE weights
    q_index, _ = min(enumerate(all_weights), key=lambda x: np.linalg.norm(x[1] - q_weights))
    q_return = means[q_index]
    q_risk = stds[q_index]

    print(f"🔍 Closest portfolio index to VQE result: {q_index}")

    os.makedirs("results", exist_ok=True)
    plot_return_distribution(means, q_return, path=f"results/return_dist_{exp_id}.png")
    plot_risk_return_scatter(stds, means, q_risk, q_return, path=f"results/risk_return_{exp_id}.png")
    plot_weight_comparison(q_weights, c_weights, tickers, path=f"results/weight_compare_{exp_id}.png")

    try:
        from qiskit.quantum_info import Statevector
        sv = Statevector(result.optimal_circuit)
        probs = np.abs(sv.data) ** 2
        plot_quantum_probabilities(probs, path=f"results/q_probs_{exp_id}.png")
    except Exception as e:
        print(f"⚠️ Skipped quantum probability plot: {e}")

    result_row = {
        "experiment_id": exp_id,
        "num_assets": num_assets,
        "num_portfolios": num_portfolios,
        "risk_aversion": risk_aversion,
        "ansatz_reps": reps,
        "quantum_return": q_return,
        "quantum_risk": q_risk,
        "quantum_sharpe": sharpe_ratio(q_return, q_risk),
        "classical_return": c_return,
        "classical_risk": c_risk,
        "classical_sharpe": sharpe_ratio(c_return, c_risk),
        "vqe_runtime_sec": round(q_runtime, 3),
        "depth": result.optimal_circuit.depth(),
        "parameters": result.optimal_circuit.num_parameters
    }

    return result_row

if __name__ == "__main__":
    experiments = [
        {"exp_id": "run_3a", "num_assets": 3, "reps": 1, "risk_aversion": 0.3},
        {"exp_id": "run_3b", "num_assets": 3, "reps": 2, "risk_aversion": 0.5},
        {"exp_id": "run_5a", "num_assets": 5, "reps": 1, "risk_aversion": 0.5},
        {"exp_id": "run_5b", "num_assets": 5, "reps": 2, "risk_aversion": 0.8},
        {"exp_id": "run_7a", "num_assets": 7, "reps": 1, "risk_aversion": 0.5},
    ]

    results = []
    for config in experiments:
        row = run_experiment(**config)
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/experiment_results.csv", index=False)
    print("\nAll experiments complete. Saved to experiments/experiment_results.csv")

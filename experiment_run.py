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

    df = pd.read_csv("processed_stock_summary.csv").head(num_assets)
    tickers = df['Stock'].tolist()
    means = df['MeanReturn'].values
    stds = df['StdReturn'].values
    cov_matrix = np.diag(stds ** 2)

    # Save the 3-asset subset to a temp file for Monte Carlo simulation
    df.head(num_assets).to_csv("tmp_subset.csv", index=False)

    # Now use that subset in both quantum and classical parts
    portfolios, _ = monteCarloPortfolios(num_portfolios, "tmp_subset.csv", seed=seed)

    all_weights = [p[0] for p in portfolios]


    # Classical best
    _, (c_weights, c_return, c_risk) = classicalMaxPortfolio(portfolios)

    # Quantum optimization
    start = time.time()
    q_weights, result = run_vqe_portfolio_optimization_continuous(means, cov_matrix)
    q_runtime = time.time() - start

    q_return_daily = np.dot(q_weights, means)
    q_return = q_return_daily * 250  # annualized

    q_risk_daily = np.sqrt(q_weights @ cov_matrix @ q_weights)
    q_risk = q_risk_daily * np.sqrt(250)  # annualized


    print(f"Quantum return: {q_return:.4f}, Quantum risk: {q_risk:.4f}")


    os.makedirs("results", exist_ok=True)
    #plot_return_distribution(means, q_return, path=f"results/return_dist_{exp_id}.png")
    #plot_risk_return_scatter(stds, means, q_risk, q_return, path=f"results/risk_return_{exp_id}.png")
    #plot_weight_comparison(q_weights, c_weights, tickers, path=f"results/weight_compare_{exp_id}.png")

    try:
        from qiskit.quantum_info import Statevector
        sv = Statevector(result.optimal_circuit)
        probs = np.abs(sv.data) ** 2
        #plot_quantum_probabilities(probs, path=f"results/q_probs_{exp_id}.png")
    except Exception as e:
        print(f"Skipped quantum probability plot: {e}")

    result_row = {
        "exp_id": exp_id,
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
    if os.path.exists("tmp_subset.csv"):
        os.remove("tmp_subset.csv")
    return result_row

if __name__ == "__main__":
    experiments = []
    for num_assets in [3, 4, 5]:
        for num_portfolios in [10, 20, 50]:
            for reps in [2,3,4]:
                for risk_aversion in [0.3, 0.5, 0.8]:
                    exp_id = f"n{num_assets}_p{num_portfolios}_r{reps}_a{int(risk_aversion*10)}"
                    experiments.append({
                        "exp_id": exp_id,
                        "num_assets": num_assets,
                        "num_portfolios": num_portfolios,
                        "reps": reps,
                        "risk_aversion": risk_aversion,
                    })

    results = []
    for config in experiments:
        row = run_experiment(**config)
        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/experiment_results3.csv", index=False)
    print("\nAll experiments complete. Saved to experiments/experiment_results.csv")

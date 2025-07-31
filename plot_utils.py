import matplotlib.pyplot as plt
import numpy as np

def plot_return_distribution(classical_returns, quantum_return, path='results/return_distribution.png'):
    plt.figure(figsize=(8,5))
    plt.hist(classical_returns, bins=10, alpha=0.6, label='Classical Portfolios')
    plt.axvline(quantum_return, color='red', linestyle='--', linewidth=2, label='Quantum Portfolio')
    plt.title("Portfolio Return Distribution")
    plt.xlabel("Simulated Portfolio Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_risk_return_scatter(classical_risks, classical_returns, quantum_risk, quantum_return, path='results/risk_return_scatter.png'):
    plt.figure(figsize=(8,6))
    plt.scatter(classical_risks, classical_returns, c='blue', label='Classical Portfolios', alpha=0.6)
    plt.scatter([quantum_risk], [quantum_return], c='red', marker='x', s=100, label='Quantum Portfolio')
    plt.title("Risk vs. Return")
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_quantum_probabilities(probabilities, path='results/quantum_probabilities.png'):
    plt.figure(figsize=(10,5))
    plt.bar(range(len(probabilities)), probabilities, color='purple')
    plt.title("Quantum Circuit Output Probabilities")
    plt.xlabel("Basis State Index")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_weight_comparison(quantum_weights, classical_weights, tickers, path='results/weight_comparison.png'):
    x = np.arange(len(tickers))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, classical_weights, width, label='Classical')
    plt.bar(x + width/2, quantum_weights, width, label='Quantum')
    plt.xticks(x, tickers, rotation=45)
    plt.ylabel("Portfolio Weight")
    plt.title("Asset Allocation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

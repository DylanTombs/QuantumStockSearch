import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load results
df = pd.read_csv("experiments/experiment_results4.csv")

# Create output folder
os.makedirs("analysis", exist_ok=True)

# === 1. Sharpe Ratio Comparison ===
df['sharpe_diff'] = df['quantum_sharpe'] - df['classical_sharpe']

plt.figure(figsize=(10, 6))
sns.boxplot(x="num_assets", y="sharpe_diff", data=df)
plt.axhline(0, linestyle='--', color='red')
plt.title("Quantum - Classical Sharpe Ratio by Number of Assets")
plt.ylabel("Sharpe Ratio Improvement")
plt.xlabel("Number of Assets")
plt.savefig("analysis/analysis4/sharpe_diff_by_assets.png")
plt.close()

# === 2. Runtime Comparison ===
plt.figure(figsize=(10, 6))
sns.barplot(x="exp_id", y="vqe_runtime_sec", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Quantum VQE Runtime per Experiment")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig("analysis/analysis4/vqe_runtime_by_experiment.png")
plt.close()

# === 3. Sharpe Ratio Heatmap (Quantum vs Risk Aversion and Ansatz reps) ===
pivot = df.pivot_table(index='risk_aversion', columns='ansatz_reps', values='quantum_sharpe')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Quantum Sharpe Ratio: Risk Aversion vs Ansatz Reps")
plt.savefig("analysis/analysis4/sharpe_heatmap.png")
plt.close()

# === 4. Return vs Risk Scatter ===
plt.figure(figsize=(10, 6))
sns.scatterplot(x="quantum_risk", y="quantum_return", label="Quantum", data=df)
sns.scatterplot(x="classical_risk", y="classical_return", label="Classical", data=df)
plt.title("Risk vs Return: Quantum vs Classical")
plt.xlabel("Risk (std dev)")
plt.ylabel("Return")
plt.legend()
plt.savefig("analysis/analysis4/risk_return_comparison.png")
plt.close()

print("Analysis complete. Visuals saved in analysis/analysis4/")

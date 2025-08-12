import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy import stats
import os
from matplotlib.ticker import PercentFormatter

# Configure visualisation
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
os.makedirs("analysis/trial", exist_ok=True)

# Load and preprocess CSV data
def load_consolidated_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the consolidated CSV results"""
    df = pd.read_csv(csv_path)
    
    # Convert string representations to Python objects
    for col in ['tickers', 'quantum.weights', 'data_stats.mean_returns', 
               'data_stats.return_stds', 'data_stats.cov_matrix']:
        df[col] = df[col].apply(ast.literal_eval)
    
    # Calculate additional metrics
    df['sharpe_diff_q_mc'] = df['quantum.sharpe_ratio'] - df['classical.monte_carlo.sharpe_ratio']
    df['sharpe_diff_q_mvo'] = df['quantum.sharpe_ratio'] - df['classical.mean_variance.sharpe_ratio']
    df['quantum.concentration'] = df['quantum.weights'].apply(lambda w: sum(x**2 for x in w))
    df['num_assets'] = df['tickers'].apply(len)
    
    return df

df = load_consolidated_results("trial/consolidated_results.csv")

# 1. Performance Differential Analysis
def performance_differential_analysis(df: pd.DataFrame):
    """Statistical comparison of quantum vs classical methods"""
    # Prepare statistical test results
    results = []
    for (num_assets, risk_aversion), group in df.groupby(['parameters.num_assets', 'parameters.risk_aversion']):
        # Quantum vs Monte Carlo
        t_stat, p_val = stats.ttest_rel(group['quantum.sharpe_ratio'], 
                                       group['classical.monte_carlo.sharpe_ratio'])
        effect_size = group['sharpe_diff_q_mc'].mean() / group['sharpe_diff_q_mc'].std()
        
        # Quantum vs MVO
        t_stat_mvo, p_val_mvo = stats.ttest_rel(group['quantum.sharpe_ratio'], 
                                              group['classical.mean_variance.sharpe_ratio'])
        effect_size_mvo = group['sharpe_diff_q_mvo'].mean() / group['sharpe_diff_q_mvo'].std()
        
        results.append({
            'num_assets': num_assets,
            'risk_aversion': risk_aversion,
            'q_mc_mean_diff': group['sharpe_diff_q_mc'].mean(),
            'q_mc_p_value': p_val,
            'q_mc_effect_size': effect_size,
            'q_mvo_mean_diff': group['sharpe_diff_q_mvo'].mean(),
            'q_mvo_p_value': p_val_mvo,
            'q_mvo_effect_size': effect_size_mvo
        })
    
    stats_df = pd.DataFrame(results)
    stats_df.to_csv("analysis/trial/performance_significance.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='parameters.num_assets', y='sharpe_diff_q_mc', 
               hue='parameters.risk_aversion', data=df)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Quantum vs Monte Carlo")
    plt.ylabel("Sharpe Ratio Difference")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='parameters.num_assets', y='sharpe_diff_q_mvo', 
               hue='parameters.risk_aversion', data=df)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Quantum vs Mean-Variance")
    plt.ylabel("")
    
    plt.suptitle("Performance Differential Analysis", y=1.02)
    plt.tight_layout()
    plt.savefig("analysis/trial/performance_comparison.png", bbox_inches='tight')
    plt.close()

# 2. Portfolio Composition Analysis
def portfolio_composition_analysis(df: pd.DataFrame):
    """Analyse weight distributions and characteristics"""
    # Weight concentration analysis
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='parameters.num_assets', y='quantum.concentration', 
               hue='parameters.risk_aversion', data=df)
    plt.title("Portfolio Concentration (Herfindahl Index)")
    plt.ylabel("Concentration (0=diversified, 1=concentrated)")
    plt.savefig("analysis/trial/portfolio_concentration.png", bbox_inches='tight')
    plt.close()
    
    # Weight distribution examples
    example_exp = df.iloc[0]['experiment_id']
    example_weights = df[df['experiment_id'] == example_exp]['quantum.weights'].iloc[0]
    example_tickers = df[df['experiment_id'] == example_exp]['tickers'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    plt.pie(example_weights, labels=example_tickers, autopct='%1.1f%%')
    plt.title(f"Portfolio Weights Example (Exp {example_exp})")
    plt.savefig("analysis/trial/portfolio_weights_example.png", bbox_inches='tight')
    plt.close()

# 3. Risk-Return Tradeoff Analysis
def risk_return_analysis(df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    
    methods = [
        ('Quantum', 'quantum.return', 'quantum.risk', 'blue'),
        ('Monte Carlo', 'classical.monte_carlo.return', 'classical.monte_carlo.risk', 'green'),
        ('Mean-Variance', 'classical.mean_variance.return', 'classical.mean_variance.risk', 'red')
    ]
    
    for label, ret_col, risk_col, color in methods:
        sns.scatterplot(x=risk_col, y=ret_col, data=df, 
                       label=label, color=color, s=100, alpha=0.6)
    
    plt.title("Risk-Return Tradeoff: Quantum vs Classical Methods", fontsize=14)
    plt.xlabel("Annualised Risk (Standard Deviation)", fontsize=12)
    plt.ylabel("Annualised Return", fontsize=12)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("analysis/trial/risk_return_tradeoff.png", bbox_inches='tight')
    plt.close()

# 4. Quantum Circuit Analysis
def quantum_metrics_analysis(df: pd.DataFrame):
    """Analyse quantum-specific performance metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Circuit depth vs performance
    sns.scatterplot(x='quantum.circuit_metrics.depth', y='quantum.sharpe_ratio',
                   hue='parameters.num_assets', size='parameters.risk_aversion',
                   data=df, ax=axes[0])
    axes[0].set_title("Circuit Depth vs Sharpe Ratio")
    
    # State entropy vs performance
    sns.scatterplot(x='quantum.state_metrics.entropy', y='quantum.sharpe_ratio',
                   hue='parameters.num_assets', size='parameters.risk_aversion',
                   data=df, ax=axes[1])
    axes[1].set_title("State Entropy vs Sharpe Ratio")
    
    # Runtime analysis
    sns.regplot(x='parameters.num_assets', y='quantum.optimizer_time', 
           data=df, ax=axes[2], order=1, ci=95)
    axes[2].set_title("Optimisation Time Scaling")
    axes[2].set_xlabel("Number of Assets")
    axes[2].set_ylabel("Time (seconds)")
    
    plt.suptitle("Quantum Performance Metrics", y=1.05)
    plt.tight_layout()
    plt.savefig("analysis/trial/quantum_metrics.png", bbox_inches='tight')
    plt.close()

# 5. Risk Metrics Comparison
def risk_metrics_analysis(df: pd.DataFrame):
    """Compare VaR, CVaR, and max drawdown across methods"""
    risk_metrics = ['var', 'cvar', 'max_drawdown']
    
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(risk_metrics, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df[[f'quantum.{metric}', 
                           f'classical.monte_carlo.{metric}',
                           f'classical.mean_variance.{metric}']])
        plt.title(f"{metric.upper()} Comparison")
        plt.xticks([0, 1, 2], ['Quantum', 'Monte Carlo', 'Mean-Variance'])
        plt.ylabel(metric.upper())
    
    plt.suptitle("Risk Metrics Comparison Across Methods", y=1.02)
    plt.tight_layout()
    plt.savefig("analysis/trial/risk_metrics_comparison.png", bbox_inches='tight')
    plt.close()

# Execute all analyses
analyses = [
    performance_differential_analysis,
    portfolio_composition_analysis,
    risk_return_analysis,
    quantum_metrics_analysis,
    risk_metrics_analysis
]

for analysis in analyses:
    analysis(df)

print("Analysis complete.")
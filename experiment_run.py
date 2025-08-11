from VQEOptimisation import (
    monteCarloPortfolios,
    classicalMaxPortfolio,
    run_vqe_portfolio_optimisation,
    sharpe_ratio
)

from scipy.optimize import minimize

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import json
import hashlib
from typing import Dict, List, Optional
import warnings
import logging
from multiprocessing import Pool, cpu_count
import scipy.stats as stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_optimisation.log'),
        logging.StreamHandler()
    ]
)

# Constants
BUSINESS_DAYS_PER_YEAR = 252
CONFIDENCE_LEVEL = 0.95
MAX_RUNTIME_SECONDS = 3600  # 1 hour timeout for quantum optimization
MIN_SAMPLE_SIZE = 30  # For statistical significance

def validate_input_data(df: pd.DataFrame) -> None:
    """Perform rigorous validation of input financial data."""
    required_columns = {'Stock', 'MeanReturn', 'StdReturn'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Input data missing required columns. Needed: {required_columns}, Found: {set(df.columns)}")
    
    if df.isnull().any().any():
        raise ValueError("Input data contains null values")
    
    if (df['StdReturn'] <= 0).any():
        raise ValueError("Standard deviations must be positive")
    
    # Check for normality of returns (Jarque-Bera test)
    for stock in df['Stock']:
        returns = np.random.normal(df[df['Stock'] == stock]['MeanReturn'].iloc[0],
                                  df[df['Stock'] == stock]['StdReturn'].iloc[0],
                                  1000)
        _, p_value = stats.jarque_bera(returns)
        if p_value < 0.01:
            warnings.warn(f"Returns for {stock} may not be normally distributed (p={p_value:.4f})")

def calculate_metrics(weights: np.ndarray, means: np.ndarray, cov_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate portfolio metrics with error checking"""
    if not np.isclose(weights.sum(), 1, atol=1e-6):
        raise ValueError(f"Weights must sum to 1 (actual sum: {weights.sum():.8f})")
    
    if (weights < -1e-6).any():
        raise ValueError(f"Negative weights detected: {weights}")
    
    portfolio_return = np.dot(weights, means) * BUSINESS_DAYS_PER_YEAR
    portfolio_risk = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(BUSINESS_DAYS_PER_YEAR)
    
    # Additional risk metrics
    var = stats.norm.ppf(1-CONFIDENCE_LEVEL, portfolio_return, portfolio_risk)
    cvar = portfolio_return - (portfolio_risk * stats.norm.pdf(stats.norm.ppf(1-CONFIDENCE_LEVEL)) / (1-CONFIDENCE_LEVEL))
    
    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe_ratio': sharpe_ratio(portfolio_return, portfolio_risk),  # Assuming 2% risk-free rate
        'var': var,
        'cvar': cvar,
        'max_drawdown': portfolio_risk * 2.5  # Approximation for normal distributions
    }

def run_quantum_optimisation(means, cov_matrix, risk_aversion, reps):
    """Simplified wrapper that works with the enhanced return structure"""
    weights, result_dict = run_vqe_portfolio_optimisation(
        means, cov_matrix, risk_aversion, reps
    )
    
    metrics = calculate_metrics(weights, means, cov_matrix)
    full_results = {
        **metrics,
        **result_dict,  # Includes all the quantum metadata
        'weights': weights.tolist()  # For easy serialization
    }
    
    return weights, full_results

def run_classical_baseline(portfolios: List, means: np.ndarray, cov_matrix: np.ndarray) -> Dict:
    """Run classical optimization with multiple methods for robust comparison."""
    # Monte Carlo baseline
    _, (c_weights, c_return, c_risk) = classicalMaxPortfolio(portfolios)
    mc_metrics = calculate_metrics(c_weights, means, cov_matrix)
    
    # Mean-variance optimization
    try:
        
        def objective(weights):
            port_return = np.dot(weights, means)
            port_risk = np.sqrt(weights @ cov_matrix @ weights)
            return - (port_return - 0.5 * port_risk**2)  # Quadratic utility
            
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(means)))
        result = minimize(objective, x0=np.ones(len(means))/len(means),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            mvo_weights = result.x
            mvo_metrics = calculate_metrics(mvo_weights, means, cov_matrix)
        else:
            raise ValueError("MVO optimization failed")
    except Exception as e:
        logging.warning(f"Mean-variance optimisation failed: {str(e)}")
        mvo_metrics = {k: np.nan for k in mc_metrics}
    
    return {
        'monte_carlo': mc_metrics,
        'mean_variance': mvo_metrics
    }

def create_experiment_hash(params: Dict) -> str:
    """Create a unique hash for experiment reproducibility."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]

def run_experiment(
    num_assets: int = 5,
    num_portfolios: int = 10,
    reps: int = 1,
    risk_aversion: float = 0.5,
    seed: int = 42,
    exp_id: Optional[str] = None,
    data_path: str = "processed_stock_summary.csv"
) -> Dict:
    # Setup experiment ID and logging
    if not exp_id:
        params = locals().copy()
        del params['exp_id']
        exp_id = create_experiment_hash(params)
    
    logging.info(f"\n=== Starting Experiment {exp_id} ===")
    logging.info(f"Configuration: Assets={num_assets}, Portfolios={num_portfolios}, "
                f"Reps={reps}, RiskAversion={risk_aversion}, Seed={seed}")
    
    # Initialize temp_path early
    temp_path = None
    try:
        # Load and validate data
        df = pd.read_csv(data_path).head(num_assets)
        validate_input_data(df)
        tickers = df['Stock'].tolist()
        means = df['MeanReturn'].values
        stds = df['StdReturn'].values
        
        # Create proper covariance matrix
        np.random.seed(seed)
        random_correl = np.random.uniform(-0.3, 0.3, size=(num_assets, num_assets))
        random_correl = (random_correl + random_correl.T) / 2
        np.fill_diagonal(random_correl, 1)
        cov_matrix = np.outer(stds, stds) * random_correl
        
        # Save temporary subset for Monte Carlo
        temp_path = f"temp_{exp_id}.csv"
        df.head(num_assets).to_csv(temp_path, index=False)
        
        # Generate Monte Carlo portfolios
        portfolios, mc_tickers = monteCarloPortfolios(
            numPortfolios=num_portfolios,
            stockCsvPath=temp_path,
            seed=seed
        )
        if len(portfolios) < MIN_SAMPLE_SIZE:
            raise ValueError(f"Insufficient portfolios generated: {len(portfolios)}")
        
        # Verify tickers match
        if tickers != mc_tickers:
            raise ValueError("Tickers mismatch between data and Monte Carlo simulation")
            
        # Run classical baselines
        classical_results = run_classical_baseline(portfolios, means, cov_matrix)
        
        # Run quantum optimization
        q_weights, quantum_results = run_quantum_optimisation(means, cov_matrix, risk_aversion, reps)
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise
        
    finally:
        # Clean up temp file in finally block to ensure it runs
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Compile results with type conversion
    result_row = {
        'experiment_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'num_assets': int(num_assets),
            'num_portfolios': int(num_portfolios),
            'ansatz_reps': int(reps),
            'risk_aversion': float(risk_aversion),
            'random_seed': int(seed)
        },
        'tickers': tickers,
        'quantum': {
            k: convert_numpy_types(v) 
            for k, v in quantum_results.items()
        },
        'classical': {
            k: convert_numpy_types(v) 
            for k, v in classical_results.items()
        },
        'data_stats': {
            'mean_returns': means.tolist(),
            'return_stds': stds.tolist(),
            'cov_matrix': cov_matrix.tolist()
        }
    }

    # Save with custom JSON encoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs("results", exist_ok=True)
    with open(f"trial/{exp_id}.json", "w") as f:
        json.dump(result_row, f, indent=2, cls=NumpyEncoder)
    
    logging.info(f"Completed Experiment {exp_id}")
    return result_row
    

def run_experiment_wrapper(args):
    """Wrapper for parallel execution."""
    try:
        return run_experiment(**args)
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        return None

def main():
    """Run a comprehensive parameter sweep with parallel execution."""
    # Define experiment grid
    parameter_grid = {
        'num_assets': [3],
        'num_portfolios': [100, 500],
        'reps': [1],
        'risk_aversion': [0.2, 0.5, 0.8],
        'seed': [123]
    }
    
    # Generate all combinations
    experiments = []
    for num_assets in parameter_grid['num_assets']:
        for num_portfolios in parameter_grid['num_portfolios']:
            for reps in parameter_grid['reps']:
                for risk_aversion in parameter_grid['risk_aversion']:
                    for seed in parameter_grid['seed']:
                        experiments.append({
                            'num_assets': num_assets,
                            'num_portfolios': num_portfolios,
                            'reps': reps,
                            'risk_aversion': risk_aversion,
                            'seed': seed
                        })
    
    # Run experiments in parallel
    logging.info(f"Starting {len(experiments)} experiments")
    start_time = time.time()
    
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        results = pool.map(run_experiment_wrapper, experiments)
    
    # Filter out failed experiments
    successful_results = [r for r in results if r is not None]
    logging.info(f"Completed {len(successful_results)}/{len(experiments)} experiments in "
                f"{(time.time()-start_time)/60:.1f} minutes")
    
    # Save consolidated results
    df = pd.json_normalize(successful_results)
    df.to_csv("trial/consolidated_results.csv", index=False)
    logging.info("Saved consolidated results to trial/consolidated_results.csv")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import time
import logging
import scipy.stats as stats

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit import transpile
from scipy.optimize import minimize

class OptimiserResult:
    def __init__(self):
        self.x = None
        self.fun = None
        self.nfev = None

class CustomCOBYLA:
    """
    Wrapper around scipy's COBYLA optimiser to match Qiskit's optimiser interface.
    COBYLA (Constrained Optimization By Linear Approximations) is derivative-free
    optimisation method, which is for quantum circuits where gradients are
    expensive to compute.
    """

    def __init__(self, maxiter=100):
        self.maxiter = maxiter

    def minimise(self, fun, x0, jac=None):
        """
        Run the COBYLA optimisation algorithm.
        
        Args:
            fun: Objective function to minimise (our VQE cost function)
            x0: Initial parameter guess
            jac: Jacobian (ignored by COBYLA as is derivative-free)

        """
        result = minimize(fun, x0, method='COBYLA', options={'maxiter': self.maxiter})
        opt_result = OptimiserResult()
        opt_result.x = result.x
        opt_result.fun = result.fun
        opt_result.nfev = result.nfev
        return opt_result

# --- Portfolio Utilities ---

def interpret_expectation_as_weights(expectations):
    """
    Convert quantum measurement expectations to portfolio weights.
    
    Mathematical explanation:
    - Pauli-Z eigenvalues are +1 and -1
    - We want to map these to portfolio weights between 0 and 1
    - Formula: weight_i = 0.5 * (1 - <Z_i>)
      * If <Z_i> = +1 (spin up), weight = 0 (don't invest)
      * If <Z_i> = -1 (spin down), weight = 1 (fully invest)
      * Intermediate values give proportional weights
    
    Args:
        expectations: List of <Z_i> expectation values for each asset
    
    Returns:
        Normalised portfolio weights that sum to 1
    """
    raw_weights = 0.5 * (1 - np.array(expectations))  # Z=+1 → 0, Z=−1 → 1
    weights = np.clip(raw_weights, 0, None)
    weights /= np.sum(weights) if np.sum(weights) != 0 else 1
    return weights

def monteCarloPortfolios(numPortfolios, stockCsvPath, numSimulations=100, seed=42):
    """
    Generate random portfolios using Monte Carlo simulation for comparison.
    
    This creates classical benchmark portfolios by:
    1. Randomly generating portfolio weights using Dirichlet distribution
    2. Simulating daily returns for 1 year (250 trading days)
    3. Computing annual return and risk (standard deviation)
    
    Maths:
    - Daily returns are sampled from multivariate normal distribution
    - Portfolio return = Σ(weight_i * return_i) for each day
    - Annual return = (1 + daily_return)^250 - 1 (compound growth)
    - Risk = standard deviation of annual returns across simulations
    
    Args:
        numPortfolios: Number of random portfolios to generate
        stockCsvPath: Path to CSV with stock statistics
        numSimulations: Monte Carlo simulations per portfolio
        seed: Random seed for reproducibility
    
    Returns:
        portfolioResults: List of (weights, return, risk) tuples
        tickers: List of stock ticker symbols
    """

    np.random.seed(seed)
    df = pd.read_csv(stockCsvPath)

    means = df['MeanReturn'].values
    stds = df['StdReturn'].values
    numAssets = len(means)

    # Simplified covariance matrix (diagonal - assumes no correlation between assets)
    # In reality, you'd use the full covariance matrix: cov[i,j] = correlation[i,j] * std[i] * std[j]
    cov = np.diag(stds ** 2)

    portfolioResults = []
    for _ in range(numPortfolios):
        weights = np.random.dirichlet(np.ones(numAssets))
        returns = []
        for _ in range(numSimulations):
            cumulative_return = 1.0 # Start with £1 invested
            # Simulate 250 trading days (1 year)
            for _ in range(250):
                # Sample daily returns from multivariate normal distribution
                ret = np.random.multivariate_normal(means, cov)
                # Calculate portfolio's daily return (weighted average)
                daily_return = np.dot(weights, ret)
                cumulative_return *= (1 + daily_return)
            # Annual return (subtract 1 to get percentage gain)
            portfolioReturn = cumulative_return - 1
            returns.append(portfolioReturn)

        # Calculate portfolio statistics across all simulations
        avgReturn = np.mean(returns)
        stdReturn = np.std(returns)
        portfolioResults.append((weights, avgReturn, stdReturn))

    return portfolioResults, df['Stock'].tolist()

def classicalMaxPortfolio(portfolios):
    """
    Find the classical portfolio with maximum expected return.
    
    This is a simple classical optimisation strategy that just picks
    the portfolio with the highest expected return (ignoring risk).
    
    """
    return max(enumerate(portfolios), key=lambda x: x[1][1])  # max by return

def sharpe_ratio(return_, risk, rf_rate=0.01):
    return float((return_ - rf_rate) / risk) if risk != 0 else 0.0

# --- Quantum Portfolio Optimization ---

def create_portfolio_hamiltonian(means, cov_matrix, risk_aversion=0.5):
    """
    Create the quantum Hamiltonian for portfolio optimisation.
    
    Maths:
    The portfolio optimisation problem can be formulated as:
    minimise: λ * x^T * Σ * x - μ^T * x
    where:
    - x = portfolio weights vector
    - Σ = covariance matrix (risk)
    - μ = expected returns vector
    - λ = risk aversion parameter
    
    In quantum form, we map this to a Hamiltonian using Pauli-Z operators:
    H = Σᵢ (-μᵢ/2 * Zᵢ) + λ * Σᵢⱼ (σᵢⱼ/4 * Zᵢ * Zⱼ)
    
    The ground state of this Hamiltonian encodes the optimal portfolio.
    
    Args:
        means: Expected returns vector (μ)
        cov_matrix: Covariance matrix (Σ)
        risk_aversion: Risk aversion parameter (λ)
    
    Returns:
        SparsePauliOp: The portfolio optimisation Hamiltonian
    """


    num_assets = len(means)
    # RETURN TERMS: -μᵢ/2 * Zᵢ
    # These terms encourage investment in assets with higher expected returns
    return_terms = []
    for i in range(num_assets):
        # Create Pauli string with Z on qubit i
        pauli = ['I'] * num_assets
        pauli[i] = 'Z'

        # Coefficient is -μᵢ/2 (negative because we want to minimise energy)
        # The factor of 1/2 comes from the mapping: weight = (1 - <Z>)/2
        coeff = -means[i] / 2
        return_terms.append(SparsePauliOp(''.join(pauli), coeff))

    # RISK TERMS: λ * σᵢⱼ/4 * Zᵢ * Zⱼ
    # These terms penalize portfolio variance (risk)
    risk_terms = []
    for i in range(num_assets):
        for j in range(num_assets):
            if cov_matrix[i, j] != 0:
                pauli = ['I'] * num_assets
                pauli[i] = 'Z'
                pauli[j] = 'Z'

                # Coefficient includes risk aversion parameter
                # Factor of 1/4 comes from the weight mapping
                coeff = risk_aversion * cov_matrix[i, j] / 4
                risk_terms.append(SparsePauliOp(''.join(pauli), coeff))

    # Combine all terms into the full Hamiltonian
    hamiltonian = sum(return_terms + risk_terms)
    return hamiltonian.simplify()

def run_vqe(hamiltonian, ansatz, estimator, optimiser, callback=None):
    """
    Run the Variational Quantum Eigensolver (VQE) algorithm.
    
    VQE is a hybrid quantum-classical algorithm that:
    1. Prepares a parameterized quantum state |ψ(θ)⟩
    2. Measures the expectation value ⟨ψ(θ)|H|ψ(θ)⟩
    3. Uses classical optimisation to minimise this expectation value
    4. Repeats until convergence
    
    The key insight: the ground state of H encodes the optimal portfolio,
    and VQE finds approximations to this ground state.
    
    Args:
        hamiltonian: The problem Hamiltonian to minimise
        ansatz: Parameterised quantum circuit
        estimator: Quantum backend for computing expectation values
        optimiser: Classical optimiser for parameter updates
        callback: Optional function called at each iteration
    
    Returns:
        VQEResult: Object containing optimal parameters and energy
    """

    def cost_function(params):
        """
        Cost function for the classical optimiser.
        
        This function:
        1. Takes circuit parameters from classical optimiser
        2. Creates quantum circuit with those parameters
        3. Computes ⟨ψ(θ)|H|ψ(θ)⟩ on quantum hardware/simulator
        4. Returns this expectation value to the optimiser
        
        Args:
            params: Circuit parameters from optimiser
            
        Returns:
            Expectation value (energy) to minimise
        """

        try:
            # Assign parameters to the circuit
            bound_circuit = ansatz.assign_parameters(params)
            
            # This is the quantum part: |ψ⟩ = U(θ)|0⟩, then compute ⟨ψ|H|ψ⟩
            job = estimator.run([(bound_circuit, hamiltonian)])
            result = job.result()
            expval = result[0].data.evs.item()
            
            if callback:
                callback(params, expval)
            return expval
        except Exception as e:
            print(f"Error in cost function: {str(e)}")
            return np.inf  # Return high cost if evaluation fails

    # Initialize random parameters matching the ansatz
    x0 = np.random.random(ansatz.num_parameters)
    result = optimiser.minimise(cost_function, x0)
    
    # Run classical optimisation to find optimal parameters
    # This is the classical part of the hybrid algorithm
    optimal_circuit = ansatz.assign_parameters(result.x)

    # Compute final energy with optimal parameters
    final_job = estimator.run([(optimal_circuit, hamiltonian)])
    final_energy = final_job.result()[0].data.evs.item()

    class VQEResult:
        def __init__(self):
            self.eigenvalue = final_energy
            self.optimal_parameters = result.x
            self.optimal_circuit = optimal_circuit

    return VQEResult()

def run_vqe_portfolio_optimisation(means, cov_matrix, risk_aversion=0.5, reps=1):
    """
    Run the complete quantum portfolio optimisation pipeline with enhanced return structure.
    
    Returns:
        weights: Optimal portfolio weights (normalised numpy array)
        result_dict: Dictionary containing complete optimisation metadata including:
            - optimal_value: Final optimised objective value
            - optimiser_evals: Number of function evaluations
            - optimiser_time: Time taken for optimisation
            - circuit_metrics: Depth, parameters, etc.
            - state_metrics: Quantum state properties
    """
    # Initialize result dictionary
    result_dict = {
        'optimal_value': None,
        'optimiser_evals': 0,
        'optimiser_time': 0,
        'circuit_metrics': {
            'depth': 0,
            'num_parameters': 0,
            'gate_counts': {}
        },
        'state_metrics': {
            'max_prob_state': 0,
            'entropy': 0,
            'num_significant_states': 0
        }
    }
    
    start_time = time.time()
    num_assets = len(means)
    
    try:
        # 1. Create Hamiltonian
        hamiltonian = create_portfolio_hamiltonian(means, cov_matrix, risk_aversion)
        
        # 2. Setup ansatz circuit
        ansatz = TwoLocal(num_assets, 'ry', 'cz', reps=reps, entanglement='linear', insert_barriers=True)
        transpiled_ansatz = transpile(ansatz, backend=AerSimulator())
        
        # Store circuit metrics
        result_dict['circuit_metrics'] = {
            'depth': transpiled_ansatz.depth(),
            'num_parameters': ansatz.num_parameters,
            'gate_counts': dict(transpiled_ansatz.count_ops())
        }
        
        # 3. Run VQE
        estimator = BackendEstimatorV2(backend=AerSimulator())
        optimiser = CustomCOBYLA(maxiter=250)
        vqe_result = run_vqe(hamiltonian, transpiled_ansatz, estimator, optimiser)
        
        # Store optimisation results
        result_dict.update({
            'optimal_value': getattr(vqe_result, 'optimal_value', None),
            'optimiser_evals': getattr(vqe_result, 'optimiser_evals', 0),
            'optimiser_time': time.time() - start_time
        })
        
        # 4. Extract weights from quantum state
        expectation_vals = []
        for i in range(num_assets):
            z_pauli = ['I'] * num_assets
            z_pauli[i] = 'Z'
            op = SparsePauliOp(''.join(z_pauli), coeffs=[1.0])
            job = estimator.run([(vqe_result.optimal_circuit, op)])
            val = job.result()[0].data.evs.item()
            expectation_vals.append(val)
            
        weights = interpret_expectation_as_weights(expectation_vals)
        
        # 5. Analyze final quantum state
        try:
            sv = Statevector(vqe_result.optimal_circuit)
            probs = np.abs(sv.data) ** 2
            result_dict['state_metrics'] = {
                'max_prob_state': np.max(probs),
                'entropy': stats.entropy(probs),
                'num_significant_states': np.sum(probs > 0.01)
            }
        except Exception as e:
            logging.warning(f"Statevector analysis failed: {str(e)}")
        
        return weights, result_dict
        
    except Exception as e:
        logging.error(f"Quantum optimisation failed: {str(e)}")
        # Return uniform weights and failed status
        weights = np.ones(num_assets)/num_assets
        result_dict['optimisation_status'] = 'failed'
        result_dict['error'] = str(e)
        return weights, result_dict





import math
import time
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit import OptimizerResult
from scipy.optimize import minimize

def monteCarloPortfolios(numPortfolios, stockCsvPath, numSimulations=100, seed=42):
    np.random.seed(seed)
    df = pd.read_csv(stockCsvPath)

    means = df['MeanReturn'].values
    stds = df['StdReturn'].values
    numAssets = len(means)

    cov = np.diag(stds ** 2)  # Diagonal covariance matrix for simplicity

    portfolioResults = []
    for _ in range(numPortfolios):
        weights = np.random.dirichlet(np.ones(numAssets))
        returns = []
        for _ in range(numSimulations):
            cumulative_return = 1.0
            for _ in range(250):  # Trading days
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

def create_portfolio_hamiltonian(means, cov_matrix, risk_aversion=0.5):
    """
    Creates a Hamiltonian representing portfolio optimization:
    H = - (expected return) + λ * (risk)
    where λ is risk aversion parameter.
    """
    num_assets = len(means)
    
    # Pauli Z terms for returns
    return_terms = []
    for i in range(num_assets):
        pauli = ['I'] * num_assets
        pauli[i] = 'Z'
        return_terms.append(SparsePauliOp(''.join(pauli), -means[i]/2))
    
    # Pauli terms for risk (variance)
    risk_terms = []
    for i in range(num_assets):
        for j in range(num_assets):
            if cov_matrix[i,j] != 0:
                pauli_i = ['I'] * num_assets
                pauli_i[i] = 'Z'
                pauli_j = ['I'] * num_assets
                pauli_j[j] = 'Z'
                risk_terms.append(
                    SparsePauliOp(''.join(pauli_i)) * 
                    SparsePauliOp(''.join(pauli_j)) * 
                    (risk_aversion * cov_matrix[i,j]/4))
    
    # Combine all terms
    hamiltonian = sum(return_terms) + sum(risk_terms)
    return hamiltonian.simplify()

class CustomCOBYLA:
    """Wrapper for SciPy's COBYLA to match Qiskit optimizer interface"""
    def __init__(self, maxiter=100):
        self.maxiter = maxiter
    
    def minimize(self, fun, x0, jac=None):
        result = minimize(fun, x0, method='COBYLA', 
                        options={'maxiter': self.maxiter})
        opt_result = OptimizerResult()
        opt_result.x = result.x
        opt_result.fun = result.fun
        opt_result.nfev = result.nfev
        return opt_result

def run_vqe(hamiltonian, ansatz, estimator, optimizer, callback=None):
    """Custom VQE implementation using Qiskit Runtime Estimator"""
    def cost_function(params):
        # Bind parameters and estimate expectation value
        bound_circuit = ansatz.bind_parameters(params)
        job = estimator.run(bound_circuit, hamiltonian)
        result = job.result()
        expval = result.values[0]
        if callback:
            callback(params, expval)
        return expval
    
    # Initial random parameters
    x0 = np.random.random(ansatz.num_parameters)
    
    # Run optimization
    result = optimizer.minimize(cost_function, x0)
    
    # Get optimal state
    optimal_circuit = ansatz.bind_parameters(result.x)
    job = estimator.run(optimal_circuit, hamiltonian)
    final_result = job.result()
    
    class VQEResult:
        def __init__(self):
            self.eigenvalue = final_result.values[0]
            self.optimal_parameters = result.x
            self.optimal_circuit = optimal_circuit
    
    return VQEResult()

def run_vqe_portfolio_optimization(means, cov_matrix, backend_name):
    # Create Hamiltonian
    hamiltonian = create_portfolio_hamiltonian(means, cov_matrix)
    
    # Variational form (ansatz)
    num_qubits = len(means)
    ansatz = TwoLocal(num_qubits, 'ry', 'cz', reps=1, entanglement='linear')
    
    # Initialize runtime service
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    
    with Session(service=service, backend=backend_name) as session:
        # Set up estimator and optimizer
        estimator = Estimator(session=session)
        optimizer = CustomCOBYLA(maxiter=30)  # Reduced for demonstration
        
        print("⚛️ Running VQE on", backend_name)
        result = run_vqe(hamiltonian, ansatz, estimator, optimizer)
        
        # Get state probabilities (using simulator for measurement)
        from qiskit.quantum_info import Statevector
        sv = Statevector(result.optimal_circuit)
        probabilities = abs(sv)**2
        
        # Interpret the result
        max_prob_idx = np.argmax(probabilities)
        binary_str = format(max_prob_idx, f'0{num_qubits}b')
        weights = np.array([int(bit) for bit in binary_str[::-1]])
        weights = weights / np.sum(weights)  # Normalize
        
        return weights

def main():
    start_time = time.time()
    numPortfolios = 8  # Reduced for faster execution
    portfolios, tickers = monteCarloPortfolios(numPortfolios, "processed_stock_summary.csv")
    
    # Extract means and covariance
    all_weights = [p[0] for p in portfolios]
    means = np.array([p[1] for p in portfolios])
    stds = np.array([p[2] for p in portfolios])
    cov_matrix = np.diag(stds**2)
    
    # Run quantum optimization
    backend_name = "ibm_brisbane"  
    quantum_weights = run_vqe_portfolio_optimization(means, cov_matrix, backend_name)
    
    # Find closest portfolio
    closest_portfolio = min(enumerate(all_weights), 
                         key=lambda x: np.linalg.norm(x[1] - quantum_weights))
    q_index = closest_portfolio[0]
    q_weights, q_return, q_risk = portfolios[q_index]
    
    print("\nQuantum Optimal Portfolio:")
    for t, w in zip(tickers, np.round(q_weights, 3)):
        print(f"  {t}: {w}")
    print(f" Quantum Return: {q_return:.4f}")
    print(f" Quantum Risk: {q_risk:.4f}")
    
    # Classical comparison
    bestClassicalIndex, (c_weights, c_return, c_risk) = classicalMaxPortfolio(portfolios)
    print("\nClassical Optimal Portfolio:")
    for t, w in zip(tickers, np.round(c_weights, 3)):
        print(f"  {t}: {w}")
    print(f" Classical Return: {c_return:.4f}")
    print(f" Classical Risk: {c_risk:.4f}")
    
    print(f"\n Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()



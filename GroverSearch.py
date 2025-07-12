# Built-in modules
import math
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def monteCarloPortfolios(numPortfolios, numAssets, numSimulations=100):
    """
    Simulate portfolio returns using Monte Carlo sampling.
    """
    np.random.seed(42)
    means = np.random.normal(0.1, 0.05, numAssets)
    cov = np.random.rand(numAssets, numAssets)
    cov = np.dot(cov, cov.transpose())  # Make positive semi-definite

    portfolioResults = []

    for _ in range(numPortfolios):
        weights = np.random.dirichlet(np.ones(numAssets))
        returns = []
        for _ in range(numSimulations):
            ret = np.random.multivariate_normal(means, cov)
            portfolioReturn = np.dot(weights, ret)
            returns.append(portfolioReturn)
        avgReturn = np.mean(returns)
        portfolioResults.append((weights, avgReturn))

    return portfolioResults

def getMarkedStates(portfolios, threshold=0.12):
    """
    Convert qualifying portfolio indices to bitstrings.
    """
    marked = []
    numBits = math.ceil(math.log2(len(portfolios)))
    for i, (weights, avgReturn) in enumerate(portfolios):
        if avgReturn >= threshold:
            bitString = format(i, f"0{numBits}b")
            marked.append(bitString)
    return marked

def groverOracle(markedStates):
    """
    Constructs a Grover oracle that marks the target bitstrings.
    """
    if not isinstance(markedStates, list):
        markedStates = [markedStates]

    numQubits = len(markedStates[0])
    qc = QuantumCircuit(numQubits)

    for target in markedStates:
        revTarget = target[::-1]
        zeroIndices = [i for i in range(numQubits) if revTarget[i] == '0']
        qc.x(zeroIndices)
        qc.compose(MCMT(ZGate(), numQubits - 1, 1), inplace=True)
        qc.x(zeroIndices)

    return qc

def runGroverSearch(portfolioSim, markedStates, backend):
    numQubits = math.ceil(math.log2(len(portfolioSim)))
    oracle = groverOracle(markedStates)
    groverOp = GroverOperator(oracle)

    optimalIters = math.floor(
        math.pi / (4 * math.asin(math.sqrt(len(markedStates) / (2**numQubits))))
    )

    qc = QuantumCircuit(numQubits)
    qc.h(range(numQubits))
    qc.compose(groverOp.power(optimalIters), inplace=True)
    qc.measure_all()

    target = backend.target
    passManager = generate_preset_pass_manager(target=target, optimization_level=3)
    transpiledCircuit = passManager.run(qc)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 10000
    result = sampler.run([transpiledCircuit]).result()
    counts = result[0].data.meas.get_counts()

    return counts, qc

# -----------------------------------------------------
# Step 5: Run Everything
# -----------------------------------------------------

def main():
    # Parameters
    numPortfolios = 16  # Requires 4 qubits
    numAssets = 4
    returnThreshold = 0.12

    print("Simulating portfolios.")
    portfolios = monteCarloPortfolios(numPortfolios, numAssets)
    markedStates = getMarkedStates(portfolios, returnThreshold)

    if not markedStates:
        print("No portfolios found above threshold.")
        return

    print(f"Marked states above {returnThreshold*100:.1f}% return): {markedStates}")

    print("Connecting to IBM Quantum services.")
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"✅ Using backend: {backend.name}")

    print("Running Grover search.")
    counts, qc = runGroverSearch(portfolios, markedStates, backend)

    print("Measurement results:")
    for outcome, count in counts.items():
        print(f"{outcome[::-1]} (index {int(outcome[::-1], 2)}): {count} counts")

    # Decode most probable result
    bestState = max(counts, key=counts.get)
    bestIndex = int(bestState[::-1], 2)
    weights, avgReturn = portfolios[bestIndex]

    print("Optimal Portfolio:")
    print(f"Weights: {np.round(weights, 3)}")
    print(f"Expected Return: {avgReturn:.4f}")

if __name__ == "__main__":
    main()

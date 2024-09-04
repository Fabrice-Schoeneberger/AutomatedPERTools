from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError
from qiskit_aer.noise.errors import pauli_error

# Create a quantum circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

if qc:
    print("true")
else:
    print("false")

# Define a custom crosstalk error
crosstalk_error = pauli_error([('X', 0.01), ('I', 0.99)])  # Example crosstalk error

# Create a noise model and add the crosstalk error to specific qubits
noise_model = NoiseModel()
noise_model.add_quantum_error(crosstalk_error, ['h'], [2])  # Apply to qubit 2 when 'h' gate is applied to qubit 0

# Simulate the circuit with the noise model
simulator = AerSimulator(noise_model=noise_model)
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit).result()

# Get the results
counts = result.get_counts()
print(counts)

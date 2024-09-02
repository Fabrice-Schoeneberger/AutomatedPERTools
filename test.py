from qiskit import QuantumCircuit
import random
import numpy as np

# Create a fresh quantum circuit with one qubit
qc = QuantumCircuit(1)

# Randomly choose and apply a single-qubit gate
random.choice([
    qc.h, qc.x, qc.y, qc.z, qc.s, qc.t, 
    lambda q: qc.rx(np.pi/2, q), 
    lambda q: qc.ry(np.pi/2, q), 
    lambda q: qc.rz(np.pi/2, q),
    qc.i
])(0)

# Draw the circuit
qc.draw()

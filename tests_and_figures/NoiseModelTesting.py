from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import pauli_basis, Pauli, PTM, SuperOp
from qiskit.providers.aer.noise import NoiseModel, pauli_error, amplitude_damping_error
from qiskit.providers.fake_provider import FakeVigoV2
from random import random, choices
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import nnls
import logging

import os
import sys

def empty_log():
    with open('experiment.log', 'w'):
        pass
empty_log()

sys.path.append(os.path.join(os.getcwd(), "pauli_lindblad_per"))
#sys.path.append("../pauli_lindblad_per")
from tomography.experiment import SparsePauliTomographyExperiment as tomography

plt.style.use("ggplot")

num = 4 #Number of errors to simulate. Decreasing this number produces more widely
        #varying fidelities, which is better for benchmarking the model

amp_damp = amplitude_damping_error(.01, .1)

errorops = choices(pauli_basis(2), k=num) #choose random pauli errors
errorprobs = [random()*.1/num for op in errorops] #assign random probabilities

#create normalized error model
twoqubit_error = pauli_error([(op, p) for op,p in zip(errorops, errorprobs)]+[(Pauli("II"), 1-sum(errorprobs))])
twoqubit_error = twoqubit_error.compose(amp_damp)
noise_model = NoiseModel()

#add error model to two-qubit gates.
noise_model.add_all_qubit_quantum_error(twoqubit_error, ['cx','cz'])

real_errors = {}

n = 2
channel = twoqubit_error.to_quantumchannel().data
twirled_channel = np.zeros([4**n,4**n])

for m in pauli_basis(n):
    p = np.kron(np.conjugate(m.to_matrix()),m.to_matrix())
    twirled_channel = np.add(twirled_channel, 1/4**n * p @ channel @ p)
transfer_matrix = PTM(SuperOp(twirled_channel)).data

real_errors = {p:1-transfer_matrix[i][i].real for i,p in enumerate(pauli_basis(n).to_labels())}
paulis = pauli_basis(n).to_labels()

def executor(circuits): #define noisy executor
    return Aer.get_backend("qasm_simulator").run(circuits, noise_model = noise_model, shots = 500).result().get_counts()

#Define circuit
qc = QuantumCircuit(3)
qc.cx(0,1)
#qc.draw()
def print_log_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                print(line, end='')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    empty_log()

def main():
    experiment = tomography(circuits = [qc], inst_map = [0,1,2], backend = FakeVigoV2())
    logger = logging.getLogger("experiment")
    print("\n")
    print_log_file("experiment.log")
    
    experiment.generate(samples = 32, single_samples = 250, depths = [2,4,8,16])

    print("\n")
    print_log_file("experiment.log")
    #run circiuts
    experiment.run(executor)

    print("\n")
    print_log_file("experiment.log")
    #return noise model
    noisedataframe = experiment.analyze()

    #get single circuit layer in noise model
    layer = experiment.analysis.get_layer_data(0)

print("\n")
main()
print_log_file("experiment.log")
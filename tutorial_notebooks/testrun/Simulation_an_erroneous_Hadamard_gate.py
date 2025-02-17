from random import random, choices
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error, amplitude_damping_error)
from qiskit.quantum_info import Pauli, pauli_basis, Operator
import numpy as np

p_error = 0.2
phase_flip = pauli_error([("Z", p_error), ("I", 1 - p_error)])
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(phase_flip, ['t'])

averages = []
shots = 1024
from qiskit import transpile, QuantumCircuit
for i in range(10000):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.sdg(0)
    circuit.h(0)
    circuit.measure_all()
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    results = backend.run([circuit], shots=shots, noise_model=noise_model).result().get_counts()
    average = (shots+results.get('0',0)-results.get('1',0))/(shots*2)
    averages.append(average)
    print(i, end='\r')
    #print(results)
    #print(average)

import numpy as np

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

from scipy.stats import norm

average = sum(averages)/len(averages)

# Calculate histogram

count, bins, _ = plt.hist(averages, bins=30, density=True, alpha=0.6, color='skyblue')
plt.xlim(0.65,0.9)
#plt.ylim(0,1)
plt.title('Simulation an erroneous Circuit', fontsize=24)

plt.xlabel('Expectation value', fontsize=20)
plt.ylabel('Probability density', fontsize=20)
plt.subplots_adjust(bottom=0.15, left=0.125)
plt.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig("test0.png")


# Fit a normal distribution to the data

mean, std = norm.fit(averages)



# Plot the PDF of the fitted normal distribution

x = np.linspace(min(bins)*0.98, max(bins)*1.02, 1000)

pdf = norm.pdf(x, mean, std)

plt.plot(x, pdf, 'k', linewidth=2, label=f'Fit parameters; mean:{mean:.4f}; std: {std:.4f}²)')



# Add labels and legend

plt.savefig("test1.png")
#plt.show(block=False)

y_average = norm.pdf(average, mean, std)

#y_ideal = norm.pdf(0.854232421875, mean, std)



plt.scatter([average], [y_average], color='red', zorder=5)  # Mark points
#plt.scatter([0.854232421875], [y_average/2], color='red', zorder=5)  # Mark points

plt.text(average, y_average, 'Measured average', color='black', fontsize=20, ha='center', va='bottom')

#plt.text(0.854232421875, y_average/2, 'Ideal value', color='black', fontsize=20, ha='center', va='bottom')
plt.vlines([average], [0], [y_average], colors='red', linestyles='dotted')
#plt.vlines([0.854232421875], [0], [y_average/2], colors='red', linestyles='dotted')

#plt.legend()
print(mean, std)
plt.savefig("test2.png")
#plt.show(block=False)


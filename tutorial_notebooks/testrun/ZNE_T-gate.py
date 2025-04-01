from random import random, choices
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.quantum_info import Pauli, pauli_basis, Operator
import numpy as np

p_error = 0.2
phase_flip = pauli_error([("Z", p_error), ("I", 1 - p_error)])
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(phase_flip, ['t'])
averages = []
averages2 = []
shots = 1024
samplesize = 10000
from qiskit import transpile, QuantumCircuit
for i in range(samplesize):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.sdg(0)
    circuit.h(0)
    circuit.measure_all()
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    results = backend.run([circuit], shots=shots, noise_model=noise_model).result().get_counts()
    average = 1-(shots+results.get('0',0)-results.get('1',0))/(shots*2)
    averages.append(average)
    print(i, end='\r')
print(samplesize)
for i in range(samplesize):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.t(0)
    circuit.t(0)
    circuit.t(0)
    circuit.sdg(0)
    circuit.sdg(0)
    circuit.h(0)
    circuit.measure_all()
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    results = backend.run([circuit], shots=shots, noise_model=noise_model).result().get_counts()
    average = 1-(shots+results.get('0',0)-results.get('1',0))/(shots*2)
    averages2.append(average)
    print(i, end='\r')
print(samplesize)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

from scipy.stats import norm

# Function to plot histogram and Gaussian fit for a dataset
def plot_data(averages, color, label, alpha):
    average = sum(averages) / len(averages)

    # Calculate histogram
    count, bins, _ = plt.hist(averages, bins=30, density=True, alpha=alpha, color=color, label=label)

    # Fit a normal distribution to the data
    mean, std = norm.fit(averages)

    # Plot the PDF of the fitted normal distribution
    x = np.linspace(min(bins) * 0.98, max(bins) * 1.02, 1000)
    pdf = norm.pdf(x, mean, std)
    plt.plot(x, pdf, color='k', linewidth=2, label=f'{label} Fit: mean={mean:.4f}, std={std:.4f}')

    # Mark points and add text
    y_average = norm.pdf(average, mean, std)
    plt.scatter([average], [y_average], color='red', zorder=5)  # Mark average
    plt.text(average, y_average, f'{label}', color='black', fontsize=12, ha='center', va='bottom')
    plt.vlines([average], [0], [y_average], colors='red', linestyles='dotted')

# Plot for averages
plot_data(averages, 'skyblue', '$\\lambda = 1$', 0.9)

# Plot for averages2
plot_data(averages2, 'lightgreen', '$\\lambda = 1,959995$', 0.7)

# Extrapolate for averages0 based on error propagation
means = [np.mean(averages), np.mean(averages2)]
a = np.polyfit([0.2,0.391999], means, 1)
stds = [np.std(averages), np.std(averages2)]
mean = a[-1]
std = sum(stds)/len(stds)
x = np.linspace(mean-(std*4), mean+(std*4), 1000)
pdf = norm.pdf(x, mean, std)
plt.plot(x, pdf, color='r', linewidth=2, alpha=0.5)
y_average = norm.pdf(mean, mean, std)
plt.scatter([mean], [y_average], color='red', zorder=5)  # Mark average
plt.text(mean, y_average, f'Extrapolated Ideal', color='black', fontsize=12, ha='center', va='bottom')
plt.vlines([mean], [0], [y_average], colors='red', linestyles='dotted')

# Add red arrow
plt.annotate('', xy=(0.21, 24), xytext=(0.24, 24),
             arrowprops=dict(facecolor='red', shrink=0.05))

# Add labels, legend, and formatting
plt.title('Simulation an erroneous Circuit', fontsize=24)
plt.xlabel('Expectation Value', fontsize=20)
plt.xlim(0.06, 0.5)
plt.ylim(0, 35)
plt.ylabel('Probability Density', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.subplots_adjust(bottom=0.15, left=0.125)
#plt.legend(fontsize=12)

print(mean, std)

plt.show()

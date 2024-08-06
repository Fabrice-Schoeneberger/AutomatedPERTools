# %%
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.fake_provider import FakeVigoV2
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import argparse
import json


folder = os.getcwd()
while not folder.endswith("AutomatedPERTools"):
    folder = os.path.dirname(folder)
sys.path.append(os.path.join(folder, "pauli_lindblad_per"))

from tomography.experiment import SparsePauliTomographyExperiment as tomography
from primitives.pauli import QiskitPauli

plt.style.use("ggplot")

parser = argparse.ArgumentParser()
    
# Definiere ein Argument
parser.add_argument('--plusone', type=str, help='Turn plusone on or off')
parser.add_argument('--sum', type=str, help='Turn plusone on or off')

# Parse die Argumente
args = parser.parse_args()

# Zugriff auf die Variable
if str(args.plusone) == "True":
    tomography_connections = True
elif str(args.plusone) == "False":
    tomography_connections = False
else:
    print(str(args.plusone))
    raise TypeError()

if str(args.sum) == "True":
    sum_over_lambda = True
elif str(args.sum) == "False":
    sum_over_lambda = False
else:
    print(str(args.sum))
    raise TypeError()


namebase = args.plusone + args.sum
print(namebase)
# %%
backend = FakeVigoV2()

# %%
print("make trotter")
def trotterLayer(h,J,dt,n):
    trotterLayer = QuantumCircuit(2*n)
    trotterLayer.rx(dt*4*h, range(2*n))
    trotterLayer.cx(*zip(*[(2*i, 2*i+1) for i in range(n)]))
    trotterLayer.rz(-4*J*dt, [2*i+1 for i in range(n)])
    trotterLayer.cx(*zip(*[(2*i, 2*i+1) for i in range(n)]))
    trotterLayer.cx(*zip(*[(2*i+1, 2*i+2) for i in range(n-1)]))
    trotterLayer.rz(-4*J*dt, [2*i+2 for i in range(n-1)])
    trotterLayer.cx(*zip(*[(2*i+1, 2*i+2) for i in range(n-1)]))
    return trotterLayer

h = 1
J = -.15
dt = .2
n = 2

def maketrotterCircuit(s):
    tL = trotterLayer(h, J, dt, n)
    trotterCircuit = QuantumCircuit(n*2)
    for i in range(s):
        trotterCircuit = trotterCircuit.compose(tL)
        trotterCircuit.barrier()

    transpiled = transpile(trotterCircuit, backend)
    return transpiled

circuits = [maketrotterCircuit(i) for i in range(1,15)]
circuits[0].draw()

# %%
def executor(circuits):
    return backend.run(circuits).result().get_counts()

# %%
print("initialize experiment")
experiment = tomography(circuits = circuits, inst_map = [0,1,2,3,4], backend = backend, tomography_connections=tomography_connections, sum_over_lambda=sum_over_lambda)

# %%
print("generate circuits")
experiment.generate(samples = 64, single_samples = 1000, depths = [2,4,8,16])

# %%
print("run experiment")
experiment.run(executor)

# %%
print("analyse experiment")
noisedataframe = experiment.analyze()

# %%
perexp = experiment.create_per_experiment(circuits)

# %%
noise_strengths = [0,0.5,1,2]
expectations = ["ZIIII","IZIII","IIZII","IIIZI"]
print("do PER runs")
perexp.generate(expectations = expectations, samples = 1000, noise_strengths = noise_strengths)

# %% [markdown]
# ## PER

# %%
print("Run PER")
perexp.run(executor)

# %%
circuit_results = perexp.analyze()

# %%
results = []
results_errors = []
results_at_noise = []
results_at_noise_errors = []
for run in circuit_results:
    tot = 0
    tot_error = [0,0]
    tot_at_noise = [0 for _ in range(len(noise_strengths))]
    tot_at_noise_errors = [0 for _ in range(len(noise_strengths))]
    for op in expectations:
        #get the full per results
        expec = run.get_result(op).expectation
        tot += expec

        #get the corresponding fit-errors
        expec_error = run.get_result(op).expectation_error
        for i in range(0,len(tot_error)):
            tot_error[i] += expec_error[i]/len(expectations)

        #get the value at the different noise levels
        expec_at_noise = run.get_result(op).get_expectations()
        for i in range(0,len(tot_at_noise)):
            tot_at_noise[i] += expec_at_noise[i]/len(expectations)

        expec_at_noise_error = [run.get_result(op).get_std_of_strengths(strength) for strength in noise_strengths]
        for i in range(0,len(tot_at_noise)):
            tot_at_noise_errors[i] += expec_at_noise_error[i]/len(expectations)

        

    results.append(tot/len(expectations))
    results_errors.append(tot_error)
    results_at_noise.append(tot_at_noise)
    results_at_noise_errors.append(tot_at_noise_errors)

savi = {}
savi["results"] = results
savi["results_errors"] = results_errors
savi["results_at_noise"] = results_at_noise
savi["results_at_noise_errors"] = results_at_noise_errors

# %%
circuit_results[-1]._per_circ.overhead(0)

# %%
noisyresult = []
for circ in circuits:
    qc = circ.copy()
    qc.measure_all()
    count= backend.run(qc).result().get_counts()
    count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
    tot = 0
    for key in count.keys():
        num = sum([(-1)**bit for bit in key[:4]])
        tot += num*count[key]
    noisyresult.append(tot/(1024*n*2))

savi["noisyresult"] = noisyresult
# %%
res = []
for circ in circuits:
    qc = circ.copy()
    qc.measure_all()
    count= Aer.get_backend('qasm_simulator').run(qc).result().get_counts()
    count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
    tot = 0
    for key in count.keys():
        num = sum([(-1)**bit for bit in key[:4]])
        tot += num*count[key]
    res.append(tot/(1024*n*2))

savi["res"] = res
with open(namebase + '_arrays.json', 'w') as file:
    json.dump(savi, file)

plt.figure(figsize=(10,6))
for i, noise in enumerate(list(circuit_results)[0].get_result(expectations[0]).get_strengths()):
    plt.plot(range(1,15), [res[i] for res in results_at_noise], 'x', label='N='+str(noise))
    
plt.plot(range(1,15), res, 'o:', label="Trotter Simulation")
plt.plot(range(1,15), noisyresult, 'o', label="Unmitigated")
plt.errorbar(range(1,15), results, yerr=[[np.abs(res[1]) for res in results_errors],[np.abs(res[0]) for res in results_errors]],fmt='x', capsize=5, label="PER")

plt.ylim([-1.8,1.8])
plt.legend()
plt.title("Trotter Simulation with PER")
plt.xlabel("Trotter Steps")
plt.ylabel("Z Magnetization")
plt.savefig(namebase+"_Trotter_Sim_PER.png")

# %%
plt.figure(figsize=(10,6))
for i, noise in enumerate(list(circuit_results)[0].get_result(expectations[0]).get_strengths()):
    if noise == 0:
        plt.errorbar(range(1,15), [res[i] for res in results_at_noise], fmt='x', yerr=[res[i] for res in results_at_noise_errors], capsize=5, label='N='+str(noise))
    else:
        plt.plot(range(1,15), [res[i] for res in results_at_noise], 'x', label='N='+str(noise))
        
    
plt.plot(range(1,15), res, 'o:', label="Trotter Simulation")
plt.plot(range(1,15), noisyresult, 'o', label="Unmitigated")
plt.plot(range(1,15), results, 'x', label="PER")

plt.ylim([-1.8,1.8])
plt.legend()
plt.title("Trotter Simulation with PER")
plt.xlabel("Trotter Steps")
plt.ylabel("Z Magnetization")
plt.savefig(namebase+"_Trotter_Sim_n_0.png")

# %%
plt.figure(figsize=(10,6))
for i, noise in enumerate(list(circuit_results)[0].get_result(expectations[0]).get_strengths()):
    if noise == 0.5:
        plt.errorbar(range(1,15), [res[i] for res in results_at_noise], fmt='x', yerr=[res[i] for res in results_at_noise_errors], label='N='+str(noise), capsize=5)
    else:
        plt.plot(range(1,15), [res[i] for res in results_at_noise], 'x', label='N='+str(noise))
        
    
plt.plot(range(1,15), res, 'o:', label="Trotter Simulation")
plt.plot(range(1,15), noisyresult, 'o', label="Unmitigated")
plt.plot(range(1,15), results, 'x', label="PER")

plt.ylim([-1.8,1.8])
plt.legend()
plt.title("Trotter Simulation with PER")
plt.xlabel("Trotter Steps")
plt.ylabel("Z Magnetization")
plt.savefig(namebase+"_Trotter_Sim_n_05.png")

# %%
ax = circuit_results[0].get_result("ZIIII").plot()
plt.title('Expectation vs Noise Strength ZIII')
plt.xlabel("Noise Strength")
plt.ylabel("Expectation")
plt.savefig(namebase+"_Expectation_vs_Noise_Strength_ZIII.png")
circuit_results[0].get_result("IZIII").plot()
plt.xlabel("Noise Strength")
plt.ylabel("Expectation")
plt.title('Expectation vs Noise Strength IZII')
plt.savefig(namebase+"_Expectation_vs_Noise_Strength_IZII.png")
circuit_results[0].get_result("IIZII").plot()
plt.xlabel("Noise Strength")
plt.ylabel("Expectation")
plt.title('Expectation vs Noise Strength IIZI')
plt.savefig(namebase+"_Expectation_vs_Noise_Strength_IIZI.png")
circuit_results[0].get_result("IIIZI").plot()
plt.xlabel("Noise Strength")
plt.ylabel("Expectation")
plt.title('Expectation vs Noise Strength IIIZ')
plt.savefig(namebase+"_Expectation_vs_Noise_Strength_IIIZ.png")

# %% [markdown]
# ## Analysis

# %%
layer1 = experiment.analysis.get_layer_data(0)

# %%
layer1.graph((1,))

# %%
layer1.plot_infidelitites((0,),(1,),(0,1))

# %%
layer1.plot_coeffs((1,),(0,),(0,1))

# %%
import pickle
with open("graph.pickle", "wb") as f:
    pickle.dump(results, f)



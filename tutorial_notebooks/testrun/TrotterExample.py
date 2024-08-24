if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
        
    # Definiere ein Argument
    parser.add_argument('--plusone', '-p', help='Takes Neighboring qubits into account', default=False, action='store_true')
    parser.add_argument('--sum', '-s', help='Same as -p and turns sumation on over neighboring qubits', default=False, action='store_true')
    parser.add_argument('--pntsamples', type=int, help='How many samples in PNT? Default: 16', default=16)
    parser.add_argument('--pntsinglesamples', type=int, help='How many single samples in PNT? Default: 100', default=100)
    parser.add_argument('--persamples', type=int, help='How many samples in PER? Default: 100', default=100)
    parser.add_argument('--shots', type=int, help='How many shots? Default: 1024', default=1024)
    parser.add_argument('--backend', type=str, help='Which backend to use? Default: FakeVigoV2', default="FakeVigoV2")
    parser.add_argument('--setqubits', type=int, nargs='+', help='Which qubits to use?: Default: 0123 and transpile')

    #  Parse die Argumente
    args = parser.parse_args()

    # %%
    from qiskit import QuantumCircuit, Aer, transpile
    from matplotlib import pyplot as plt
    import os
    import sys
    import numpy as np
    import json
    import time
    tim = time.localtime()
    print("%s.%s. %s:%s" % (tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min))


    folder = os.getcwd()
    while not folder.endswith("AutomatedPERTools"):
        folder = os.path.dirname(folder)
    sys.path.append(os.path.join(folder, "pauli_lindblad_per"))

    from tomography.experiment import SparsePauliTomographyExperiment as tomography
    from primitives.pauli import QiskitPauli

    plt.style.use("ggplot")
    # %%

    import qiskit.providers.fake_provider as fake_provider # FakeMelbourneV2, FakeCasablancaV2, FakeVigoV2, FakeLagosV2, FakeGuadalupeV2, FakeGuadalupe, FakeGeneva
    # Zugriff auf die Variablen
    backend = fake_provider.FakeVigoV2()
    if args.backend != "FakeVigoV2":
        method = getattr(fake_provider, args.backend)
        backend = method()

    qubits = [0,1,2,3] #[9,10,11,12] for MelbourneV2
    num_qubits = 4
    if args.setqubits != None:
        if len(args.setqubits) != 4:
            raise Exception("Must be 4 qubits when given")
        qubits = args.setqubits
        num_qubits = backend.num_qubits
        
    tomography_connections = args.plusone
    sum_over_lambda = args.sum
    if sum_over_lambda:
        tomography_connections = True

    pntsamples = args.pntsamples
    pntsinglesamples = args.pntsinglesamples
    persamples = args.persamples
    shots = args.shots

    namebase = "" 
    print("Arguments where set as:")
    for arg_name, arg_value in vars(args).items():
        if arg_name == "setqubits" and arg_value == None:
            arg_value = "[0,1,2,3]_and_transpile"
        print("\t%s: %s" % (arg_name, str(arg_value)))
        namebase += str(arg_value) + "_"
    print("Namebase will be: " + namebase)
    # %%

    # %%
    print("make trotter")
    def trotterLayer(h,J,dt,n):
        n=2
        trotterLayer = QuantumCircuit(num_qubits)
        trotterLayer.rx(dt*4*h, qubits)
        trotterLayer.cx(*zip(*[(qubits[2*i], qubits[2*i+1]) for i in range(n)]))
        trotterLayer.rz(-4*J*dt, [qubits[2*i+1] for i in range(n)])
        trotterLayer.cx(*zip(*[(qubits[2*i], qubits[2*i+1]) for i in range(n)]))
        trotterLayer.cx(*zip(*[(qubits[2*i+1], qubits[2*i+2]) for i in range(n-1)]))
        trotterLayer.rz(-4*J*dt, [qubits[2*i+2] for i in range(n-1)])
        trotterLayer.cx(*zip(*[(qubits[2*i+1], qubits[2*i+2]) for i in range(n-1)]))
        return trotterLayer

    h = 1
    J = -.15
    dt = .2
    n = 2

    def maketrotterCircuit(s):
        tL = trotterLayer(h, J, dt, n)
        trotterCircuit = QuantumCircuit(num_qubits)
        for i in range(s):
            trotterCircuit = trotterCircuit.compose(tL)
            trotterCircuit.barrier()

        transpiled = transpile(trotterCircuit, backend)
        return transpiled

    circuits = [maketrotterCircuit(i) for i in range(1,15)]
    used_qubits = set()
    for circuit in circuits: 
        for c in circuit: #look at the commands
            for bit in c.qubits: #record which qubits they use
                used_qubits.add(bit.index) #and save those
    qubits = used_qubits
    print("Qubits set to ", qubits)
    #circuits[0].draw()

    # %%
    def executor(circuits):
        return backend.run(circuits, shots=shots).result().get_counts()

    # %%
    print("initialize experiment")
    experiment = tomography(circuits = circuits, inst_map = range(backend.num_qubits), backend = backend, tomography_connections=tomography_connections, sum_over_lambda=sum_over_lambda)

    # %%
    print("generate circuits")
    experiment.generate(samples = pntsamples, single_samples = pntsinglesamples, depths = [2,4,8,16])
    tim = time.localtime()
    print("%s.%s. %s:%s" % (tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min))
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
    expectations = []
    for q in qubits:
        expect = "I"*(backend.num_qubits) #15
        expect = expect[:q] + 'Z' + expect[q+1:]
        expectations.append("".join(reversed(expect)))
    print("do PER runs")
    tim = time.localtime()
    print("%s.%s. %s:%s" % (tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min))
    perexp.generate(expectations = expectations, samples = persamples, noise_strengths = noise_strengths)

    # %%
    print("Run PER")
    perexp.run(executor)

    # %%
    circuit_results = perexp.analyze()

    # %%
    results_errors = []
    results_at_noise = []
    results_at_noise_errors = []
    results = []

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
        count= backend.run(qc, shots=shots).result().get_counts()
        count = {tuple(int(k) for i, k in enumerate(key) if len(key)-1-i in qubits):count[key] for key in count.keys()} #not sure yet if this works for another qubits size, but I think it does
        tot = 0
        for key in count.keys():
            num = sum([(-1)**bit for bit in key])
            tot += num*count[key]
        noisyresult.append(tot/(shots*1024*n*2))

    savi["noisyresult"] = noisyresult
    # %%
    res = []
    for circ in circuits:
        qc = circ.copy()
        qc.measure_all()
        count= Aer.get_backend('qasm_simulator').run(qc, shots=shots).result().get_counts()
        count = {tuple(int(k) for i, k in enumerate(key) if len(key)-1-i in qubits):count[key] for key in count.keys()}
        tot = 0
        for key in sorted(count.keys()):
            num = sum([(-1)**bit for bit in key])
            tot += num*count[key]
        res.append(tot/(shots*1024*n*2))

    savi["res"] = res
    with open(namebase + '_arrays.json', 'w') as file:
        json.dump(savi, file)

    plt.figure(figsize=(10,6))
    for i, noise in enumerate(noise_strengths):
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
    for i, noise in enumerate(noise_strengths):
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
    for i, noise in enumerate(noise_strengths):
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
    for i in range (len(expectations)):
        ax = circuit_results[0].get_result(expectations[i]).plot()
        plt.title("Expectation vs Noise Strength " + expectations[i])
        plt.xlabel("Noise Strength")
        plt.ylabel("Expectation")
        plt.savefig(namebase+"_Expectation_vs_Noise_Strength_" + expectations[i] + ".png")

    # %% [markdown]
    # ## Analysis

    # %%
    #layer1 = experiment.analysis.get_layer_data(0)

    # %%
    #layer1.graph((1,))

    # %%
    #layer1.plot_infidelitites((0,),(1,),(0,1))

    # %%
    #layer1.plot_coeffs((1,),(0,),(0,1))

    # %%
    #import pickle
    #with open("graph.pickle", "wb") as f:
    #    pickle.dump(results, f)

    tim = time.localtime()
    print("%s.%s. %s:%s" % (tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min))



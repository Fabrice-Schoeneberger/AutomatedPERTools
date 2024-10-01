def print_time():
    import time
    tim = time.localtime()
    print("%02d.%02d. %02d:%02d:%02d" % (tim.tm_mday, tim.tm_mon, tim.tm_hour, tim.tm_min, tim.tm_sec))

def main():
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
    parser.add_argument('--cross', '-c', help='Simulates Cross Talk Noise', default=False, action='store_true')
    parser.add_argument('--allqubits', '-a', help='runs over all qubits in the tomography', default=False, action='store_true')
    parser.add_argument('--onlyTomography', help='Only does the tomography and then ends the program', default=False, action='store_true')
    parser.add_argument('--setqubits', type=int, nargs='+', help='Which qubits to use?: Default: 0123 and transpile')

    #  Parse die Argumente
    args = parser.parse_args()

    # %%
    import qiskit
    from qiskit import QuantumCircuit, Aer, transpile
    from matplotlib import pyplot as plt
    import os
    import sys
    import numpy as np
    import json
    print_time()


    folder = os.getcwd()
    i = 0
    parentfolder = "AutomatedPERTools"
    while not folder.endswith(parentfolder):
        folder = os.path.dirname(folder)
        i+=1
        if i == 50:
            raise Exception("Parent Folder not found. Is "+ parentfolder + " the correct name?")
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
    
    do_cross_talk_noise = args.cross
    tomography_connections = args.plusone
    sum_over_lambda = args.sum
    if sum_over_lambda:
        tomography_connections = True
    tomography_all_qubits = args.allqubits
    onlyTomography = args.onlyTomography

    pntsamples = args.pntsamples
    pntsinglesamples = args.pntsinglesamples
    persamples = args.persamples
    shots = args.shots

    # %%
    print("make trotter")
    def trotterLayer(h,J,dt,n):
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

    namebase = "" 
    print("Arguments where set as:")
    for arg_name, arg_value in vars(args).items():
        if arg_name == "setqubits" and arg_value == None:
            arg_value = str(qubits)
        print("\t%s: %s" % (arg_name, str(arg_value)))
        namebase += str(arg_value) + "_"
    namebase = namebase[:-1]
    os.makedirs(namebase, exist_ok=True)
    print("Namebase will be: " + namebase)
    namebase += "/"
    #circuits[0].draw()
    # %% Put gates in empty qubits
    """ if True:
        import random
        for i in range(backend.num_qubits):
            if i not in used_qubits:
                for circ in circuits:
                    for j in range(random.randint(1,9)):
                        random.choice([
                            circ.h, circ.x, circ.y, circ.z, circ.s, circ.t, 
                            lambda q: circ.rx(np.pi/2, q), 
                            lambda q: circ.ry(np.pi/2, q), 
                            lambda q: circ.rz(np.pi/2, q),
                            circ.i
                        ])(i) """
                


    # %% Executer and Friends
    import multiprocessing
    def executor(circuits):
        if do_cross_talk_noise:
            manager = multiprocessing.Manager()
            #new_circuits = manager.list()
            new_circuits = []
            processes = []
            for circ in circuits:
                print("Opening process number ", circuits.index(circ))
                # Altering all circuits might take a while so let's do multiprocessing
                apply_cross_talk(circ, new_circuits)
                #process = multiprocessing.Process(target=apply_cross_talk, args=(circ, new_circuits))
                #processes.append(process)
                #process.start()
                #while len(multiprocessing.active_children()) > 100:
                #    time.sleep(10)

            #for process in processes:
            #    process.join()
            new_circuits = list(new_circuits)
        else:
            new_circuits =circuits.copy()

        return backend.run(new_circuits, shots=shots).result().get_counts()

    from primitives.circuit import QiskitCircuit
    def circuit_to_layers(qc: QiskitCircuit):
        layers = []
        inst_list = [inst for inst in qc if not inst.ismeas()] 

        #pop off instructions until inst_list is empty
        while inst_list:

            circ = qc.copy_empty() #blank circuit to add instructions
            layer_qubits = set() #qubits in the support of two-qubit clifford gates

            for inst in inst_list.copy(): #iterate through remaining instructions

                #check if current instruction overlaps with support of two-qubit gates
                #already on layer
                if not layer_qubits.intersection(inst.support()):
                    circ.add_instruction(inst) #add instruction to circuit and pop from list
                    inst_list.remove(inst)

                if inst.weight() == 2:
                    layer_qubits = layer_qubits.union(inst.support()) #add support to layer

            if circ: #append only if not empty
                layers.append(circ)

        return layers

    gate_triggered_cross_talk_chance = 0.000015 # Number by https://arxiv.org/pdf/2110.12570
    randomly_triggered_cross_talk_chance = 0.00000015 # Number guessed. No paper found so far. Assumption: Errer chance is way lower than for the gate triggered one
    def apply_cross_talk(circuit, new_circuits):
        import random

        if backend.num_qubits != len(circuit.qubits):
            circuit = transpile(circuit, backend)

        #rebuild the circuit
        circ = circuit.copy_empty_like()
        # Cut into layer, so you know all 2 qubit gates in one layer are parallel
        layers = circuit_to_layers(QiskitCircuit(circuit))
        for layer in layers:
            # Count up which qubit has the most instructions on it. Assume all instructions take the same time to resolve
            qubits = []
            for inst in layer:
                if inst.weight() == 1:
                    qubits.append(inst.instruction.qubits)
            # Save which qubit is the most gated and also how many gates it has
            most_gate_qubit = -1
            most_gate_qubit_count = -1
            for qubit in set(qubits):
                if qubits.count(qubit) > most_gate_qubit_count:
                    most_gate_qubit_count = qubits.count(qubit)
                    most_gate_qubit = qubit
            # rebuild the circuit
            for layer_inst in layer:
                inst = layer_inst.instruction
                circ.append(inst)
                if random.random() < gate_triggered_cross_talk_chance:
                    # colloect neighbors
                    if layer_inst.weight() == 1:
                        neighbors = set([bit for connection in backend.coupling_map if inst.qubits[0].index in connection for bit in connection]).remove(inst.qubits[0].index)
                    else:
                        neighbors = set([bit for connection in backend.coupling_map if inst.qubits[0].index in connection or inst.qubits[1].index in connection for bit in connection])
                        neighbors.remove(inst.qubits[0].index)
                        neighbors.remove(inst.qubits[1].index)
                    # Pick a random neighbor to apply the cross talk to
                    chosen_neighbor = random.choice(neighbors)
                    if layer_inst.weight() == 1:
                        circ.cx(inst.qubits[0].index, chosen_neighbor)
                    else:
                        # Pick qubit to make cross talk from
                        circ.cx(inst.qubits[random.choice([0,1])].index, chosen_neighbor)
                    
                # At every single qubit layer, determined by the most gated qubit, apply a random cnot gate, with chance=cross_talk_chance
                if inst.qubits == most_gate_qubit:
                    for edge in backend.coupling_map: # Every edge has a chance to send noise
                        if random.random() < randomly_triggered_cross_talk_chance: # And every qubit does that individual from one another
                            circ.cx(edge[0], edge[1])
                        if random.random() < randomly_triggered_cross_talk_chance: # Order could play a role, but the chance is to low,
                            circ.cx(edge[1], edge[0]) # that both hit at the same time, that it is ignored here
                            # Another thing to add is: Bydefault, these cnot gates are also noisy themself
                            # I consider this an upside, as cross talk noise is also not always the same.
            # Multiqubit gates take a longer time to resolve, on average 3 times longer.
            # This means, that there are 3 times more chances for a cross talk noise to occure
            for edge in backend.coupling_map: 
                if random.random() < 3*randomly_triggered_cross_talk_chance: 
                    circ.cx(edge[0], edge[1])
                if random.random() < 3*randomly_triggered_cross_talk_chance: 
                    circ.cx(edge[1], edge[0])
        new_circuits.append(circ)

    # %% initialize experiment
    print("initialize experiment")
    print_time()
    experiment = tomography(circuits = circuits, inst_map = [i for i in range(backend.num_qubits)], backend = backend, tomography_connections=tomography_connections, sum_over_lambda=sum_over_lambda, tomography_all_qubits=tomography_all_qubits)

    # %% generate circuits
    print("generate circuits")
    experiment.generate(samples = pntsamples, single_samples = pntsinglesamples, depths = [2,4,8,16])
    print_time()
    # %% run experiment
    print("run experiment")
    experiment.run(executor)

    # %% analyse experiment
    import pickle
    with open(namebase + "experiment.pickle", "wb") as f:
        pickle.dump(experiment, f)
    print("analyse experiment")
    print_time()
    noisedataframe = experiment.analyze()
    print("Create PER Experiment")
    print_time()
    with open(namebase + "noisedataframe.pickle", "wb") as f:
        pickle.dump(noisedataframe, f)
    if onlyTomography:
        return
    # %% create per experiment
    perexp = experiment.create_per_experiment(circuits)

    # %% generate per experiment and noise strengths
    noise_strengths = [0,0.5,1,2]
    expectations = []
    for q in qubits:
        expect = "I"*(backend.num_qubits) #15
        expect = expect[:q] + 'Z' + expect[q+1:]
        expectations.append("".join(reversed(expect)))
    print("do PER runs")
    print_time()
    perexp.generate(expectations = expectations, samples = persamples, noise_strengths = noise_strengths)

    # %%
    print("Run PER")
    print_time()
    perexp.run(executor)

    # %%
    print("Analyze PER")
    print_time()
    circuit_results = perexp.analyze()

    print("Delete Pickled PERrun Data")
    print_time()
    perexp.delete_pickles()

    # %%
    results_errors = []
    results_at_noise = []
    results_at_noise_errors = []
    results = []

    for run in circuit_results:
        tot = 0
        tot_error = 0
        tot_at_noise = [0 for _ in range(len(noise_strengths))]
        tot_at_noise_errors = [0 for _ in range(len(noise_strengths))]
        for op in expectations:
            #get the full per results
            expec = run.get_result(op).expectation
            tot += expec/len(expectations)

            #get the corresponding fit-errors
            expec_error = run.get_result(op).expectation_error
            tot_error += expec_error/len(expectations)

            #get the value at the different noise levels
            expec_at_noise = run.get_result(op).get_expectations()
            for i in range(0,len(tot_at_noise)):
                tot_at_noise[i] += expec_at_noise[i]/len(expectations)

            expec_at_noise_error = [run.get_result(op).get_std_of_strengths(strength) for strength in noise_strengths]
            for i in range(0,len(tot_at_noise)):
                tot_at_noise_errors[i] += expec_at_noise_error[i]/len(expectations)

            

        results.append(tot)
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
        count= backend.run(qc, shots=shots*persamples).result().get_counts()
        count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
        tot = 0
        for key in count.keys():
            num = sum([(-1)**bit for i, bit in enumerate(key) if len(key)-1-i in qubits])
            tot += num*count[key]
        noisyresult.append(tot/(shots*persamples*n*2))

    savi["noisyresult"] = noisyresult
    # %%
    res = []
    for circ in circuits:
        qc = circ.copy()
        qc.measure_all()
        count= Aer.get_backend('qasm_simulator').run(qc, shots=shots*persamples).result().get_counts()
        count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
        tot = 0
        for key in count.keys():
            num = sum([(-1)**bit for i, bit in enumerate(key) if len(key)-1-i in qubits])
            tot += num*count[key]
        res.append(tot/(shots*persamples*n*2))

    savi["res"] = res
    with open(namebase + '_arrays.json', 'w') as file:
        json.dump(savi, file)

    plt.figure(figsize=(10,6))
    for i, noise in enumerate(noise_strengths):
        plt.plot(range(1,15), [res[i] for res in results_at_noise], 'x', label='N='+str(noise))
        
    plt.plot(range(1,15), res, 'o:', label="Trotter Simulation")
    plt.plot(range(1,15), noisyresult, 'o', label="Unmitigated")
    plt.errorbar(range(1,15), results, yerr=results_errors, fmt='x', capsize=5, label="PER", color='#FFC0CB')

    plt.ylim([-1.8,1.8])
    plt.legend()
    plt.title("Trotter Simulation with PER")
    plt.xlabel("Trotter Steps")
    plt.ylabel("Z Magnetization")
    plt.savefig(namebase+"Trotter_Sim_PER.png")

    # %%
    #for i in range (len(expectations)):
    #    ax = circuit_results[0].get_result(expectations[i]).plot()
    #    plt.title("Expectation vs Noise Strength " + expectations[i])
    #    plt.xlabel("Noise Strength")
    #    plt.ylabel("Expectation")
    #    plt.savefig(namebase+"_Expectation_vs_Noise_Strength_" + expectations[i] + ".png")

    with open(namebase + "circuit_results.pickle", "wb") as f:
        pickle.dump(circuit_results, f)

    print_time()



if __name__ == "__main__":
    main()

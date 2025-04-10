import time
tim = time.time()
laststring = ""
last_time_string = ""
_noise_model = None
_twoqubit_error_template = None
_singlequbit_error_template = None
validate = None
def print_time(printstring=""):
    global tim, laststring, last_time_string
    import time
    local_tim = time.localtime()
    new_tim = time.time()
    time_difference = new_tim-tim
    tim = new_tim
    days = int(time_difference // (24 * 3600))
    time_difference %= (24 * 3600)
    hours = int(time_difference // 3600)
    time_difference %= 3600
    minutes = int(time_difference // 60)
    seconds = int(time_difference % 60)
    st = f" Time taken: {days:02}:{hours:02}:{minutes:02}:{seconds:02}"
    if not (laststring == "" and last_time_string == ""):
        print(laststring+last_time_string+st)    
    while len(printstring) < 24:
        printstring += " "
    printstring+="\t"
    laststring = printstring
    local_time_string = "%02d.%02d. %02d:%02d:%02d" % (local_tim.tm_mday, local_tim.tm_mon, local_tim.tm_hour, local_tim.tm_min, local_tim.tm_sec)
    last_time_string = local_time_string
    print(printstring+local_time_string+" Time taken: --:--:--:--", end="\r")

def get_index(qc, inst, i=None):
    if i != None:
        qubit = inst.qubits[i]
        index = -1
        for register in qc.qregs:  # Assuming you're using `QuantumCircuit`
            if qubit in register:
                index = register.index(qubit)
                break
        return index
    else:
        qubits = inst.qubits
        indexes = []
        for qubit in qubits:
            index = -1
            for register in qc.qregs:  # Assuming you're using `QuantumCircuit`
                if qubit in register:
                    index = register.index(qubit)
                    break
            indexes.append(index)
        return indexes

def get_backend(args, return_perfect=False, return_backend_qubits=False):
    if return_perfect or args is None or args.num_qubits == 0:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    else:
        from qiskit.providers.fake_provider import GenericBackendV2
        num = args.num_qubits
        coupling_map = [[i,i+1] for i in range(num-1)]+[[i+1,i] for i in range(num-1)]
        backend = GenericBackendV2(num_qubits=num, coupling_map=coupling_map)

    if not args is None and args.real_backend and not return_perfect:
        global validate
        if validate in None:
            validate = input("Are you sure you want to run on a REAL backend? Yes/no")
        if validate != "Yes":
            raise Exception("Program aborted. Real backend NOT used")
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            #if you have never run this on a machine you need to run this first on any program (not here): #service = QiskitRuntimeService.save_account(channel="ibm_quantum", token="...")
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=args.num_qubits)
            print(backend.name)
    if return_backend_qubits:
        return backend.num_qubits
    return backend

def get_noise_model():
    global _noise_model, _twoqubit_error_template, _singlequbit_error_template
    if not _noise_model is None:
        return (_noise_model, _twoqubit_error_template, _singlequbit_error_template)
    from random import random, choices
    from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
    from qiskit.quantum_info import Pauli, pauli_basis

    def remove_Identity(pauli_list):
        new_list = []
        for i in pauli_list:
            new_list.append(i)
        new_list.remove(Pauli('I'*len(pauli_list[0])))
        return new_list
    
    num = choices([3,4,5,6,7,8])[0]  #number of errors
    #singlequbit_errorops = choices(remove_Identity(pauli_basis(1)), k=num)
    #singlequbit_errorprobs = [random()*.1/(num*10) for op in singlequbit_errorops] #assign random probabilities
    #twoqubit_errorops = list(set(choices(remove_Identity(pauli_basis(2)), k=num))) #choose random pauli errors
    #twoqubit_errorprobs = [random()*.1/num for op in twoqubit_errorops] #assign random probabilities
    singlequbit_errorops = [Pauli('Y'), Pauli('Z'), Pauli('X')]
    singlequbit_errorprobs = [0.0018781587123864844, 0.00037277073796095685, 0.0015945514328675244]
    #twoqubit_errorops = [Pauli('ZX'), Pauli('YZ'), Pauli('IY'), Pauli('YY'), Pauli('XY')]
    #twoqubit_errorprobs = [0.00678362584027, 0.008802700270751796, 0.0032989083407153896, 0.01917444731546973, 0.019520575974201874]
    twoqubit_errorops = [Pauli('IX'), Pauli('YY'), Pauli('IZ'), Pauli('XZ'), Pauli('YI'), Pauli('ZX')]
    twoqubit_errorprobs = [0.00829, 0.00985, 0.00707, 0.00850, 0.00528, 0.00239]
    #create normalized error model
    #singlequbit_error_template = [(op, p) for op,p in zip(singlequbit_errorops, singlequbit_errorprobs)]+[(Pauli("I"), 1-sum(singlequbit_errorprobs))]
    singlequbit_error_template = [(Pauli("I"), 1)]
    singlequbit_error = pauli_error(singlequbit_error_template)
    twoqubit_error_template = [(op, p) for op,p in zip(twoqubit_errorops, twoqubit_errorprobs)]+[(Pauli("II"), 1-sum(twoqubit_errorprobs))]
    twoqubit_error = pauli_error(twoqubit_error_template)
    noise_model = NoiseModel()
    #add error model to two-qubit gates.
    #noise_model.add_all_qubit_quantum_error(singlequbit_error, ['id','rz','sx'])
    noise_model.add_all_qubit_quantum_error(twoqubit_error, ['cx','cz'])
    (_noise_model, _twoqubit_error_template, _singlequbit_error_template) = (noise_model, twoqubit_error_template, singlequbit_error_template)
    return (_noise_model, _twoqubit_error_template, _singlequbit_error_template)

def executor(circuits, backend, shots, noise_model=None):
    from qiskit_aer import AerSimulator
    from qiskit.providers.fake_provider import GenericBackendV2
    if backend.__class__ == GenericBackendV2 or backend.__class__ == AerSimulator:
        backend = get_backend(None, return_perfect=True)
        if not noise_model is None:
            results = backend.run(circuits, shots=shots, noise_model = noise_model).result().get_counts()
            #for i, r in enumerate(results):
            #    if not 1024 in r.values():
            #        pass
                    #print(circuits[i])
                    #print(r)
        else:
            results = backend.run(circuits, shots=shots).result().get_counts()
    elif hasattr(backend, 'configuration') and not backend.configuration().simulator:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(mode=backend)
        qc_job = sampler.run(circuits, shots=1024)
        results = qc_job.result()[0].data.meas.get_counts()
    else:
        raise AttributeError("Backend type is not recognized")

    return results

def calculate_with_simple_backend(circuits, shots, persamples, backend, qubits, n, noise_model, apply_cross_talk=False):
    res = []
    if apply_cross_talk:
        circuits = apply_cross_talk_proxy(circuits, backend)
    for circ in circuits:
        qc = circ.copy()
        qc.measure_all()
        count = executor([qc], backend, shots*persamples, noise_model)
        count = {tuple(int(k) for k in key):count[key] for key in count.keys()}
        tot = 0
        for key in count.keys():
            num = sum([(-1)**bit for i, bit in enumerate(key) if len(key)-1-i in qubits])
            tot += num*count[key]
        res.append(tot/(shots*persamples*n*2))
    return res

def get_error_for_circuit(circuit, twoqubit_error_template, singlequbit_error_template, backend):
    from qiskit.quantum_info import Pauli
    def mul_Pauli(pauli1, pauli2):
        result = pauli1.compose(pauli2)
        nophase = Pauli((result.z, result.x))
        return nophase

    num_qubits = backend.num_qubits
    identity_string = 'I'*num_qubits
    error_state = {Pauli(identity_string): 1}
    for inst in circuit:
        temp_error_state = {}
        indexes = get_index(circuit, inst)
        if len(indexes) == 1:
            index = indexes[0]
            for (op, p) in singlequbit_error_template:
                op = Pauli("".join(reversed(identity_string[:index] + str(op)+ identity_string[index+1:])))
                for og_op in error_state:
                    og_p = error_state[og_op]
                    new_op = mul_Pauli(op, og_op)
                    temp_error_state[new_op] = temp_error_state.get(new_op, 0) + p*og_p
        elif len(indexes) == 2:
            for (op, p) in twoqubit_error_template:
                temp_string = (identity_string[:indexes[0]] + "".join(reversed(str(op)))[0]+ identity_string[indexes[0]+1:])
                op = Pauli("".join(reversed(temp_string[:indexes[1]] + "".join(reversed(str(op)))[1]+ temp_string[indexes[1]+1:])))
                for og_op in error_state:
                    og_p = error_state[og_op]
                    new_op = mul_Pauli(op, og_op)
                    temp_error_state[new_op] = temp_error_state.get(new_op, 0) + p*og_p
        else:
            if inst.operation.name == 'barrier':
                continue
            raise Exception("Too many qubits")
        error_state = temp_error_state
    return [(op, error_state[op]) for op in error_state]

def find_used_qubits(circuits, backend):
    used_qubits = set()
    for circuit in circuits: 
        for inst in circuit: #look at the commands
            for j in range(len(inst.qubits)): #record which qubits they use
                used_qubits.add(get_index(circuit, inst, i=j)) #and save those
    return used_qubits

# %% Cross Talk Noise Functions
def apply_cross_talk_proxy(circuits, backend):
    print("", end="")
    import multiprocessing, os, pickle, uuid

    manager = multiprocessing.Manager()
    new_circuits = manager.list()
    lock = multiprocessing.Lock()
    # For some reason pickleing every circuit indiviually and sending it via the process is WAY slower than pickleing all at once and sending over the pickle file
    id = str(uuid.uuid4())
    pickle_file_name = "cross_talk_circuits_"+id+".pickle"
    with open(pickle_file_name, "wb") as f:
        pickle.dump(circuits, f)
    #if do_multiprocessing:
    len_circuits = len(circuits)
    for i in range(len_circuits):
        # Altering all circuits might take a while so let's do multiprocessing
        print("Apply Cross Talk Noise %s/%s" % (i+1, len_circuits), "; Number of active Threads: %03s" % len(multiprocessing.active_children()), "; List length: %s" % len(new_circuits), end='\r')
        process = multiprocessing.Process(target=apply_cross_talk_individual, args=(i, pickle_file_name, new_circuits, backend, lock))
        process.start()
        while len(multiprocessing.active_children()) > multiprocessing.cpu_count():
            print("test?",end='\r')
            pass
    while len(multiprocessing.active_children()) > 1:
        print("Apply Cross Talk Noise %s/%s" % (i+1, len_circuits), "; Number of active Threads: %03s" % len(multiprocessing.active_children()), "; List length: %s" % len(new_circuits), end='\r')
        pass
    print("Apply Cross Talk Noise %s/%s" % (i+1, len_circuits), "; Number of active Threads: %03s" % len(multiprocessing.active_children()), "; List length: %s" % len(new_circuits))
    os.remove(pickle_file_name)
    manager = None
    return list(new_circuits)

def circuit_to_layers(qc):
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

            if inst.weight() >= 2:
                layer_qubits = layer_qubits.union(inst.support()) #add support to layer

        if circ: #append only if not empty
            layers.append(circ)

    return layers

gate_triggered_cross_talk_chance = 0.000015 # Number by https://arxiv.org/pdf/2110.12570
randomly_triggered_cross_talk_chance = 0.00000015 # Number guessed. No paper found so far. Assumption: Error chance is way lower than for the gate triggered one
def apply_cross_talk_individual(i, pickle_file_name, new_circuits, backend, lock):
    import pickle, os, sys, random
    from qiskit import transpile
    from primitives.circuit import QiskitCircuit

    with open(pickle_file_name, 'rb') as file:
        circuits = pickle.load(file)
        circuit = circuits[i]
        circuits = []
    i = 0
    parentfolder = "AutomatedPERTools"
    folder = os.getcwd()
    while not folder.endswith(parentfolder):
        folder = os.path.dirname(folder)
        i+=1
        if i == 50:
            raise Exception("Parent Folder not found. Is "+ parentfolder + " the correct name?")
    sys.path.append(os.path.join(folder, "pauli_lindblad_per"))

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
                # collect neighbors
                if layer_inst.weight() == 1:
                    neighbors = list(set([bit for connection in backend.coupling_map if get_index(circ, inst, i=0) in connection for bit in connection]))
                    neighbors.remove(get_index(circ, inst, i=0))
                else:
                    neighbors = list(set([bit for connection in backend.coupling_map if get_index(circ, inst, i=0) in connection or get_index(circ, inst, i=1) in connection for bit in connection]))
                    neighbors.remove(get_index(circ, inst, i=0))
                    neighbors.remove(get_index(circ, inst, i=1))
                # Pick a random neighbor to apply the cross talk to
                chosen_neighbor = random.choice(neighbors)
                if layer_inst.weight() == 1:
                    circ.cx(get_index(circ, inst, i=0), chosen_neighbor)
                else:
                    # Pick qubit to make cross talk from
                    circ.cx(get_index(circ, inst, i=random.choice([0,1])), chosen_neighbor)
                
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
    circ.measure_all()
    with lock:
        new_circuits.append(circ)

# %% Start main(), Define the optional arguments
def main():
    import argparse
    parser = argparse.ArgumentParser()

    # PNT Arguments
    parser.add_argument('--pntsamples', type=int, help='How many samples in PNT? Default: 16', default=16)
    parser.add_argument('--pntsinglesamples', type=int, help='How many single samples in PNT? Default: 100', default=100)
    parser.add_argument('--depths', type=int, nargs='+', help='Decide the depths of the pnt-samples. Default: [2,4,8,16]')
    parser.add_argument('--onlyTomography', help='Only does the tomography and then ends the program', default=False, action='store_true')
    # PER Arguments
    parser.add_argument('--persamples', type=int, help='How many samples in PER? Default: 1000', default=1000)
    parser.add_argument('--noise_strengths', type=float, nargs='+', help='Decide the noise strengths at which PER will run. Default: [0.5,1,2]')
    parser.add_argument('--expectations', type=str, nargs='+', help='Decide the expectation values which whill be measured at PER. Default: Z on all used qubits')
    parser.add_argument('--do_multithreading', help='Activate the PER Multithreading. Warning: Could lead to complications', default=False, action='store_true')
    # General Arguments
    parser.add_argument('--shots', type=int, help='How many shots per circuit? Default: 1024', default=1024)
    parser.add_argument('--num_qubits', type=int, help='Define how many qubits the simulated backend should have. Layout: Line? Default: How ever many the biggest circuit has', default=0)
    parser.add_argument('--circuitfilename', type=str, help='Set the name of the file, that contains the function making you circuit. Default: circuit.py', default="circuits")
    parser.add_argument('--circuitfunction', type=str, help='Set the name of the function, that return your circuits. The function can only take a backend as input and has to return an array of circuits. Default: make_initial_Circuit', default="make_initial_Circuit")
    parser.add_argument('--cross', '-c', help='Simulates Cross Talk Noise', default=False, action='store_true')
    parser.add_argument('--make_plots', help='If true, generates standard PER and vZNE plots', default=False, action='store_true')
    parser.add_argument('--setqubits', type=int, nargs='+', help='Which qubits on the backend should be used? When set tries to fullfill request, but does not garantie. Default: 0123 and transpile')
    parser.add_argument('--foldername_extra', type=str, help='Attach something to the end of the foldernamebase', default="")
    parser.add_argument('--real_backend', help='Toggels the use of a real IBM Backend. You will be prompted an extra input to confirm this. Use QiskitRuntimeService.save_account(channel="ibm_quantum", token="<your_token>") first', default=False, action='store_true')

    #  Parse the Arguments
    args = parser.parse_args()

    # %% imports and system path appending
    import pickle
    from matplotlib import pyplot as plt
    import os
    import json
    import uuid
    import numpy as np
    

    import os
    import sys
    sys.path.append(os.path.join(folder, "pauli_lindblad_per"))


    from tomography.experiment import SparsePauliTomographyExperiment as tomography

    plt.style.use("ggplot")
    # %% Decipher the Arguments

    backend = get_backend(args)

    qubits = [0,1,2,3] #[9,10,11,12] for MelbourneV2
    if args.setqubits != None:
        if len(args.setqubits) != 4:
            raise Exception("Must be 4 qubits when given")
        qubits = args.setqubits
    
    depths = [2,4,8,16]
    if args.depths != None:
        depths = args.depths
    noise_strengths = [0.5,1,2]
    if args.noise_strengths != None:
        noise_strengths = args.noise_strengths
    do_cross_talk_noise = args.cross
    onlyTomography = args.onlyTomography

    pntsamples = args.pntsamples
    pntsinglesamples = args.pntsinglesamples
    persamples = args.persamples
    make_plots = args.make_plots
    shots = args.shots
    if args.expectations != None:
        expectations = args.expectations
    else: 
        qubits = [0,1,2,3]
        expectations = []
        for q in qubits:
            expect = "I"*4
            expect = expect[:q] + 'Z' + expect[q+1:]
            expectations.append("".join(reversed(expect)))
    (noise_model, twoqubit_error_template, singlequbit_error_template) = get_noise_model()

    # %% Make the initial Circuits
    print("")
    print("Make Circuits")
    n = 2

    if args.circuitfilename.endswith(".py"):
        circuitfilename = args.circuitfilename[:-3]
    else:
        circuitfilename = args.circuitfilename
    import importlib.util
    spec = importlib.util.spec_from_file_location(circuitfilename, f"{circuitfilename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_circuit = getattr(module, args.circuitfunction)
    circuits = get_circuit(backend)
    qubits = find_used_qubits(circuits, backend)
    max_qubits = 0
    for circuit in circuits: 
        max_qubits = max(max_qubits, max(qubits)+1)
    if args.num_qubits == 0:
        args.num_qubits = max_qubits
        backend = get_backend(args)
        circuits = get_circuit(backend)
    elif max_qubits > backend.num_qubits:
        raise Exception("Backend has to few qubits for a circuit. " +str(backend.num_qubits)+"/"+str(max_qubits)+ " given. Give more qubits with --num_qubits x")
    
    print("Qubits set to ", qubits)
    # %% Make namebase
    namebase = "" 
    print("Arguments where set as:")
    for arg_name, arg_value in vars(args).items():
        if arg_name == "setqubits" and arg_value == None:
            arg_value = str(qubits)
        if arg_name == "depths" and arg_value == None:
            arg_value = str(depths)
        print("\t%s: %s" % (arg_name, str(arg_value)))
        if str(arg_value) != "":
            namebase += str(arg_value) + "_"
    print("Backend name is: "+ str(backend.name))
    namebase += "_" + str(backend.name)
    namebase = namebase[:-1]
    #os.makedirs(namebase, exist_ok=True)
    print("Namebase will be: " + namebase)
    namebase += "/"

    # %% initialize experiment
    print_time("initialize experiment")
    experiment = tomography(circuits = circuits, inst_map = [i for i in range(get_backend(args, return_backend_qubits=True))], backend = backend)
    # %% generate PNT circuits
    print_time("generate circuits")
    experiment.generate(samples = pntsamples, single_samples = pntsinglesamples, depths = depths)

    # %% run PNT experiment
    print_time("run experiment")
    experiment.run(executor, shots, do_cross_talk=do_cross_talk_noise, apply_cross_talk=apply_cross_talk_proxy, noise_model=noise_model)

    # %% analyse PNT experiment.
    print_time("analyse experiment")
    noisedataframe = experiment.analyze()
    # %% Save all the data. End Tomography Only
    print_time("Saving data")
    id = str(uuid.uuid4())
    coeffs_dict_list = []
    infidelities_list = []
    cliff_layer_list = []
    for layer in experiment.analysis.get_all_layer_data():
        coeffs_dict = dict(layer.noisemodel.coeffs)
        infidelities = {term: 1-layer._term_data[term].fidelity for term in layer._term_data}
        coeffs_dict_list.append(coeffs_dict)
        infidelities_list.append(infidelities)
        cliff_layer_list.append(layer.layer._cliff_layer)
    os.makedirs("automatedPERrun_collection/PNT", exist_ok=True)
    os.makedirs("automatedPERrun_collection/PNT/"+ namebase, exist_ok=True)
    os.makedirs("automatedPERrun_collection/PNT/"+ namebase + "coeffs/", exist_ok=True)
    os.makedirs("automatedPERrun_collection/PNT/"+ namebase + "infidelities/", exist_ok=True)
    os.makedirs("automatedPERrun_collection/PNT/"+ namebase + "cliff_layer/", exist_ok=True)
    with open("automatedPERrun_collection/PNT/" + namebase + "coeffs/" + str(id) + ".pickle", "wb") as f:
        pickle.dump(coeffs_dict_list, f)
    with open("automatedPERrun_collection/PNT/" + namebase + "infidelities/" + str(id) + ".pickle", "wb") as f:
        pickle.dump(infidelities_list, f)
    with open("automatedPERrun_collection/PNT/" + namebase + "cliff_layer/" + str(id) + ".pickle", "wb") as f:
        pickle.dump(cliff_layer_list, f)
    with open("automatedPERrun_collection/PNT/" + namebase + "noise_model.pickle", "wb") as f:
        pickle.dump((noise_model, twoqubit_error_template, singlequbit_error_template), f)
    with open("automatedPERrun_collection/PNT/" + namebase + "circuits.pickle", "wb") as f:
        pickle.dump(circuits, f)
        
    if onlyTomography:
        print_time("Tomography Ended")
        print("")
        return
    # %% create per experiment
    print_time("Create PER Experiment")
    perexp = experiment.create_per_experiment(circuits)

    # %% generate per experiment and noise strengths
    print_time("do PER runs")
    perexp.generate(expectations = expectations, samples_per_circuit = persamples, noise_strengths = noise_strengths, do_multithreading=args.do_multithreading)
    # %% Run PER
    print_time("Run PER")
    perexp.run(executor, shots, do_cross_talk=do_cross_talk_noise, apply_cross_talk=apply_cross_talk_proxy, noise_model=noise_model)

    # %% Analyze PER and Delete Pickled PERrun Data
    print_time("Analyze PER")
    circuit_results = perexp.analyze()

    print_time("Delete Pickled PERrun Data")
    perexp.delete_pickles()

    # %% Extract data
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


    # %% Calculate unmitigated error and without error
    noisyresult = calculate_with_simple_backend(circuits, shots, persamples, backend, qubits, n, noise_model, apply_cross_talk=do_cross_talk_noise)
    res = calculate_with_simple_backend(circuits, shots, persamples, get_backend(args, return_perfect=True), qubits, n, noise_model)

    savi["noisyresult"] = noisyresult
    savi["res"] = res

    # %% Plot PER
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#E6194B', '#3CB44B', '#0082C8', '#F58231', '#911EB4', '#FFD700', '#46F0F0', '#F032E6', '#A9A9A9']
    fontsize =20
    markers = ['D', 'X', 'p', '*', 'D', 'X', 'p', '*', 'D', 'X', 'p', '*']

    with open(namebase + 'PER_data/'+id+'.json', 'w') as file:
        json.dump(savi, file)
    os.makedirs("automatedPERrun_collection/PER/", exist_ok=True)
    os.makedirs(os.path.join("automatedPERrun_collection/PER/", str(persamples)+"_"+str(noise_strengths)), exist_ok=True)
    os.makedirs(os.path.join("automatedPERrun_collection/PER/", str(persamples)+"_"+str(noise_strengths), "rel_value"), exist_ok=True)
    os.makedirs(os.path.join("automatedPERrun_collection/PER/", str(persamples)+"_"+str(noise_strengths), "rel_error"), exist_ok=True)
    def make_PER_plots():
        for j in range(len(noise_strengths)+2): # 0 vZNE results, 1-x results at noise level
            plt.figure(figsize=(20,12))
            for i, noise in enumerate(noise_strengths):
                if j != 0 and j!= 1 and noise == noise_strengths[j-2]:
                    plt.errorbar(range(1,15), [res[i] for res in results_at_noise], fmt=markers[i]+'--', yerr=[res[i] for res in results_at_noise_errors], capsize=12*fontsize/20, label='N='+str(noise), color= colors[i], zorder=1, markersize=10*fontsize/20, linewidth=2*fontsize/20)
                else:
                    plt.plot(range(1,15), [res[i] for res in results_at_noise], markers[i], label='N='+str(noise), color= colors[i], zorder=2, markersize=10*fontsize/20)
                
            plt.plot(range(1,15), res, 'o:', label="Trotter Simulation", color= colors[len(noise_strengths)], zorder=2, markersize=10*fontsize/20, linewidth=2*fontsize/20)
            plt.plot(range(1,15), noisyresult, 'o', label="Unmitigated", color= colors[len(noise_strengths)+1], zorder=2, markersize=10*fontsize/20)
            if j == 1:
                plt.errorbar(range(1,15), results, yerr=[[np.abs(res) for res in results_errors]],fmt='s--', capsize=12*fontsize/20, label="PER", color= colors[len(noise_strengths)+2], zorder=1, markersize=10*fontsize/20, linewidth=2*fontsize/20)
            else:
                plt.plot(range(1,15), results, 's', label="PER", color= colors[len(noise_strengths)+2], zorder=2, markersize=10*fontsize/20)

            plt.ylim([-2.1,2.1])
            plt.legend(fontsize=fontsize, loc='upper right')
            plt.title("Trotter Simulation with PER", fontsize=fontsize*1.2)
            plt.xlabel("Trotter Steps", fontsize=fontsize)
            plt.ylabel("Z Magnetization", fontsize=fontsize)
            plt.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.savefig("automatedPERrun_collection/PER/"+str(persamples)+"_"+str(noise_strengths)+"/Trotter_Sim_PER"+str(j)+".png")
    def make_vZNE_fit_plots(i):
        fontsize = 30
        def plot(perdat):
            """Plots the expectation values against an exponential fit.
            """
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.errorbar(list(sorted(perdat["data"].keys())), [perdat["data"][s]/perdat["counts"][s] for s in list(sorted(perdat["data"].keys()))], yerr=[np.std(perdat["dataStatistic"][strength]) for strength in list(sorted(perdat["data"].keys()))],  linestyle = "None", marker = "o", color = "tab:blue", capsize=12*fontsize/20, markersize=12*fontsize/20)
            a = perdat["expectation"]
            b = perdat["b"]
            xlin = np.linspace(0, max(list(sorted(perdat["data"].keys()))), 100)
            ax.plot(xlin, [a*np.exp(b*x) for x in xlin], color = "tab:blue", linestyle="--", linewidth=2*fontsize/20)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            
        for ex in expectations:
            # Change the number in circuit_results[NUMBER] to show different circuits fits
            ax = plot(circuit_results[i].get(ex, None))
            # If the error becomes to big, uncomment the next line to still see anything but a straight line
            #plt.ylim(-1.5,1.5)
            plt.title('Expectation vs Noise Strength '+str(ex), fontsize=fontsize*1.2)
            plt.xlabel("Noise Strength", fontsize=fontsize)
            plt.ylabel("Expectation", fontsize=fontsize)
            plt.savefig("automatedPERrun_collection/PER/"+str(persamples)+"_"+str(noise_strengths)+"/Expectation_vs_Noise_Strength_"+str(ex)+".png")

    if make_plots:
        make_PER_plots()
        make_vZNE_fit_plots(0)
    rel_value = [abs(res[i]-results[i])/abs(res[i]-noisyresult[i]) for i, _ in enumerate(res)]
    rel_error = [results_errors[i]/abs(res[i]-noisyresult[i]) for i, _ in enumerate(res)]
    import pickle
    with open(os.path.join("automatedPERrun_collection/PER", namebase, "rel_value/") + str(id) + ".pickle", "wb") as f:
        pickle.dump(rel_value, f)
    with open(os.path.join("automatedPERrun_collection/PER", namebase, "rel_error/") + str(id) + ".pickle", "wb") as f:
        pickle.dump(rel_error, f)


    with open(namebase + "circuit_results.pickle", "wb") as f:
        pickle.dump(circuit_results, f)

    print_time()
    print("")


# %% Start Program
if __name__ == "__main__":
    print_time("Starting")
    main()

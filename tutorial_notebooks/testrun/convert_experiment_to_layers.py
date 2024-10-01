import os
import sys
import shutil

home_folder = os.getcwd()
while not home_folder.endswith("AutomatedPERTools"):
    home_folder = os.path.dirname(home_folder)
sys.path.append(os.path.join(home_folder, "pauli_lindblad_per"))
import pickle

def decipher_name(namebase):
    dic = dict()
    split_namebase = namebase.split("_")
    for i,j in enumerate(split_namebase):
        #print(i, j)
        pass
    #parser.add_argument('--plusone', '-p', help='Takes Neighboring qubits into account', default=False, action='store_true')
    dic["tomography_connections"] = (split_namebase[0] == "True")
    #parser.add_argument('--sum', '-s', help='Same as -p and turns sumation on over neighboring qubits', default=False, action='store_true')
    dic["sum_over_lambda"] = (split_namebase[1] == "True")
    if dic["sum_over_lambda"]:
        dic["tomography_connections"] = True
    #parser.add_argument('--pntsamples', type=int, help='How many samples in PNT? Default: 16', default=16)
    dic["pntsamples"] = int(split_namebase[2])
    #parser.add_argument('--pntsinglesamples', type=int, help='How many single samples in PNT? Default: 100', default=100)
    dic["pntsinglesamples"] = int(split_namebase[3])
    #parser.add_argument('--persamples', type=int, help='How many samples in PER? Default: 100', default=100)
    dic["persamples"] = int(split_namebase[4])
    #parser.add_argument('--shots', type=int, help='How many shots? Default: 1000', default=1000)
    dic["shots"] = int(split_namebase[5])
    #parser.add_argument('--backend', type=str, help='Which backend to use? Default: FakeVigoV2', default="FakeVigoV2")
    dic["backend"] = split_namebase[6]
    import qiskit.providers.fake_provider as fake_provider
    backend = fake_provider.FakeVigoV2()
    if split_namebase[6] != "FakeVigoV2":
        method = getattr(fake_provider, split_namebase[6])
        backend = method()

    #parser.add_argument('--cross', '-c', help='Simulates Cross Talk Noise', default=False, action='store_true')
    dic["do_cross_talk_noise"] = (split_namebase[7] == "True")
    #parser.add_argument('--allqubits', '-a', help='runs over all qubits in the tomography', default=False, action='store_true')
    dic["allqubits"] = (split_namebase[8] == "True")
    #parser.add_argument('--onlyTomography', help='Only does the tomography and then ends the program', default=False, action='store_true')
    dic["onlyTomography"] = (split_namebase[9] == "True")

    #parser.add_argument('--setqubits', type=int, nargs='+', help='Which qubits to use?: Default: 0123 and transpile')
    import ast
    dic["qubits"]= ast.literal_eval(split_namebase[10])
    dic["num_qubits"] = len(dic["qubits"])
    #if split_namebase[7] == '[0,1,2,3]' and split_namebase[8] == 'and' and split_namebase[9] == 'transpile':
    return dic

def model_terms(layer, *list):
    return layer._model_terms(list)

folders = [f for f in os.listdir() if os.path.isdir(f)]
filtered_folders = [f for f in folders if "FakeVigoV2" in f and not decipher_name(f)["do_cross_talk_noise"] and decipher_name(f)["pntsamples"]== 256]

for folder in filtered_folders:
    print(folder)
    if "layer_data.pickle" in os.listdir(folder):
        os.makedirs("server_run_collection/"+folder, exist_ok=True)
        shutil.move(folder+"/layer_data.pickle", "server_run_collection/"+folder)
    if "experiment.pickle" in os.listdir(folder):
        if not folder in os.listdir("server_run_collection"):
            os.makedirs("server_run_collection/"+folder, exist_ok=True)
        if not "layer_data.pickle" in os.listdir("server_run_collection/"+folder):
            if "experiment.pickle" in os.listdir(folder):
                os.makedirs("server_run_collection/"+folder, exist_ok=True)
                with open(folder + "/experiment.pickle", "rb") as f:
                    experiment = pickle.load(f)
                experiment.analyze()
                def model_terms(layer, *list):
                    return layer._model_terms(list)
                
                layer = experiment.analysis.get_layer_data(0)
                coeffs_dict0 = dict(layer.noisemodel.coeffs)
                infidelities0 = {term: 1-layer._term_data[term].fidelity for term in layer._term_data}

                layer = experiment.analysis.get_layer_data(1)
                coeffs_dict1 = dict(layer.noisemodel.coeffs)
                infidelities1 = {term: 1-layer._term_data[term].fidelity for term in layer._term_data}

                os.makedirs("server_run_collection/"+folder, exist_ok=True)
                with open("server_run_collection/" + folder + "/coeffs.pickle", "wb") as f:
                    pickle.dump([coeffs_dict0, coeffs_dict1], f)
                with open("server_run_collection/" + folder + "/infidelities.pickle", "wb") as f:
                    pickle.dump([infidelities0, infidelities1], f)
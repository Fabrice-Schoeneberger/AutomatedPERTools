from scipy.optimize import nnls
import numpy as np
from matplotlib import pyplot as plt

from framework.noisemodel import NoiseModel
from tomography.layerlearning import LayerLearning
from tomography.termdata import TermData, COLORS
from primitives.circuit import Circuit
from tomography.benchmarkinstance import BenchmarkInstance, SINGLE, PAIR

import logging
from itertools import cycle

logger = logging.getLogger("experiment")

class LayerNoiseData:
    """This class is responsible for aggregating the data associated with a single layer,
    processing it, and converting it into a noise model to use for PER"""

    def __init__(self, layer : LayerLearning, sum_over_lambda=False, plusone = set(), used_qubits = None):
        self._term_data = {} #keys are terms and the values are TermDatas
        self.layer = layer
        self.sum_over_lambda=sum_over_lambda
        self.plusone = plusone
        self.used_qubits = used_qubits
        self.pair_sim_meas_dic = {}
        self.single_sim_meas_dic = {}
        for pauli in layer._procspec.model_terms:
            pair = layer.pairs[pauli]
            self._term_data[pauli] = TermData(pauli, pair)

    def sim_meas(self, pauli):
        """Given an instance and a pauli operator, determine how many terms can be measured"""
        return [term for term in self.layer._procspec.model_terms if pauli.simultaneous(term)]

    def single_sim_meas(self, pauli, prep):
        return [term for pair,term in self.layer.single_pairs if pauli.simultaneous(term) and prep.simultaneous(pair)]

    def add_expectations(self):
        for inst in self.layer.instances:
            self.add_expectation(inst)

    def add_expectation(self, inst : BenchmarkInstance):
        """Add the result of a benchmark instance to the correct TermData object"""

        basis = inst.meas_basis
        prep = inst.prep_basis
        single_sim_meas_dic = {}

        if inst.type == SINGLE:

            if not basis in single_sim_meas_dic: # This is always true
                self.single_sim_meas_dic[basis] = self.single_sim_meas(basis, prep)
                single_sim_meas_dic[basis] = self.single_sim_meas(basis, prep)

            for pauli in single_sim_meas_dic[basis]:
                expectation = inst.get_expectation(pauli)
                self._term_data[pauli].add_single_expectation(expectation)

        elif inst.type == PAIR:

            if not basis in self.pair_sim_meas_dic: # This is always true
                self.pair_sim_meas_dic[basis] = self.sim_meas(basis)

            for pauli in self.pair_sim_meas_dic[basis]:
                #add the expectation value to the data for this term
                expectation = inst.get_expectation(pauli)
                self._term_data[pauli].add_expectation(inst.depth, expectation, inst.type)

        
    def fit_noise_model(self):
        """Fit all of the terms, and then use obtained SPAM coefficients to make degerneracy
        lifting estimates"""

        for term in self._term_data.values(): #perform all pairwise fits
            term.fit()
        
        for pair,pauli in self.layer.single_pairs:
            self._term_data[pauli].fit_single()
            pair_dat = self._term_data[pair]
            pair_dat.fidelity = pair_dat.fidelity**2/self._term_data[pauli].fidelity

        import pickle
        savefile = []
        for key in self.pair_sim_meas_dic:
            keydic = {}
            for pauli in self.pair_sim_meas_dic[key]:
                pair = self._term_data[pauli].pair
                plotdata = {}
                a,b = self._term_data[pauli]._fit()
                plotdata["a"] = a
                plotdata["b"] = b
                plotdata["depth"] = self._term_data[pauli].depths()
                plotdata["expectations"] = self._term_data[pauli].expectations()
                keydic[(pauli,pair)] = plotdata
            savefile.append((key, keydic))
        with open("termplot.pickle", "wb") as f:
            pickle.dump(savefile, f)

        
        logger.info("Fit noise model with following fidelities:") 
        logger.info([term.fidelity for term in self._term_data.values()])

        #get noise model from fits
        self.nnls_fit()

    def _issingle(self, term):
        return term.pauli != term.pair and term.pair in self._term_data
  
    
    def nnls_fit(self):
        """Generate a noise model corresponding to the Clifford layer being benchmarked
        for use in PER"""

        def sprod(a,b): #simplecting inner product between two Pauli operators
            return int(not a.commutes(b))
        
        def get_indexes(string):
            indexes = []
            for i, f in enumerate(string):
                if f != "I":
                    indexes.append(len(string)- i-1)
            return indexes

        F1 = [] #First list of terms
        F2 = [] #List of term pairs
        F1_mini = []
        fidelities = [] # list of fidelities from fits
        from qiskit.quantum_info import Pauli, pauli_basis
        predone = {Pauli('IIIIX'): np.float64(2.7604759922361666e-07), Pauli('IIIIY'): np.float64(2.7604759922361666e-07), Pauli('IIIIZ'): np.float64(2.7604759922361666e-07), Pauli('IIIXI'): np.float64(0.0977182333117168), Pauli('IIIYI'): np.float64(1.0528508498031108e-05), Pauli('IIIZI'): np.float64(0.09863571890753331), Pauli('IIXII'): np.float64(2.7604759922361666e-07), Pauli('IIYII'): np.float64(0.00086443562696914), Pauli('IIZII'): np.float64(-0.001108282488732959), Pauli('IXIII'): np.float64(2.7604759922361666e-07), Pauli('IYIII'): np.float64(2.7604759922361666e-07), Pauli('IZIII'): np.float64(2.7604759922361666e-07), Pauli('XIIII'): np.float64(2.7604759922361666e-07), Pauli('YIIII'): np.float64(2.7604759922361666e-07), Pauli('ZIIII'): np.float64(2.7604759922361666e-07), Pauli('IIIXX'): np.float64(0.09953459966910827), Pauli('IIIXY'): np.float64(0.09870129433840336), Pauli('IIIXZ'): np.float64(0.09760302120088082), Pauli('IIIYX'): np.float64(2.7604759922361666e-07), Pauli('IIIYY'): np.float64(2.7604759922361666e-07), Pauli('IIIYZ'): np.float64(2.7604759922361666e-07), Pauli('IIIZX'): np.float64(0.09825801372838039), Pauli('IIIZY'): np.float64(0.0966240932190986), Pauli('IIIZZ'): np.float64(0.10091994064607201), Pauli('IIXXI'): np.float64(0.10134730953374327), Pauli('IIXYI'): np.float64(-9.976518413612467e-06), Pauli('IIXZI'): np.float64(0.09825801372838039), Pauli('IIYXI'): np.float64(0.10252075870831678), Pauli('IIYYI'): np.float64(-0.001567284399190516), Pauli('IIYZI'): np.float64(0.09810262977677264), Pauli('IIZXI'): np.float64(0.10080612781242282), Pauli('IIZYI'): np.float64(-0.0030389328697042473), Pauli('IIZZI'): np.float64(0.10115570626810666), Pauli('IXXII'): np.float64(2.7604759922361666e-07), Pauli('IXYII'): np.float64(0.05137484385277957), Pauli('IXZII'): np.float64(0.05213632053555728), Pauli('IYXII'): np.float64(2.7604759922361666e-07), Pauli('IYYII'): np.float64(0.050871746205822244), Pauli('IYZII'): np.float64(0.050430599747674054), Pauli('IZXII'): np.float64(2.7604759922361666e-07), Pauli('IZYII'): np.float64(0.049877025904787264), Pauli('IZZII'): np.float64(0.050968294184473684), Pauli('XXIII'): np.float64(2.7604759922361666e-07), Pauli('XYIII'): np.float64(2.7604759922361666e-07), Pauli('XZIII'): np.float64(2.7604759922361666e-07), Pauli('YXIII'): np.float64(2.7604759922361666e-07), Pauli('YYIII'): np.float64(2.7604759922361666e-07), Pauli('YZIII'): np.float64(2.7604759922361666e-07), Pauli('ZXIII'): np.float64(2.7604759922361666e-07), Pauli('ZYIII'): np.float64(2.7604759922361666e-07), Pauli('ZZIII'): np.float64(2.7604759922361666e-07)}
        #predone = {Pauli('IIIIX'): np.float64(0.0), Pauli('IIIIY'): np.float64(0.0), Pauli('IIIIZ'): np.float64(0.0), Pauli('IIIXI'): np.float64(0.09999999999999987), Pauli('IIIYI'): np.float64(0.0), Pauli('IIIZI'): np.float64(0.09999999999999987), Pauli('IIXII'): np.float64(0.0), Pauli('IIYII'): np.float64(0.0), Pauli('IIZII'): np.float64(0.0), Pauli('IXIII'): np.float64(0.0), Pauli('IYIII'): np.float64(0.0), Pauli('IZIII'): np.float64(0.0), Pauli('XIIII'): np.float64(0.0), Pauli('YIIII'): np.float64(0.0), Pauli('ZIIII'): np.float64(0.0), Pauli('IIIXX'): np.float64(0.09999999999999987), Pauli('IIIXY'): np.float64(0.09999999999999987), Pauli('IIIXZ'): np.float64(0.09999999999999987), Pauli('IIIYX'): np.float64(0.0), Pauli('IIIYY'): np.float64(0.0), Pauli('IIIYZ'): np.float64(0.0), Pauli('IIIZX'): np.float64(0.09999999999999987), Pauli('IIIZY'): np.float64(0.09999999999999987), Pauli('IIIZZ'): np.float64(0.09999999999999987), Pauli('IIXXI'): np.float64(0.09999999999999987), Pauli('IIXYI'): np.float64(0.0), Pauli('IIXZI'): np.float64(0.09999999999999987), Pauli('IIYXI'): np.float64(0.09999999999999987), Pauli('IIYYI'): np.float64(0.0), Pauli('IIYZI'): np.float64(0.09999999999999987), Pauli('IIZXI'): np.float64(0.09999999999999987), Pauli('IIZYI'): np.float64(0.0), Pauli('IIZZI'): np.float64(0.09999999999999987), Pauli('IXXII'): np.float64(0.0), Pauli('IXYII'): np.float64(0.0), Pauli('IXZII'): np.float64(0.0), Pauli('IYXII'): np.float64(0.0), Pauli('IYYII'): np.float64(0.0), Pauli('IYZII'): np.float64(0.0), Pauli('IZXII'): np.float64(0.0), Pauli('IZYII'): np.float64(0.0), Pauli('IZZII'): np.float64(0.0), Pauli('XXIII'): np.float64(0.0), Pauli('XYIII'): np.float64(0.0), Pauli('XZIII'): np.float64(0.0), Pauli('YXIII'): np.float64(0.0), Pauli('YYIII'): np.float64(0.0), Pauli('YZIII'): np.float64(0.0), Pauli('ZXIII'): np.float64(0.0), Pauli('ZYIII'): np.float64(0.0), Pauli('ZZIII'): np.float64(0.0)}
        logger.info(self.used_qubits)
        for datum in self._term_data.values():
            pauli = datum.pauli
            indexes = get_indexes(str(pauli))
            skip = False
            for index in indexes:
                if datum.fidelity == 1:
                    logger.info("skip")
                    skip = True
                    break
            if not skip:
                F1_mini.append(datum.pauli)
            F1.append(datum.pauli)
            #fidelities.append(datum.fidelity)
            if str(datum.pauli) in ["IXYII","IYYII","IZYII","IXZII","IYZII","IZZII"]:
                fidelities.append(1-2.7604759922361666e-07)
            else:
                fidelities.append(1-predone.get(datum.pauli.pauli,2.7604759922361666e-07))
            print((datum.pauli, 1-predone.get(datum.pauli.pauli,2.7604759922361666e-07), fidelities[-1]))
            #If the Pauli is conjugate to another term in the model, a degeneracy is present
            if self._issingle(datum):
                F2.append(datum.pauli)
            else:
                pair = datum.pair
                F2.append(pair)
        #logger.info("LayerNoiseData")
        #logger.info(F1)
        #create commutativity matrices
        M1 = [[sprod(a,b) for a in F1] for b in F1]
        M2 = [[sprod(a,b) for a in F1] for b in F2]

        #check to make sure that there is no degeneracy
        if np.linalg.matrix_rank(np.add(M1,M2)) != len(F1):
            raise Exception("Matrix is not full rank, something went wrong!")
       
        #perform least-squares estimate of model coefficients and return as noisemodel 
        coeffs,_ = nnls(np.add(M1,M2), -np.log(fidelities)) 

        if self.sum_over_lambda:
            paulilength = len(F1[0].to_label())
            for qubit in self.plusone:
                logger.info([f.to_label() for f in F1])
                #Filter out all model terms with the extra qubit active with at least one connected qubit
                filtered_and_cut_list = [f for f in F1 if f.to_label()[paulilength-1-qubit]!='I' and (f.to_label()[:paulilength-1-qubit]+f.to_label()[paulilength-1-qubit+1:] != "I"*(paulilength-1))]
                #In case of 2 or more connected qubits we need to seperate them
                sorted_lists = dict()
                for f in filtered_and_cut_list:
                    for i, char in enumerate(f.to_label()):
                        #find which *other* qubits are uses and sort them into lists
                        if char != "I" and i != paulilength-1-qubit:
                            if not i in sorted_lists: #make the new list if it didn't exist so far
                                sorted_lists[i] = []
                            sorted_lists[i].append(f)
                            break
                for index in sorted_lists:
                    #for each connected main qubit, filter them by x,y,z and sum their coeffs up
                    for pauli in "XYZ":
                        x = sum([coeffs[F1.index(model_term)] for model_term in sorted_lists[index] if model_term.to_label()[index]==pauli])
                        #add these coeffs to the single pauli coeffs of the main qubit. The extra fluf in coeffs[...] is just to find the correct coeff
                        coeffs[[F1.index(f) for f in F1 if f.to_label()[i]==pauli and f.to_label()[:i]+f.to_label()[i+1:]=="I"*(paulilength-1)][0]] += x
            logger.info("Summed extra qubits up and added it to main")


        self.noisemodel = NoiseModel(self.layer._cliff_layer, F1, coeffs)

    def _model_terms(self, links): #return a list of Pauli terms with the specified support
        groups = []
        for link in links:
            paulis = []
            for pauli in self._term_data.keys():
                overlap = [pauli[q].to_label() != "I" for q in link]
                support = [p.to_label() == "I" or q in link for q,p in enumerate(pauli)]
                if all(overlap) and all(support):
                    paulis.append(pauli)
            groups.append(paulis)

        return groups

    def get_spam_coeffs(self):
        """Return a dictionary of the spam coefficients of different model terms for use in 
        readout error mitigation when PER is carried out."""

        return dict(zip(self._term_data.keys(), [termdata.spam for termdata in self._term_data.values()]))

    def plot_coeffs(self, *links):
        """Plot the model coefficients in the generator of the sparse model corresponding
        to the current circuit layer"""

        coeffs_dict = dict(self.noisemodel.coeffs)
        groups = self._model_terms(links)
        fig, ax = plt.subplots()
        colcy = cycle(COLORS)
        for group in groups:
            c = next(colcy)
            coeffs = [coeffs_dict[term] for term in group]
            ax.bar([term.to_label() for term in group], coeffs, color=c)

    def graph(self, *links):
        """Graph the fits values for a certain subset of Pauli terms"""

        groups = self._model_terms(links)
        fig, ax = plt.subplots()
        for group in groups:
            for term in group:
                termdata = self._term_data[term]
                termdata.graph(ax)

        return ax

    def plot_infidelitites(self, *links):
        """Plot the infidelities of a subset of Pauli terms"""

        groups = self._model_terms(links)
        fig, ax = plt.subplots()
        colcy = cycle(COLORS)
        for group in groups:
            c = next(colcy)
            infidelities = [1-self._term_data[term].fidelity for term in group]
            ax.bar([term.to_label() for term in group], infidelities, color=c)
        return ax
    
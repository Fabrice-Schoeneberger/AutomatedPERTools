import logging
 
logging.basicConfig(filename="experiment.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)

class SparsePauliTomographyExperiment:
    """This class carries out the full experiment by creating and running a LayerLearning
    instance for each distinct layer, running the analysis, and then returning a PERCircuit
    with NoiseModels attached to each distinct layer"""

    def __init__(self, circuits, inst_map, backend, tomography_connections=False, sum_over_lambda=False):

        circuit_interface = None
        #Make sure it's a quantumcircuit as others don't work
        if circuits[0].__class__.__name__ == "QuantumCircuit":
            circuit_interface = QiskitCircuit
            processor = QiskitProcessor(backend)
        else:
            raise Exception("Unsupported circuit type") 
    
        self._profiles = set()
        used_qubits = set()
        for circuit in circuits: 
            for i in circuit.get_qubit_indices():
                used_qubits.add(i)
            
            circ_wrap = circuit_interface(circuit)
            parsed_circ = PERCircuit(circ_wrap)
            for layer in parsed_circ._layers:
                if layer.cliff_layer:
                    self._profiles.add(layer.cliff_layer)
                    
        plusone = set() #Here come the extra
        #tomography used qubits + all connected qubits
        if tomography_connections:
            #Get all connections with used qubits inside
            connection_map = [connection for connection in processor._qpu.coupling_map if any([used_qubit in connection for used_qubit in used_qubits])]
            for connection in connection_map: #Add all qubits with direct connections to the list
                plusone.add(connection[0])
                plusone.add(connection[1])

            logger.info("Added the following extra qubits")
            logger.info(plusone-used_qubits)
            plusone = plusone-used_qubits
            for bit in plusone:
                used_qubits.add(bit) #set used qubits to the expanded list

        #Now see which qubits are unused by all circuits
        unused_qubits = [bit for bit in inst_map if bit not in used_qubits]
        logger.info("The following Qubits were determinded unused")
        logger.info(unused_qubits)

        logger.info("Generated layer profile with %s layers:"%len(self._profiles))
        for layer in self._profiles:
            logger.info(layer)

        self._procspec = ProcessorSpec(inst_map, processor, unused_qubits, plusone)
        self.instances = []
        self._inst_map = inst_map
        self.unused_qubits = unused_qubits
        self.sum_over_lambda = sum_over_lambda
        self._layers = None

        self._layers = []
        for l in self._profiles:
            learning = LayerLearning(l,self._procspec)
            self._layers.append(learning)

        self.analysis = Analysis(self._layers, self._procspec, sum_over_lambda=sum_over_lambda, plusone=plusone)
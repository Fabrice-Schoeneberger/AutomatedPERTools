from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, Z, MEASURE
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
import pyquil.paulis as Pauli

with local_forest_runtime():
    n = 2
    prog = Program(
        Declare("ro", "BIT", n),
        H(0),
        CNOT(0, 1),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(10)
    qc = get_qc(str(n)+'q-qvm')
    #qc = get_qc('1q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
    executable = qc.compile(prog)
    result = qc.run(executable)
    bitstrings = result.get_register_map()['ro']
    for p in prog.instructions:
        if p._InstructionMeta__name == 'Declare':
            continue
        if hasattr(p, "qubits"):
            print([q.index for q in p.qubits])
        elif hasattr(p, "qubit"):
            print(p.qubit)
        else:
            print(p)
            raise Exception("No qubits")
    #print(bitstrings)
    #bitstrings = qvm.run(qvm.compile(prog)).get_register_map()
    #print(prog)
    pauli1 = Pauli.PauliTerm("X", index=0)
    pauli2 = Pauli.PauliTerm("X", index=0)
    pauli = pauli1 * pauli2
    print(pauli.pauli_string())
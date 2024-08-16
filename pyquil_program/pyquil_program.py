from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, Z, MEASURE, S, X, Y, I
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
import pyquil.paulis as Pauli
from pyquil.paulis import PauliTerm

def main():
    n = 5
    prog = Program(
        Declare("ro", "BIT", n),
        H(0),
        CNOT(0, 1),
        H(3),
        MEASURE(0, ("ro", 0)),
        MEASURE(1, ("ro", 1)),
    ).wrap_in_numshots_loop(10)
    qc = get_qc(str(n)+'q-qvm')
    #qc = get_qc('1q-qvm')  # You can make any 'nq-qvm' this way for any reasonable 'n'
    executable = qc.compile(prog)
    result = qc.run(executable)
    bitstrings = result.get_register_map()['ro']
    #for p in prog.instructions:
        #print(p._InstructionMeta__name)
        #print(p.name)
        #if p._InstructionMeta__name == 'Declare':
            #continue
        #if hasattr(p, "qubits"):
            #print([q.index for q in p.qubits])
        #elif hasattr(p, "qubit"):
            #print([p.qubit.index])
        #else:
            #print(p)
            #raise Exception("No qubits")
    #print(prog.get_qubit_indices())
    #print(bitstrings)
    #bitstrings = qvm.run(qvm.compile(prog)).get_register_map()
    #print(prog)
    prog += H(0)
    #print(prog)
    pauli = PauliTerm.from_list([("X", 0),("Y", 2), ("Z", 3), ("X", 4)])
    pauli1 = PauliTerm.from_list([(p,i) for i, p in enumerate("IX")])
    pauli2 = PauliTerm.from_list([(p,i) for i, p in enumerate("XX")])
    pauliliste= []
    pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("XX")]))
    pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("YX")]))
    pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("IY")]))
    #print(pauli.pauli_string(range(1+max(pauli.get_qubits()))))
    #print(str(pauli))
    #print(Pauli.check_commutation(pauli_list=[pauli2], pauli_two=pauli1))

    def conjugate_pauli_with_cliffords(pauli_term: PauliTerm, program: Program) -> PauliTerm:
        # Iterate through the program's instructions in reverse order
        term = pauli_term.pauli_string(range(1+max(pauli_term.get_qubits())))
        for instruction in reversed(program.instructions):
            if instruction.name == "H":
                for qubit in instruction.qubits:
                    if term[qubit.index] == "X":
                        term = PauliTerm("Z", qubit.index)
                    elif term[qubit.index] == "Z":
                        term = PauliTerm("X", qubit.index)
            elif instruction.name == "S":
                for qubit in instruction.qubits:
                    if term[qubit.index] == "X":
                        term = PauliTerm("Y", qubit.index)
                    elif term[qubit.index] == "Y":
                        term = PauliTerm("X", qubit.index)
            elif instruction.name == "S^-1":
                for qubit in instruction.qubits:
                    if term[qubit.index] == "Y":
                        term = PauliTerm("X", qubit.index)
                    elif term[qubit.index] == "X":
                        term = PauliTerm("Y", qubit.index)
            elif instruction.name == "CNOT":
                control_qubit = instruction.qubits[0].index
                target_qubit = instruction.qubits[1].index
                if term[control_qubit] == "X":
                    term = PauliTerm("X", target_qubit)
                if term[target_qubit] == "Z":
                    term = PauliTerm("Z", control_qubit)
            print(term)

        return PauliTerm.from_list([(p,i) for i, p in enumerate(term)])
    
    prog = Program()
    #prog += CNOT(0,1)
    prog += H(0)
    pauli = PauliTerm.from_list([(p,i) for i, p in enumerate("XX")])
    print(pauli.pauli_string(range(1+max(pauli.get_qubits()))))
    print(conjugate_pauli_with_cliffords(pauli, prog))
    # sudo docker run --rm -it -v ~/pyquil:/root/pyquil rigetti/forest python ~/Dokumente/Masterarbeit/AutomatedPERTools/pyquil_program/pyquil_program.py









from sys import platform
if platform == "linux" or platform == "linux2":
    main()
    # linux
elif platform == "darwin":
    raise Exception("Not supported")
    # OS X
elif platform == "win32":
    with local_forest_runtime():
        main()
    # Windows...

from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, Z, MEASURE
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare
import pyquil.paulis as Pauli
from pyquil.paulis import PauliTerm

#with local_forest_runtime():
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
print(prog.get_qubit_indices())
#print(bitstrings)
#bitstrings = qvm.run(qvm.compile(prog)).get_register_map()
print(prog)
prog += H(0)
print(prog)
pauli = PauliTerm.from_list([("X", 0),("Y", 2), ("Z", 3), ("X", 4)])
pauli1 = PauliTerm.from_list([(p,i) for i, p in enumerate("IX")])
pauli2 = PauliTerm.from_list([(p,i) for i, p in enumerate("XX")])
pauliliste= []
pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("XX")]))
pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("YX")]))
pauliliste.append(PauliTerm.from_list([(p,i) for i, p in enumerate("IY")]))
print(pauli.pauli_string(range(1+max(pauli.get_qubits()))))
print(str(pauli))
print(Pauli.check_commutation(pauli_list=[pauli2], pauli_two=pauli1))
# sudo docker run --rm -it -v ~/pyquil:/root/pyquil rigetti/forest python ~/Dokumente/Masterarbeit/AutomatedPERTools/pyquil_program/pyquil_program.py


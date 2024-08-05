from pyquil import get_qc, Program
from pyquil.gates import H, CNOT, Z, MEASURE
from pyquil.api import local_forest_runtime
from pyquil.quilbase import Declare

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
    print(bitstrings)
    #bitstrings = qvm.run(qvm.compile(prog)).get_register_map()
    #print(prog)
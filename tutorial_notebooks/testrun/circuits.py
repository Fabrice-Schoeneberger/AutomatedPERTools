def make_initial_Circuit(backend):
    from qiskit import transpile, QuantumCircuit
    qubits = [0,1,2,3]
    def trotterLayer(h,J,dt,n):
        trotterLayer = QuantumCircuit(4)
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
        trotterCircuit = QuantumCircuit(4)
        for i in range(s):
            trotterCircuit = trotterCircuit.compose(tL)
            trotterCircuit.barrier()

        transpiled = transpile(trotterCircuit, backend)
        return transpiled

    return [maketrotterCircuit(i) for i in range(1,15)]

def make_initial_Circuit2(backend):
    from qiskit import transpile, QuantumCircuit
    circuit = QuantumCircuit(5)
    circuit.cx(1,2)
    #circuit.barrier()
    #circuit.cx(1,0)
    #circuit.cx(3,4)
    #circuit.barrier()
    #circuit.cx(0,1)
    #circuit.cx(2,3)
    #circuit.cx(4,5)
    #circuit.cx(7,6)
    return [transpile(circuit, backend)]

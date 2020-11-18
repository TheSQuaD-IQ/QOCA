# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy

def expPauli_circuit(theta, Pauli_string):
    """
    Return the circuit for exp(- I theta [Pauli_string])
    theta (float): angle of rotation
    Pauli_string (list or array): ['X','Y','Z', ...] 
    """

    Pauli_string = np.array(Pauli_string)
    num_qubits = len(Pauli_string)
    nonI_idx = np.where(Pauli_string != 'I')[0] # qubits that don't have the identity operator

    qr = QuantumRegister(num_qubits)
    circ = QuantumCircuit(qr, name='exp(- i {} {})'.format(theta,Pauli_string))

    # Basis change
    for i in np.where(Pauli_string == 'X')[0]:
        circ.h(qr[int(i)])

    for i in np.where(Pauli_string == 'Y')[0]:
        circ.rx(np.pi/2., qr[int(i)])

    # CNOTS
    for i in range(len(nonI_idx)-1):
        circ.cx(qr[int(nonI_idx[i])],qr[int(nonI_idx[i+1])])

    # Rz(theta)
    circ.rz(2*theta,qr[int(nonI_idx[-1])]) 

    # CNOTS
    for i in reversed(range(len(nonI_idx)-1)):
        circ.cx(qr[int(nonI_idx[i])],qr[int(nonI_idx[i+1])])

    # Basis change
    for i in np.where(Pauli_string == 'X')[0]:
        circ.h(qr[int(i)])

    for i in np.where(Pauli_string == 'Y')[0]:
        circ.rx(-np.pi/2., qr[int(i)])

    # convert to gate
    return circ.to_instruction()


class QOCA(VariationalForm):
    """
    QOCA
    """

    CONFIGURATION = {
        'name': 'QOCA',
        'description': 'QOCA Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QOCA_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 1
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, hamiltonian_terms, drive_terms=np.array([]), depth=1, initial_state=None):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            hamiltonian_terms (list of arrays) : list of Pauli Hamiltionian terms grouped w.r.t the 
                                                 parametrization. e.g. if H = H1 + H2 + H3 where H1 and H2
                                                 have parameter a and H3 has parameter b, then the input
                                                 should be of the form [[H1,H2],[H3]].
                                                 Note that the order Hi's appear in the list will correspond
                                                 to their order in the circuit.
                                                 Hi's are given as Pauli strings: Hi = ['X','Y','Z', ...].
            initial_state (InitialState): an initial state object
            drive_terms (list of arrays) : same as hamiltonian_terms, but for the drive part of the circuit.
        """
        self.validate(locals())
        super().__init__()

        self._support_parameterized_circuit = True  # support setting via Terra's Parameter class 

        hamiltonian_terms = np.array(hamiltonian_terms)
        drive_terms = np.array(drive_terms)
        circuit_terms = np.concatenate(hamiltonian_terms, drive_terms)

        self._num_hamiltonian_params = len(hamiltonian_terms)
        self._num_drive_params       = len(drive_terms)
        self._num_terms              = len(circuit_terms)

        self._num_parameters_per_depth = self._num_terms

        self._num_parameters = depth * self._num_parameters_per_depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth 
        self._initial_state = initial_state 

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """

        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        
        circuit = QuantumCircuit(q)

        if self._initial_state is not None:
            init_circuit = deepcopy(self._initial_state.construct_circuit('circuit', q))
            init_circuit.name = 'Initial State'
            circuit.append(init_circuit.to_instruction(), [q[i] for i in range(q.size)])
                  
        param_idx = 0
     
        for block in range(self._depth):

            params_for_this_block = parameters[param_idx:param_idx + self._num_parameters_per_depth]
            circuit.barrier(q)

            # Build circuit term by term
            for group in enumerate(i, circuit_term):
                parameter = params_for_this_block[i]
                for term in group:
                    # Circuit for the exponential of Pauli terms
                    circuit.append(expPauli_circuit(parameter, term))

            param_idx += self._num_parameters_per_depth

        return circuit

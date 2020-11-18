# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class drive(VariationalForm):
    """
    drive with PBC
    Requires qubits to be labeled grouping spins up up up up, down down down down
    """

    CONFIGURATION = {
        'name': 'drive',
        'description': 'drive Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'drive_schema',
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

    def __init__(self, num_qubits, depth=3, initial_state=None):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            entangler_map (dict) : dictionary of entangling gates, in the format
                                    { source : [list of targets] },
                                    or None for full entanglement.
            entanglement (str): 'full' or 'linear'
            initial_state (InitialState): an initial state object
        """
        self.validate(locals())
        super().__init__()
        self._num_parameters_per_depth = num_qubits
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
        # Hamiltonian parameters
        Nsite= int(self._num_qubits/2)

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

            
            # Drive terms
            # initial Rx and Ry rotations
            circuit.rx(params_for_this_block[0],q[0])
            circuit.ry(params_for_this_block[1],q[0])

            circuit.rx(params_for_this_block[0],q[0+Nsite])
            circuit.ry(params_for_this_block[1],q[0+Nsite])

            # Other drive terms
            drive_param_idx = 2
            for i in range(Nsite-1):
                i += 1
                circuit.append(expPauli_circuit(params_for_this_block[drive_param_idx], i*['Z']+['X']), [q[i] for i in range(i+1)])
                circuit.append(expPauli_circuit(params_for_this_block[drive_param_idx+1], i*['Z']+['Y']), [q[i] for i in range(i+1)])

                circuit.append(expPauli_circuit(params_for_this_block[drive_param_idx], i*['Z']+['X']), [q[i+Nsite] for i in range(i+1)])
                circuit.append(expPauli_circuit(params_for_this_block[drive_param_idx+1], i*['Z']+['Y']), [q[i+Nsite] for i in range(i+1)])

                drive_param_idx += 2

            param_idx += self._num_parameters_per_depth

        return circuit

# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class short_QOCA(VariationalForm):
    """
    short_QOCA
    """

    CONFIGURATION = {
        'name': 'short_QOCA',
        'description': 'short_QOCA Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'short_QOCA_schema',
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

    def __init__(self, num_qubits, depth=3, initial_state=None, 
                 more_VHA_params=False, drive = False, more_drive_params = False):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            initial_state (InitialState): an initial state object
            more_VHA_params (bool) : Sets different variational parameters for all ZZ couplings and all 
            XX+YY gates for spins of the same kind.
            drive (bool) : Adds a drive term to the VHA
            more_drive_params (bool) : Sets different variational paramters for every site of the 
            drive terms.
        """
        self.validate(locals())
        super().__init__()

        self._support_parameterized_circuit = True  # support setting via Terra's Parameter class 

        Nsite = int(num_qubits/2)

        self._num_V_terms = Nsite

        if more_VHA_params is False: 
            self._num_V_params = 1
        else:
            self._num_V_params = Nsite

        if drive:
            if more_drive_params:
                self._num_drive_params = num_qubits
            else:
                self._num_drive_params = 2
        else:
            self._num_drive_params = 0

        self._num_parameters_per_depth = self._num_V_params + self._num_drive_params

        self._num_parameters = depth * self._num_parameters_per_depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth 
        self._initial_state = initial_state 
        self._more_VHA_params = more_VHA_params 
        self._more_drive_params = more_drive_params 
        self._drive = drive

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
        Nsite=int(self._num_qubits/2)

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

            V_params = [1] * self._num_V_terms

            if self._more_VHA_params is False:
                # V_params  = V_params*params_for_this_block[0]
                V_params = self._num_V_terms * [params_for_this_block[0]]
                idx = 0
            else:
                V_params  = params_for_this_block[0:self._num_V_params]

            if self._drive:
                drive_params = params_for_this_block[-self._num_drive_params:]
                if self._more_drive_params is False:
                    drive_params = np.array([params_for_this_block[-2],params_for_this_block[-1]]*Nsite)

            # exp(i theta V)
            circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])

            if self._drive:
                circuit.barrier(q)
                # Drive terms

                # spin up
                circuit.rx(drive_params[0],q[0])
                circuit.ry(drive_params[1],q[0])

                # spin down
                circuit.rx(drive_params[0],q[0+Nsite])
                circuit.ry(drive_params[1],q[0+Nsite])

                drive_params_idx = 2
                for i in range(Nsite - 1):

                    if i > 0:
                        circuit.cx(q[i-1],q[i])
                        circuit.cx(q[i-1+Nsite],q[i+Nsite])

                    Pauli_string_X = ['Z','X']
                    Pauli_string_Y = ['Z','Y']

                    # spin up
                    circuit.append(expPauli_circuit(drive_params[drive_params_idx], Pauli_string_X), [q[i], q[i+1]])
                    circuit.append(expPauli_circuit(drive_params[drive_params_idx+1], Pauli_string_Y), [q[i], q[i+1]])

                    # spin down
                    circuit.append(expPauli_circuit(drive_params[drive_params_idx], Pauli_string_X), [q[i+Nsite], q[i+1+Nsite]])
                    circuit.append(expPauli_circuit(drive_params[drive_params_idx+1], Pauli_string_Y), [q[i+Nsite], q[i+1+Nsite]])

                    if i == Nsite-2:
                        for j in range(Nsite-2):
                            circuit.cx(q[i-1-j],q[i-j])
                            circuit.cx(q[i-1-j+Nsite],q[i-j+Nsite])
                    drive_params_idx += 2

            param_idx += self._num_parameters_per_depth

        return circuit

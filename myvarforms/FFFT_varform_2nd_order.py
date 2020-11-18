# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class FFFT_varform_2nd_order(VariationalForm):
    """
    VHA with FFFT to compute the kinetic term in its diagonal form
    Requires qubits to be labeled grouping spins up up up up, down down down down
    """

    CONFIGURATION = {
        'name': 'FFFT_varform_2nd_order',
        'description': 'FFFT_varform_2nd_order Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'FFFT_varform_2nd_order_schema',
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

    def __init__(self, num_qubits, depth=3, entangler_map=None,
                 entanglement='full', initial_state=None, 
                 hardware_efficient_layer=False, more_params=False):
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
        if more_params: 
            self._num_parameters_per_depth = num_qubits
        else:
            self._num_parameters_per_depth = 2
        self._num_parameters = depth * self._num_parameters_per_depth
        if hardware_efficient_layer: self._num_parameters += 2*num_qubits
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state
        self._hardware_efficient_layer = hardware_efficient_layer
        self._more_params = more_params

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
        t=-1.
        U=4.
        mu=U/2.
        Nsite=4
        Norb=2*Nsite

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

        if self._hardware_efficient_layer:
            for qubit in range(self.num_qubits):
                circuit.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                circuit.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2

            for src, targ in self._entangler_map:
                circuit.cx(q[src], q[targ])
     
        for block in range(self._depth):

            params_for_this_block = parameters[param_idx:param_idx + self._num_parameters_per_depth]
            
            circuit.barrier(q)

            if self._more_params is False:
                V_params = np.ones(Nsite)*params_for_this_block[0]
                T_params = np.ones(Nsite)*params_for_this_block[1]
            else:
                V_params = params_for_this_block[0:Nsite]
                T_params = params_for_this_block[Nsite:]

            # exp(i theta V / 2)
            circuit.append(expV(V_params*U/8, Nsite), [q[i] for i in range(Norb)])

            # FFFT to go to momentum space
            circuit.append(FFFT4(), [q[i] for i in range(4)])
            circuit.append(FFFT4(), [q[i+4] for i in range(4)])

            # exp(i theta T_k) in momentum sace
            circuit.append(expTk(2.*T_params*t, Nsite), [q[i] for i in range(Norb)])

            # FFFTd to go back to real space
            circuit.append(FFFT4d(), [q[i] for i in range(4)])
            circuit.append(FFFT4d(), [q[i+4] for i in range(4)])

            # exp(i theta V / 2)
            circuit.append(expV(V_params*U/8, Nsite), [q[i] for i in range(Norb)])

            param_idx += self._num_parameters_per_depth

        return circuit

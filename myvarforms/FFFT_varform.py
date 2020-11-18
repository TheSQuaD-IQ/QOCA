# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class FFFT_varform(VariationalForm):
    """
    VHA with FFFT to compute the kinetic term in its diagonal form
    Requires qubits to be labeled grouping spins up up up up, down down down down
    """

    CONFIGURATION = {
        'name': 'FFFT_varform',
        'description': 'FFFT_varform Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'FFFT_varform_schema',
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
                 hardware_efficient_layer=False, more_params=False,
                 switch_V_T=False, drive = False):
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

        Nsite = int(num_qubits/2)
        if more_params: 
            self._num_V_params = Nsite
            self._num_T_params = Nsite
        else:
            self._num_V_params = 1
            self._num_T_params = 1
        if drive:
            self._num_drive_params = num_qubits
        else:
            self._num_drive_params = 0

        self._num_parameters_per_depth = self._num_V_params + self._num_T_params + self._num_drive_params

        self._num_parameters = depth * self._num_parameters_per_depth
        if hardware_efficient_layer: self._num_parameters += 2*num_qubits
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        self._initial_state = initial_state
        self._hardware_efficient_layer = hardware_efficient_layer
        self._more_params = more_params
        self._switch_V_T = switch_V_T
        self._drive = drive

        self._support_parameterized_circuit = True  # support setting via Terra's Parameter class 


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
                V_params = [params_for_this_block[0]] * Nsite
                T_params = [params_for_this_block[1]] * Nsite
            else:
                V_params = params_for_this_block[0:self._num_V_params]
                T_params = params_for_this_block[self._num_V_params:self._num_V_params+self._num_T_params]
            
            if self._drive: 
                drive_params = params_for_this_block[-self._num_drive_params:]


            if self._switch_V_T: # applying T than V

                # FFFT to go to momentum space
                circuit.append(FFFT(Nsite), [q[i] for i in range(Nsite)])
                circuit.append(FFFT(Nsite), [q[i+Nsite] for i in range(Nsite)])

                # exp(i theta T_k) in momentum sace
                circuit.append(expTk(T_params, Nsite), [q[i] for i in range(self._num_qubits)])

                # FFFTd to go back to real space
                circuit.append(FFFTd(Nsite), [q[i] for i in range(Nsite)])
                circuit.append(FFFTd(Nsite), [q[i+Nsite] for i in range(Nsite)])

                # exp(i theta V)
                circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])


            else: # applying V than T
                
                # exp(i theta V)
                circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])

                # FFFT to go to momentum space
                circuit.append(FFFT(Nsite), [q[i] for i in range(Nsite)])
                circuit.append(FFFT(Nsite), [q[i+Nsite] for i in range(Nsite)])

                # exp(i theta T_k) in momentum sace
                circuit.append(expTk(T_params, Nsite), [q[i] for i in range(self._num_qubits)])

                # FFFTd to go back to real space
                circuit.append(FFFTd(Nsite), [q[i] for i in range(Nsite)])
                circuit.append(FFFTd(Nsite), [q[i+Nsite] for i in range(Nsite)])

            if self._drive:
                # Drive terms
                # initial Rx and Ry rotations
                circuit.rx(drive_params[0],q[0])
                circuit.ry(drive_params[1],q[0])

                circuit.rx(drive_params[0],q[0+Nsite])
                circuit.ry(drive_params[1],q[0+Nsite])

                # Other drive terms
                drive_param_idx = 2
                for i in range(Nsite-1):
                    i += 1
                    circuit.append(expPauli_circuit(drive_params[drive_param_idx], i*['Z']+['X']), [q[i] for i in range(i+1)])
                    circuit.append(expPauli_circuit(drive_params[drive_param_idx+1], i*['Z']+['Y']), [q[i] for i in range(i+1)])

                    circuit.append(expPauli_circuit(drive_params[drive_param_idx], i*['Z']+['X']), [q[i+Nsite] for i in range(i+1)])
                    circuit.append(expPauli_circuit(drive_params[drive_param_idx+1], i*['Z']+['Y']), [q[i+Nsite] for i in range(i+1)])

                    drive_param_idx += 2

            param_idx += self._num_parameters_per_depth

        return circuit

# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class VHA(VariationalForm):
    """
    VHA with PBC
    Requires qubits to be labeled grouping spins up up up up, down down down down
    """

    CONFIGURATION = {
        'name': 'VHA',
        'description': 'VHA Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'VHA_schema',
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
                 more_params=False, Zstrings = True, drive = False):
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
            self._num_T_params = 2
        if drive:
            self._num_drive_params = num_qubits
        else:
            self._num_drive_params = 0

        self._num_parameters_per_depth = self._num_V_params + self._num_T_params + self._num_drive_params

        self._num_parameters = depth * self._num_parameters_per_depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        self._initial_state = initial_state
        self._more_params = more_params
        self._Zstrings = Zstrings
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
        # Hamiltonian parameters
        t=-1.
        U=4.
        mu=U/2.
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

            if self._more_params is False:
                V_params  = np.ones(Nsite)*params_for_this_block[0]
                T1_params = np.ones(Nsite)*params_for_this_block[1]
                T2_params = np.ones(Nsite)*params_for_this_block[2]
            else:
                V_params  = params_for_this_block[0:4]
                T1_params = params_for_this_block[4:8]
                T2_params = params_for_this_block[8:12]

            drive_params = params_for_this_block[-8:]

            # exp(i theta V)
            circuit.append(expV(V_params*U/4, Nsite), [q[i] for i in range(self._num_qubits)])

            # T = (XXII + YYII + IIXX + IIYY) + (IXXI + IYYI + XZZX + YZZY)
            circuit.append(XX_gate(T1_params[0]*t/2.), [q[0],q[1]])     # XXII
            circuit.append(XX_gate(T1_params[0]*t/2.), [q[0+4],q[1+4]]) # XXII
            circuit.append(YY_gate(T1_params[1]*t/2.), [q[0],q[1]])     # YYII
            circuit.append(YY_gate(T1_params[1]*t/2.), [q[0+4],q[1+4]]) # YYII

            circuit.append(XX_gate(T1_params[2]*t/2.), [q[2],q[3]])     # IIXX
            circuit.append(XX_gate(T1_params[2]*t/2.), [q[2+4],q[3+4]]) # IIXX
            circuit.append(YY_gate(T1_params[3]*t/2.), [q[2],q[3]])     # IIYY
            circuit.append(YY_gate(T1_params[3]*t/2.), [q[2+4],q[3+4]]) # IIYY

            circuit.append(XX_gate(T2_params[0]*t/2.), [q[1],q[2]])     # IXXI
            circuit.append(XX_gate(T2_params[0]*t/2.), [q[1+4],q[2+4]]) # IXXI
            circuit.append(YY_gate(T2_params[1]*t/2.), [q[1],q[2]])     # IYYI
            circuit.append(YY_gate(T2_params[1]*t/2.), [q[1+4],q[2+4]]) # IYYI

            # if self._PBC:
            #     if self._Zstrings:
            #         circuit.append(XZsX_gate(T2_params[2]*t/2., L=4), [q[i] for i in range(Nsite)])     # XZZX
            #         circuit.append(XZsX_gate(T2_params[2]*t/2., L=4), [q[i+4] for i in range(Nsite)])   # XZZX
            #         circuit.append(YZsY_gate(T2_params[3]*t/2., L=4), [q[i] for i in range(Nsite)])     # YZZY
            #         circuit.append(YZsY_gate(T2_params[3]*t/2., L=4), [q[i+4] for i in range(Nsite)])   # YZZY
            #     else:
            #         circuit.append(XX_gate(T2_params[2]*t/2.), [q[0],q[3]])     # XIIX
            #         circuit.append(XX_gate(T2_params[2]*t/2.), [q[0+4],q[3+4]]) # XIIX
            #         circuit.append(YY_gate(T2_params[3]*t/2.), [q[0],q[3]])     # YIIY
            #         circuit.append(YY_gate(T2_params[3]*t/2.), [q[0+4],q[3+4]]) # YIIY


            if self._drive:
                # Drive terms
                # spin ups
                circuit.rx(drive_params[0],q[0])
                circuit.ry(drive_params[1],q[0])
                circuit.append(expPauli_circuit(drive_params[2], ['Z','X']), [q[i] for i in range(2)])
                circuit.append(expPauli_circuit(drive_params[3], ['Z','Y']), [q[i] for i in range(2)])
                circuit.append(expPauli_circuit(drive_params[4], ['Z','Z','X']), [q[i] for i in range(3)])
                circuit.append(expPauli_circuit(drive_params[5], ['Z','Z','Y']), [q[i] for i in range(3)])
                circuit.append(expPauli_circuit(drive_params[6], ['Z','Z','Z','X']), [q[i] for i in range(4)])
                circuit.append(expPauli_circuit(drive_params[7], ['Z','Z','Z','Y']), [q[i] for i in range(4)])

                # spin downs
                circuit.rx(drive_params[0],q[0+Nsite])
                circuit.ry(drive_params[1],q[0+Nsite])
                circuit.append(expPauli_circuit(drive_params[2], ['Z','X']), [q[i+Nsite] for i in range(2)])
                circuit.append(expPauli_circuit(drive_params[3], ['Z','Y']), [q[i+Nsite] for i in range(2)])
                circuit.append(expPauli_circuit(drive_params[4], ['Z','Z','X']), [q[i+Nsite] for i in range(3)])
                circuit.append(expPauli_circuit(drive_params[5], ['Z','Z','Y']), [q[i+Nsite] for i in range(3)])
                circuit.append(expPauli_circuit(drive_params[6], ['Z','Z','Z','X']), [q[i+Nsite] for i in range(4)])
                circuit.append(expPauli_circuit(drive_params[7], ['Z','Z','Z','Y']), [q[i+Nsite] for i in range(4)])

            param_idx += self._num_parameters_per_depth

        return circuit

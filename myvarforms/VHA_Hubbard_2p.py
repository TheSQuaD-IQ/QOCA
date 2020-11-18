# -*- coding: utf-8 -*-


import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy
from .gates import *


class VHA_Hubbard_2p(VariationalForm):
    """
    VHA_Hubbard_2p
    """

    CONFIGURATION = {
        'name': 'VHA_Hubbard_2p',
        'description': 'VHA_Hubbard_2p Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'VHA_Hubbard_2p_schema',
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

    def __init__(self, num_qubits, hopping_couplings, depth=3, initial_state=None, 
                 more_VHA_params=False, drive = False, more_drive_params = False, 
                 apply_T_first = False):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of rotation layers
            hopping_couplings (list of arrays) : list of qubit couplings for spin up hopping terms.
                                                 Each of them will have a exp(XX + YY) gate.
                                                 Repeat for i+Nsite for spin down hopping terms.
                                                 Arranged as [(group 1),(group 2),(group 3),...]
                                                 where all terms of group i commute.
            initial_state (InitialState): an initial state object
            more_VHA_params (bool) : Sets different variational parameters for all ZZ couplings and all 
            XX+YY gates for spins of the same kind.
            drive (bool) : Adds a drive term to the VHA
            more_drive_params (bool) : Sets different variational paramters for every site of the 
            drive terms.
            apply_T_first (bool) : If set True, applies exp(T) before exp(V) in the VHA
        """
        self.validate(locals())
        super().__init__()

        self._support_parameterized_circuit = True  # support setting via Terra's Parameter class 

        Nsite = int(num_qubits/2)

        num_groups = 0
        num_terms = 0
        for group in hopping_couplings:
            if len(group) != 0: 
                num_groups += 1
                num_terms += len(group)

        self._num_T_groups = num_groups
        self._num_T_terms = num_terms
        self._num_V_terms = Nsite

        self._hopping_couplings = hopping_couplings 

        if more_VHA_params is False: 
            self._num_V_params = 1
            self._num_T_params = 0
        else:
            self._num_V_params = Nsite
            self._num_T_params = num_terms

        if drive:
            if more_drive_params:
                self._num_drive_params = num_qubits
            else:
                self._num_drive_params = 1
        else:
            self._num_drive_params = 0

        self._num_parameters_per_depth = self._num_V_params + self._num_T_params + self._num_drive_params

        self._num_parameters = depth * self._num_parameters_per_depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth 
        self._initial_state = initial_state 
        self._more_VHA_params = more_VHA_params 
        self._more_drive_params = more_drive_params 
        self._drive = drive
        self._apply_T_first = apply_T_first

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
        # t=-1.
        # U=4.
        # mu=U/2.
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

            T_params = [1] * self._num_T_terms
            V_params = [1] * self._num_V_terms

            if self._more_VHA_params is False:
                # V_params  = V_params*params_for_this_block[0]
                V_params = self._num_V_terms * [params_for_this_block[0]]
                T_params = self._num_T_terms * [params_for_this_block[1]]
                # idx = 0
                # for i in range(self._num_T_params):
                #     num_hoppings = len(self._hopping_couplings[i])
                #     T_params[idx:idx+num_hoppings] = num_hoppings * [params_for_this_block[i+1]]
                #     idx += len(self._hopping_couplings[i])
            else:
                V_params  = params_for_this_block[0:self._num_V_params]
                T_params  = params_for_this_block[self._num_V_params:self._num_V_params+self._num_T_terms]

            if self._drive:
                drive_params = params_for_this_block[-self._num_drive_params:]
                if self._more_drive_params is False:
                    drive_params = np.array([params_for_this_block[-2],params_for_this_block[-1]]*Nsite)


            if self._apply_T_first:
                # exp(i theta T)
                theta_idx = 0
                for group in self._hopping_couplings:
                    for term in group:
                        term = np.sort(term)
                        num_Z = abs(term[1]-term[0]) - 1
                        num_qb = abs(term[1]-term[0]) + 1
                        stringXX = ['X'] + ['Z']*num_Z + ['X']
                        stringYY = ['Y'] + ['Z']*num_Z + ['Y']
                        theta = T_params[theta_idx]

                        circuit.append(XX_YY(theta, num_qb), [q[int(i + term[0])] for i in range(num_qb)])
                        circuit.append(XX_YY(theta, num_qb), [q[int(i + term[0] + Nsite)] for i in range(num_qb)])


                        theta_idx += 1

                # exp(i theta V)
                circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])

            else:
                # exp(i theta V)
                circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])

                # exp(i theta T)
                theta_idx = 0
                for group in self._hopping_couplings:
                    for term in group:
                        term = np.sort(term)
                        num_Z = abs(term[1]-term[0]) - 1
                        num_qb = abs(term[1]-term[0]) + 1
                        stringXX = ['X'] + ['Z']*num_Z + ['X']
                        stringYY = ['Y'] + ['Z']*num_Z + ['Y']
                        theta = T_params[theta_idx]

                        circuit.append(XX_YY(theta, num_qb), [q[int(i + term[0])] for i in range(num_qb)])
                        circuit.append(XX_YY(theta, num_qb), [q[int(i + term[0] + Nsite)] for i in range(num_qb)])


                        theta_idx += 1

            # # exp(i theta V)
            # circuit.append(expV(V_params, Nsite), [q[i] for i in range(self._num_qubits)])

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

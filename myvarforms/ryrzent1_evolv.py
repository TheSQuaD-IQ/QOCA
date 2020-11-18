import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from copy import deepcopy


class RYRZENT1_EVOLV(VariationalForm):
    """Layers of Y+Z rotations followed by entangling gates."""

    CONFIGURATION = {
        'name': 'RYRZENT1_EVOLV',
        'description': 'RYRZENT1_EVOLV Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ryrzent1_schema',
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
                 entanglement='full', initial_state=None):
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
        self._num_parameters = num_qubits * depth * 4
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state
        #self._prior_circuit = prior_circuit

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
        if self._initial_state is not None:
            circuit = deepcopy(self._initial_state.construct_circuit('circuit', q))
        #if self._prior_circuit is not None:
        #    circuit = self._prior_circuit
        else:
            circuit = QuantumCircuit(q)
        #q2 = QuantumRegister(self._num_qubits, name='q2')

        param_idx = 0

        for block in range(self._depth):

            rotations1 = QuantumCircuit(q)
            rotations2 = QuantumCircuit(q)
            entangler  = QuantumCircuit(q)
    
    
            
            # build rotations 1 circuit
            for qubit in range(self._num_qubits):
                rotations1.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                rotations1.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2
    
            # param_idx = int(len(parameters)/2)
            # build rotations 2 circuit
            for qubit in range(self._num_qubits):
                rotations2.u3(parameters[param_idx], 0.0, 0.0, q[qubit])  # ry
                rotations2.u1(parameters[param_idx + 1], q[qubit])  # rz
                param_idx += 2
    
            # build entangler
            for node in self._entangler_map:
                for target in self._entangler_map[node]:
                    # cx
                    entangler.cx(q[node], q[target])
    
    
            # add rotations and entanglers to circuit
            #circ1 = circuit.combine(rotations1)
            #circ2 = circ1.combine(entangler.inverse())
            #circ3 = circ2.combine(rotations2)
            #circ4 = circ3.combine(entangler.inverse()) # reverse entangler back     
            
            circuit.extend(rotations1)
            circuit.extend(entangler.inverse())
            circuit.extend(rotations2)
            circuit.extend(entangler.inverse())


            circuit.barrier(q)



        return circuit
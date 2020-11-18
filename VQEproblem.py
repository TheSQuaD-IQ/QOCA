# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '/Users/alexandre/Documents/Maitrise/Projects/Quantum_Simulation/vqe_hubbard')

from imports import *




class VQEproblem:
    
    def __init__(self, **params):
        super().__init__()
        
        # params (dict) : paramters for different functions
        self._Hamiltonian_params = params['Hamiltonian_params']
        self._VQE_params = params['VQE_params']
        self._var_form_params = self._VQE_params['var_form_params']
        self._optimizer_params = self._VQE_params['optimizer_params']
        self._vqe_algorithm_params = self._VQE_params['vqe_algorithm_params']
        self._simulation_params = self._VQE_params['simulation_params']
        self._noise_params = self._VQE_params['noise_params']

        
        # Generate the Hamiltonian
        # self.Hamiltonian = qt2qk(self._Hamiltonian_params['H_generator'](**self._Hamiltonian_params))

        # # Solve it classically
        # self.eigenstates = self.Hamiltonian.eigenstates()
        
        # Hamiltonian in Pauli terms
        if params['Hamiltonian_params']['H_to_Pauli_op']:
            self._qubitH = params['Hamiltonian_params']['H_to_Pauli_op'](**params['Hamiltonian_params'])
        elif params['Hamiltonian_params']['qubitOp']:
            self._qubitH = params['Hamiltonian_params']['qubitOp']
        else:
            # Generate the Hamiltonian
            self.Hamiltonian = qt2qk(self._Hamiltonian_params['H_generator'](**self._Hamiltonian_params))
            self._qubitH  = MatrixOperator(matrix=self.Hamiltonian.data)

        self._num_qubits  = self._qubitH.num_qubits
        
        # optimizer
        if self._noise_params['turn_on_noise']:
            self._optimizer_function = SPSA(**self._optimizer_params['optimizer_arguments'])
        else:
            self._optimizer_function = self._optimizer_params['optimizer'](**self._optimizer_params['optimizer_arguments'])
        
        # variational form of the hamiltonian
        self._var_form = self._var_form_params['variational_form'](num_qubits=self._num_qubits,**self._var_form_params['variational_form_args'])

        # backend
        self._backend = self._simulation_params['backend']


    def circuit_generator(self, parameters=np.array([0]), random_parameters=True):
        """
        Generates the quantum circuit of the variational form
        """
        if parameters.any():
            circuit = self._var_form.construct_circuit(parameters)
        else:
            if random_parameters is False:
                circuit = self._var_form.construct_circuit(np.zeros(self._var_form._num_parameters))
            else:
                random_parameters = self._random_initial_points(num_instances=1)[0]
                circuit = self._var_form.construct_circuit(np.array(random_parameters))

        

        return circuit
    
    def run_sim(self, initial_point=None, file_name='', file_dir=None, backend=None):
        """
        proceeds to do a statevector simulation of the VQE algorithm

        Args:
            initial_point (list) : initial paramters for the optimizer. If None, random parameters are chosen
            file_name (str) : name of the file in which the results shall be printed
            file_dir (str) : name of the directory in which the results file shall be created
        """
        if backend:
            _backend = backend
        else:
            _backend = self._backend

        if self._vqe_algorithm_params['initial_point'] is not None:
            if self._vqe_algorithm_params['initial_point'] is 'zeros':
                num_var_params = self._var_form._num_parameters
                initial_point = np.zeros(num_var_params)
            else:
                initial_point = self._vqe_algorithm_params['initial_point']


        self.results_transcript = {
            # 'gs_energy':      self.eigenstates[0][0],
            # 'gs_ket':         self.eigenstates[1][0],
            'eval_count':     [],
            'mean_energy':    [],
            # 'std_energy':     [],
            'parameter_sets': []
            }

        callback = self._append_results_transcript

        #creates the algorithm to be executed
        vqe_algorithm = VQE(self._qubitH, self._var_form, self._optimizer_function,
                     initial_point=initial_point,
                    callback=callback)
            
        # backend
        backend_ = Aer.get_backend(_backend)

        # coupling map
        coupling_map = self._simulation_params['coupling_map']
        if coupling_map:
            backend_.configuration().coupling_map = coupling_map
            #quantum_instance = QuantumInstance(backend_)
            #quantum_instance._backend_config['coupling_map']=coupling_map
        
        # Noise model for noisy simulation
        if self._noise_params['turn_on_noise']:
            # device      = IBMQ.get_backend(self._noise_params['device'])
            device      = self._noise_params['backend']
            properties  = device.properties()
            gate_times  = self._noise_params['gate_times']
            noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)
            # TODO: test is qasm_simulator is necessary for noisy simulations
            #backend_ = Aer.get_backend('qasm_simulator')
            # Here I can input the device coupling map if I want
            #coupling_map = device.configuration().coupling_map
            #backend_.configuration().coupling_map = coupling_map
            # TODO: check which basis_gates to use, the device or the simulator
            #basis_gates = noise_model.basis_gates
            basis_gates = device.configuration().basis_gates
            backend_.configuration().basis_gates = basis_gates
        else:
            noise_model = None

        backend_options = self._simulation_params['backend_options']

        quantum_instance = QuantumInstance(backend_, noise_model=noise_model, 
                                            backend_options=backend_options)
        
        # run the algorithm
        results = vqe_algorithm.run(quantum_instance)

        # add results of the VQE to the results_transcript
        self.results_transcript['final_results'] = results

        if file_name or self._vqe_algorithm_params['write_to_file']:

            self._vqe_algorithm_params['write_to_file'] = True
            if file_name:
                self._vqe_algorithm_params['file_name']     = file_name
                #self._vqe_algorithm_params['file_dir']      = file_dir
            # print results_transcript dict to pickle file
            self._print_to_file(self.results_transcript)
            
       
        
        print('The computed ground state energy is: {:.12f}'.format(results['eigvals'][0]))
        return results


    def _append_results_transcript(self, eval_count, parameter_sets, mean_energy, std_energy):

        self.results_transcript['eval_count'].append(eval_count)
        self.results_transcript['mean_energy'].append(mean_energy)
        # self.results_transcript['std_energy'].append(std_energy)
        self.results_transcript['parameter_sets'].append(parameter_sets.tolist())

        if int(eval_count)%250 == 0:
            print('eval_count = {}, energy = {}'.format(eval_count, mean_energy))

        if int(eval_count)%50 == 0:
            self._print_to_file(self.results_transcript)

    def _print_to_file(self, obj):

        file_dir  = self._vqe_algorithm_params['file_dir']
        file_name = self._vqe_algorithm_params['file_name']

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(file_dir+ file_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    def _random_initial_points(self, num_instances=None, lower_bound=-np.pi, upper_bound=np.pi):
        """
        Creates a collection of random initial parameters for the optimizer

        Args:
            num_instances (int) : number of instances in the batch
        """    
        if num_instances:
            num_of_instances = num_instances
        # else:
        #     num_of_instances = self._num_instances

        random_params = np.random.uniform(lower_bound,upper_bound, (num_of_instances, self._var_form._num_parameters))
        
        return random_params.tolist()
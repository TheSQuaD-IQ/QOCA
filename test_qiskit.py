import os
import sys
sys.path.insert(0, '..')
from VQEproblem import *



# directory where to put the results
results_dir = os.getcwd()+'/results/'

# Number of fermionic sites
N = 1
M = 2
Nsite = int(N*M)
hopping_couplings = NxM_nearest_coupling_list(N,M)

# variational form
var_form = VHA_Hubbard

# Turn ON/OFF device noise in the simulation
turn_on_noise = False

global params
params = {
    # Hamiltonian params
    'Hamiltonian_params': {
        
        'name':             'Fermi-Hubbard',
        'H_generator':      hb.Fermi_Hubbard_H_2D,
        'H_to_Pauli_op':    hb.FH_2D_Pauli,
        't':                -1.,
        'U':                4.,
        'mu':               2.,
        'N':                N,
        'M':                M,
        'Nsite':            Nsite,
        'half_filling':     True,
        'convention':       'spin', # 'spin': |n_1_up, n_2_up ; n_1_down, n_2_down >
                                    # 'site': |n_1_up, n_1_down ; n_2_up, n_2_down >  
        'pbc':              False,   # Periodic Boundary Conditions
        'eta':              -0.05   # For Green functions
    },
    
    # VQE params
    'VQE_params': {
        # Variational form parameters
        'var_form_params': {
            'variational_form': var_form,
            'variational_form_args': {
                'depth':            1,
                'entangler_map':    None,
                'entanglement':     'linear',
                'initial_state':    None
            }
            
            
        },
        # Optimizer settings
        'optimizer_params': {
            'optimizer':             COBYLA,
            'optimizer_arguments':   {'maxiter':6000, 'disp':False, 'rhobeg':np.pi/2., 'tol':None},
            'cost_function':         None
        
        },
        # VQE_algorithm settings
        'vqe_algorithm_params': {
            'initial_point': None,
            'write_to_file': True,
            'file_name':     'results',
            'file_dir':      results_dir,
            'main_dir':      results_dir
        },
        
        # simulation settings
        'simulation_params': {
            'coupling_map':  None,
            'backend':       'statevector_simulator',
            'backend_options':  {
                'max_parallel_threads':             0, # see statevector_simulator.py for description
                'max_parallel_experiments':         0, # of backend options
                #'max_memory_mb':                    0,
                'statevector_parallel_threshold':   1
            }
        },
        
        # Noise settings
        'noise_params': {
            'turn_on_noise': turn_on_noise,
            'device':        'ibmq_16_melbourne',
            'gate_times':    None,
        }
        
    },
    
    # batch params
    'num_instances':         2
    
}

############## Classical solver ##############

H = qt2qk(params['Hamiltonian_params']['H_generator'](**params['Hamiltonian_params']))
evals, ekets = H.eigenstates()


############## GS of T ##############

num_qubits = 2 * Nsite
circ_initial_state = T_ground_state_circuit_spins(num_qubits)
gsT_initial_state_aqua = Custom(num_qubits, circuit=circ_initial_state)

############## GS of U ##############

num_qubits = 2 * Nsite
circ_initial_state = U_ground_state_circuit_spins(num_qubits)
gsU_initial_state_aqua = Custom(num_qubits, circuit=circ_initial_state)



############## Run the VQE simulation ##############

# set the depth
depth = int(1)
variational_form = FFFT_varform_2sites
varform_name = 'FFFT_varform_2sites'

# sets which initial_state to choose
initial_state_name = 'gsT'
# initial_state_name = 'gsU'
# initial_state_name = 'None'

if initial_state_name == 'gsT':
    # sets the initial state
    initial_state = deepcopy(gsT_initial_state_aqua)
    # adjust the file name according to which initial_state was chosen
    switch_V_T = False
    file_name = 'gsT_initial_state_'
elif initial_state_name == 'gsU':
    initial_state = deepcopy(gsU_initial_state_aqua)
    switch_V_T = True
    file_name = 'gsU_initial_state_'
elif initial_state_name=='None':
    initial_state = None
    switch_V_T = False
    file_name = 'zero_initial_state_'

# add the depth and varform to the file name
file_name += 'depth_{}_'.format(depth)
file_name += 'varform_{}'.format(varform_name)

# Set params for simulation
params['VQE_params']['var_form_params']['variational_form'] = variational_form
params['VQE_params']['var_form_params']['variational_form_args'] = {
                'depth':            depth,
                'entangler_map':    None,
                'entanglement':     'linear',
                'initial_state':    initial_state,
                'hardware_efficient_layer': False,
                'more_params':      False,
                'switch_V_T':       switch_V_T
            }
num_var_params = variational_form(num_qubits=2*Nsite, **params['VQE_params']['var_form_params']['variational_form_args'])._num_parameters_per_depth
params['VQE_params']['vqe_algorithm_params']['initial_point'] = np.zeros(num_var_params*depth)
# params['VQE_params']['vqe_algorithm_params']['initial_point'] = np.array([-np.pi, np.pi])
params['VQE_params']['optimizer_params']['optimizer'] = COBYLA
params['VQE_params']['optimizer_params']['optimizer_arguments'] = {
    'maxiter':4000*depth, 
    'disp':False, 
    'rhobeg':np.pi/16., 
    'tol':None
}
params['VQE_params']['vqe_algorithm_params']['file_name'] = file_name

# Create problem instance
prob = VQEproblem(**params)

# Run the simulation
start_time = timeit.default_timer()
results = prob.run_sim()
end_time = timeit.default_timer()

# Compute final state fidelity with the exact GS
gs_ket = ekets[0]
final_ket = Qobj(prob.results_transcript['final_results']['min_vector'], 
                                                    dims = gs_ket.dims)
fidelities = np.array([abs(final_ket.overlap(ket))**2 for ket in ekets])
eig_ket_idx = np.arange(len(fidelities))

# Print results
print('The exact ground state energy is: {}'.format(np.real(evals[0])))
print('The fidelity with the exact ground state is: {}'.format(fidelities[0]))
print('.\n.\n.')
print('Time of the simulation: {}'.format(end_time-start_time))
print('The number of steps is: {}'.format(results['num_optimizer_evals']))
print('The final state has the most overlap with eigenstate number {}'.format(np.argmax(fidelities)))
print('This overlap is: {}'.format(fidelities[np.argmax(fidelities)]))

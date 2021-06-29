#!/usr/bin/env qiskit0.10
'''
 * Author:    Alexandre Choquette
 * Created:   Nov. 19, 2019
 * Patent:    PCT/CA2021/050468
'''


from qutip import *
from scipy.sparse import diags
import numpy as np
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
import hubbard as hb
from support_functions import *
from qiskit.aqua.components.initial_states import *
from qiskit.compiler import transpile

"""
Throughout this file, we use the convention 
|n_0_up, n_1_up, ... : n_0_down, n_1_down, ... >
to label our sites within a statevector.
"""

######### Fermionic Operators #########

def f_destroy(n, N, start_at_zero=True):
    """
    returns a fermionic descruction operator of an electron 
    for the n'th orbital out of N orbitals.
    if start_at_zero=False, sites are labelled from 1 to N
    if start_at_zero=True, sites are labelled from 0 to N-1
    """
    
    if start_at_zero:
        n+=1

    if n < 1 or n > N:
        if start_at_zero==True:
            raise ValueError('n has to be in [0, ..., N-1 ]')
        else:
            raise ValueError('n has to be in [1, ..., N ]')
        
    if n == 1:
        if N == 1:
            return destroy(2)
        else:
            return tensor(destroy(2), qeye([2 for i in range(N-1)]))
    
    elif n == N:
        return tensor([sigmaz() for i in range(N-1)] + [destroy(2)])
    
    else:
        return tensor([sigmaz() for i in range(n-1)] + [destroy(2)] + [qeye(2) for i in range(N-n)])
    
def f_create(n, N, start_at_zero=True):
    """
    hc of destroy
    """
    return f_destroy(n, N, start_at_zero=start_at_zero).dag()

def vacuum(N):
    """
    Vacuum state of N fermionic orbitals
    """
    single_site_vac = basis(2,0)
    return tensor([single_site_vac for i in range(N)])

######### Configuration of system #########

def site_labelling_2D(N, M):
    """
    Returns the qubit labelling matrix for a 2D array. Snake shape.
    0 1 2 3 4 
    9 8 7 6 5
    """
    
    labels_list = []
    
    for n in range(N):
        ms = []
        for m in range(M):
            ms.append(n*M + m)
        if n % 2 == 0:
            labels_list.append(ms)
        else:
            labels_list.append(ms[::-1])
    
    return np.array(labels_list)

def NxM_nearest_coupling_list(N, M, periodic=False):
    """
    Returns the the list of qubit couplings for a 2D array 
    of fermionic sites. Sorted as 
    [horrizontal_even,
     horrizontal_odd,
     vertical_even,
     vertical_odd,
     periodic_horrizontal,
     periodic_vertical]
    """
    
    hopping_couplings = []

    horrizontal_even = []
    horrizontal_odd  = []
    vertical_even    = []
    vertical_odd     = []
    if periodic:
        periodic_horrizontal = []
        periodic_vertical = []

    labels = site_labelling_2D(N,M)
    for n in range(N):
        for m in range(M):
            if m==M-1: 
                break
            if m % 2 == 0:
                horrizontal_even.append(np.sort([labels[n][m],labels[n][m+1]]).tolist())

            else:
                horrizontal_odd.append(np.sort([labels[n][m],labels[n][m+1]]).tolist())

    if periodic and M>2:
        for n in range(N):
            periodic_horrizontal.append(np.sort([labels[n][0],labels[n][-1]]).tolist())

    transposed_labels = site_labelling_2D(N,M).transpose()
    for m in range(M):
        for n in range(N):
            if n==N-1: 
                break
            if n % 2 == 0:
                vertical_even.append(np.sort([transposed_labels[m][n],transposed_labels[m][n+1]]).tolist())
            else:
                vertical_odd.append(np.sort([transposed_labels[m][n],transposed_labels[m][n+1]]).tolist())

    if periodic and N>2:
        for m in range(M):
            periodic_vertical.append(np.sort([transposed_labels[m][0],transposed_labels[m][-1]]).tolist())

    if len(horrizontal_even)!= 0: hopping_couplings.append(horrizontal_even)
    if len(horrizontal_odd)!= 0: hopping_couplings.append(horrizontal_odd)
    if len(vertical_even)!= 0: hopping_couplings.append(vertical_even)
    if len(vertical_odd)!= 0: hopping_couplings.append(vertical_odd)
    if periodic:
        if len(periodic_horrizontal)!= 0: hopping_couplings.append(periodic_horrizontal)
        if len(periodic_vertical)!= 0: hopping_couplings.append(periodic_vertical)

    return hopping_couplings



######### Hamiltonian #########

def hopping_term(i, j, Norb):
    """
    returns adag_i a_j + adag_j a_i
    for a system with Norb fermionic orbitals.
    """
    ai  = f_destroy(i, Norb)
    adi = ai.dag()
    aj  = f_destroy(j, Norb)
    adj = aj.dag()

    return adi*aj + adj*ai

def interaction_term(i, Nsite):
    """
    returns ni_up * ni_down
    for a system with Nsite sites.
    """
    ai_up  = f_destroy(i, 2*Nsite)
    adi_up = ai_up.dag()
    ni_up  = adi_up * ai_up

    ai_down  = f_destroy(i+Nsite, 2*Nsite)
    adi_down = ai_down.dag()
    ni_down  = adi_down * ai_down

    return ni_up * ni_down

def chemical_potential_term(i, Nsite):
    """
    returns ni_up + ni_down
    for a system with Nsite sites.
    """
    ai_up  = f_destroy(i, 2*Nsite)
    adi_up = ai_up.dag()
    ni_up  = adi_up * ai_up

    ai_down  = f_destroy(i+Nsite, 2*Nsite)
    adi_down = ai_down.dag()
    ni_down  = adi_down * ai_down

    return ni_up + ni_down

def hopping_Hamiltonian(hopping_couplings, Nsite):
    """
    returns the hopping Hamiltonian given a list
    of hopping couplings.
    """
    Norb = 2*Nsite

    T_up   = Qobj(dims=[[2 for j in range(Norb)] for i in range(2)])
    T_down = Qobj(dims=[[2 for j in range(Norb)] for i in range(2)])

    for group in hopping_couplings:
        for hop in group:
            T_up   += hopping_term(hop[0], hop[1], Norb)
            T_down += hopping_term(hop[0]+Nsite, hop[1]+Nsite, Norb)

    return T_up + T_down

def interaction_Hamiltonian(Nsite):
    """
    returns the interaction Hamiltonian 
    for Nsite.
    """
    Norb = 2*Nsite
    U   = Qobj(dims=[[2 for j in range(Norb)] for i in range(2)])
    for i in range(Nsite):
        U += interaction_term(i, Nsite)
    
    return U

def chemical_potential_Hamiltonian(Nsite):
    """
    returns the chemical potential Hamiltonian 
    for Nsite.
    """
    Norb = 2*Nsite
    Mu   = Qobj(dims=[[2 for j in range(Norb)] for i in range(2)])
    for i in range(Nsite):
        Mu += chemical_potential_term(i, Nsite)
    
    return Mu

def Fermi_Hubbard_Hamiltonian(t, U, mu, N, M, periodic=False, **params):
    """
    returns the Hamiltonian for the NxM
    Fermi-Hubbard model.
    """
    Nsite = int(N*M)
    hopping_couplings = NxM_nearest_coupling_list(N, M, periodic=periodic)

    H_T  = t * hopping_Hamiltonian(hopping_couplings, Nsite)
    H_U  = U * interaction_Hamiltonian(Nsite)
    H_Mu = mu * chemical_potential_Hamiltonian(Nsite)

    return H_T + H_U - H_Mu


def total_spin(Nsite):
    """
    returns the total spin Hamiltonian 
    for Nsite.
    """
    Norb = 2*Nsite
    Sz   = Qobj(dims=[[2 for j in range(Norb)] for i in range(2)])
    for i in range(Nsite):
        Sz += spin_on_site(i, Nsite)
    
    return Sz

def spin_on_site(i, Nsite):
    """
    returns ni_up - ni_down
    for a system with Nsite sites.
    """
    ai_up  = f_destroy(i, 2*Nsite)
    adi_up = ai_up.dag()
    ni_up  = adi_up * ai_up

    ai_down  = f_destroy(i+Nsite, 2*Nsite)
    adi_down = ai_down.dag()
    ni_down  = adi_down * ai_down

    return ni_up - ni_down


######### Pauli Hamiltonian #########

def V_weighted_Pauli(U, N, M):
    
    Nsite = N*M
    l = int(2*Nsite)
    Pauli_V   = []
    weights_V = [(U/4+0.j)]*Nsite
    
    for i in range(Nsite):
        Plist = ['I'] * l
        
        Plist[i] = 'Z'
        Plist[i+Nsite] = 'Z'
        
        Pstring = ''.join(Plist)
        Pauli_V.append(Pauli(label=Pstring))
    
    return Pauli_V, weights_V

def T_weighted_Pauli(t, N, M, periodic=False):
    
    hopping_couplings = NxM_nearest_coupling_list(N, M, periodic=periodic)
    Nsite = N*M
    l = int(2*Nsite)
    Pauli_T   = []
    
    for group in hopping_couplings:
        for term in group:
            
            Plist_XX_up = ['I'] * l
            Plist_YY_up = ['I'] * l
            
            Plist_XX_down = ['I'] * l
            Plist_YY_down = ['I'] * l
            
            term = np.sort(term)
            num_Z = abs(term[1]-term[0]) - 1
            num_qb = abs(term[1]-term[0]) + 1
            stringXX = ['X'] + ['Z']*num_Z + ['X']
            stringYY = ['Y'] + ['Z']*num_Z + ['Y']
            
            # ups
            Plist_XX_up[term[0]:term[1]+1] = stringXX
            Plist_YY_up[term[0]:term[1]+1] = stringYY

            Pstring_XX = ''.join(Plist_XX_up)
            Pstring_YY = ''.join(Plist_YY_up)
            
            Pauli_T.append(Pauli(label=Pstring_XX))
            Pauli_T.append(Pauli(label=Pstring_YY))
            
            # downs
            Plist_XX_down[term[0]+Nsite:term[1]+1+Nsite] = stringXX
            Plist_YY_down[term[0]+Nsite:term[1]+1+Nsite] = stringYY

            Pstring_XX = ''.join(Plist_XX_down)
            Pstring_YY = ''.join(Plist_YY_down)
            
            Pauli_T.append(Pauli(label=Pstring_XX))
            Pauli_T.append(Pauli(label=Pstring_YY))
            
    weights_T = [(t/2.+0.j)]*len(Pauli_T)
    
    return Pauli_T, weights_T

def FH_Paulis_Weights(t, U, mu, N, M, periodic=False):
    """
    Returns the weighted Pauli strings of the Fermi-Hubbard Hamiltonian.
    For now, works only at half-filling.
    """
    Nsite = N*M
    Norb = 2*Nsite

    Pauli_T, weights_T = T_weighted_Pauli(t, N, M, periodic=periodic)
    Pauli_V, weights_V = V_weighted_Pauli(U, N, M)
    
    Pauli_I  = [Pauli(label='I'*Norb)]
    Weight_I = [(-Nsite + 0.j)]
    
    Paulis  = Pauli_I + Pauli_T + Pauli_V
    weights = Weight_I + weights_T + weights_V
    
    return Paulis, weights

def Fermi_Hubbard_Hamiltonian_Pauli(t, U, mu, N, M, periodic=False, **params):
    """
    returns the Hamiltonian for the NxM
    Fermi-Hubbard model as Pauli terms.
    For now, works only at half-filling.

    Was tested, gives the same Hamiltonian Matrix
    """

    Paulis, weights = FH_Paulis_Weights(t, U, mu, N, M, periodic)
    
    Pauli_op = WeightedPauliOperator.from_list(Paulis, weights=weights)
    
    return Pauli_op


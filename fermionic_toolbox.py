#!/usr/bin/env python

from scipy.sparse import csr_matrix as sm
from scipy.sparse import kron as kron
from scipy.sparse import identity as sp_identity
from scipy.sparse import linalg
from scipy.sparse import diags
from scipy.linalg import eigh
import numpy as np
import itertools
import math

import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

import matrix_toolbox as mt


# ************ basic fermionic functions


def anticommutator(c1,c2):
    return c1*c2 + c2*c1

def commutator(c1,c2):
    return c1*c2 - c2*c1

def f_destroy(n, i, Nsite):
    """
    returns a fermionic descruction operator of a spin-i electron for the n'th site out of N_tot sites.
    assumes that the sites are ordered as:
    |site_0, site_1, site_2, .... , site_(Nsite - 2), site_(Nsite - 1)>
    """

    def sz_chain(n):
        """
        returns the tensor (Kronecker) product of n sigma Z 2 x 2 matrices
        sI = [[1, 0],
              [0, -1]]
        out = sI \otimes sI \otimes sI \otimes .....
        """
        sI = sm([[1, 0], [0, -1]])
        out = sI.copy()

        for x in range(n - 1):
            out = kron(out, sI)
        return out


    def id_chain(n):
        """
        returns the kronecker product of n 2 x 2 identity matrices
        """
        return sp_identity(2 ** n)

    def f_destroy_1site(i):
        """
        returns the anihilation operator of a spin-i electron for 1 site
        i = 0 is spin up
        i = 1 is spin down
        """
        c = sm([[0, 1], [0, 0]])
        if i==0:
            return kron(c, id_chain(1))
        elif i==1:
            return kron(sz_chain(1), c)
        else:
            raise ValueError('index has to be 0 for spin up and 1 for spin down')

    if n < 0 or n >= Nsite:
        raise ValueError('n has to be in [0, 1, ..., Nsite - 1]')
    if n == 0:
        """ 
        the factor 2 is there because f_destroy_1site(i) is 4x4 and id_chain returns a chain of 2x2
        matrices.
        """
        return sm(kron(f_destroy_1site(i), id_chain(2*(Nsite - 1))))
    elif n == Nsite - 1:
        return sm(kron(sz_chain(2*(Nsite - 1)), f_destroy_1site(i)))
    else:
        tmp = kron(sz_chain(2*n), f_destroy_1site(i))
        # this times indentity
        return sm(kron(tmp, id_chain(2*(Nsite - 1 - n))))

def f_create(n, i, Nsite):
    """
    hc of destroy
    """
    return f_destroy(n,i, Nsite).conj().transpose()

def total_electron_number_op(Nsite, **params):
    """
    returns N = n_1 + n_2 + ... +n_Nsite 
    where n_i = n_i_up + n_i_down
    """
    hilbert_space = 4**Nsite
    N = sm((hilbert_space,hilbert_space))

    for i in range(Nsite):

        nup   = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite)
        ndown = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite)
        ni    = nup+ndown

        N += ni

    return(N)

def total_up_electron_number_op(Nsite, **params):
    hilbert_space = 4**Nsite
    N = sm((hilbert_space,hilbert_space))

    for i in range(Nsite):

        nup   = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite)

        N += nup

    return(N)

def total_down_electron_number_op(Nsite, **params):
    hilbert_space = 4**Nsite
    N = sm((hilbert_space,hilbert_space))

    for i in range(Nsite):

        ndown = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite)

        N += ndown

    return(N)

def double_occupation_op(site, Nsite, **params):
    """
    returns the operator niup*nidown for site 'site'
    """
    niup   = f_create(site, 0, Nsite)*f_destroy(site, 0, Nsite)
    nidown = f_create(site, 1, Nsite)*f_destroy(site, 1, Nsite)

    return(niup*nidown)

def vacuum(Nsite, **params):
    """
    returns the vacuum state 
    """
    vac     = np.zeros(4**Nsite)
    vac[0] += 1
    return(vac) 

def Nsite_from_ket(ket):
    """
    returns the number of sites from a ket
    """
    return int(math.log(len(ket), 4))






# ************ Tools that relates to the Hubbard model

def hopping_matrix(Nsite, t, **params):
    """
    Returns the nearest neighbor hopping matrix for Nsite sites and strenght t.
    """
    diagonals = t * np.ones(Nsite-1)
    t_matrix  = diags(diagonals, offsets=-1)+diags(diagonals, offsets=1)
    return t_matrix.toarray()

def Fermi_Hubbard_H(t, U, mu, Nsite, half_filling, **params):
    """
    Returns the Fermi-Hubbard Hamiltonian (using the fermionic operator encoding).
    """ 
    #b = [0.0001, 0.00005]
    hilbert_space = 4**Nsite
    if half_filling:
        mu = U/2.
    t_matrix = hopping_matrix(Nsite, t)

    T  = sm((hilbert_space, hilbert_space))
    V  = sm((hilbert_space, hilbert_space))
    Mu = sm((hilbert_space, hilbert_space))
    #B  = sm((hilbert_space, hilbert_space))
    # loop over all the sites
    for i in range(Nsite):
        # Coulomb interaction term V
        V  += f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) * f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite)
        # chimical potential term Mu
        Mu += (f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) + f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite))
        for j in range(Nsite):
            for spin in range(2):
                # kinetic energy term T
                T += - t_matrix[i][j] * f_create(i, spin, Nsite) * f_destroy(j, spin, Nsite)
        #B += b[i]*(f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) - f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite))

                
    H = T + U*V - mu*Mu #+ B
    
    return H

def diagonalize_Fermi_Hubbard(t, U, mu, Nsite, half_filling, number_of_ekets=10, **params):
    """
    Returns the  diagonalization of the Fermi-Hubbard Hamiltonian
    """ 
    if(half_filling):
        mu = U/2.0
    H  = Fermi_Hubbard_H(t, U, mu, Nsite, half_filling)
    return mt.diagonalize_all_ekets(H)

def expect(operator, state):
    """
    Returns the expectation value of an operator
    """
    return(state.conj() @ operator @ state)

def up_electron_number_array(ket):
    """
    Returns an array of the occupation for spin up electrons for all the sites
    in the state ket. 
    """ 
    Nsite = Nsite_from_ket(ket)
    nup   = []
    for i in range(Nsite):
        # create operators for site i
        niup_op   = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) # 0 for spin up
        
        # compute occupancy for site i
        niup   = expect(niup_op, ket)
        
        # append list of all sites
        nup.append(niup)
    
    return nup

def down_electron_number_array(ket):
    """
    Returns an array of the occupation for spin down electrons for all the sites
    in the state ket. 
    """
    Nsite = Nsite_from_ket(ket)
    ndown = []
    for i in range(Nsite):
        # create operators for site i
        nidown_op = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite) # 1 for spin down
        
        # compute occupancy for site i
        nidown = expect(nidown_op, ket)
        
        # append list of all sites
        ndown.append(nidown)
    
    return ndown

def all_electron_number_array(ket):
    """
    Returns the occupation for spin ups + spin downs electrons for all the sites
    in the state ket. 
    """ 
    Nsite = Nsite_from_ket(ket)
    n = []
    for i in range(Nsite):
        # create operators for site i
        niup_op   = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) # 0 for spin up
        nidown_op = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite) # 1 for spin down
        ni_op     = niup_op + nidown_op
        
        # compute occupancy for site i
        ni   = expect(ni_op, ket)
        
        # append list of all sites
        n.append(ni)
    
    return n

def double_occupancy_array(ket):
    """
    Returns the double occupancy nup*ndown for all the sites
    in the state ket. 
    """ 
    Nsite            = Nsite_from_ket(ket)
    double_occupancy = []
    for i in range(Nsite):
        # create operators for site i
        niup_op      = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) # 0 for spin up
        nidown_op    = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite) # 1 for spin down
        nup_ndown_op = niup_op * nidown_op
        
        # compute occupancy for site i
        nup_ndown    = expect(nup_ndown_op, ket)
        
        # append list of all sites
        double_occupancy.append(nup_ndown)
    
    return double_occupancy


def local_moment_array(ket):
    """
    Returns the local moment (nup-ndown)**2 for all the sites
    in the state ket. 
    """ 
    Nsite            = Nsite_from_ket(ket)
    local_moment = []
    for i in range(Nsite):
        # create operators for site i
        niup_op      = f_create(i, 0, Nsite) * f_destroy(i, 0, Nsite) # 0 for spin up
        nidown_op    = f_create(i, 1, Nsite) * f_destroy(i, 1, Nsite) # 1 for spin down
        local_moment_op = (niup_op - nidown_op)**2
        
        # compute occupancy for site i
        m2    = expect(local_moment_op, ket)
        
        # append list of all sites
        local_moment.append(m2)
    
    return local_moment



def total_electron_number(ket):
    """
    returns the total number of electrons N in a state ket
    """
    Nsite = Nsite_from_ket(ket)
    N_op = total_electron_number_op(Nsite)
    return(expect(N_op, ket))

def occupancy_sweep(sweep, energy_level=0, **params):
    """
    Returns the occupancy of each site of the lattice at a given energy_level for all the values
    in the sweep
    """
    Nsite          = params['Nsite']
    sweep_lenght   = len(sweep['vector'])
    occupancy_up   = []
    occupancy_down = []
    
    for sweep_variable in range(sweep_lenght):
        ket = sweep['result'][sweep_variable][1][:,energy_level]
        occupancy_up.append(up_electron_number_array(ket))
        occupancy_down.append(down_electron_number_array(ket))

    return occupancy_up, occupancy_down

def double_occupancy_sweep(sweep, energy_level=0, **params):
    """
    Returns the double occupancy of each site of the lattice at a given energy_level for all the values
    in the sweep
    """
    Nsite            = params['Nsite']
    sweep_lenght     = len(sweep['vector'])
    double_occupancy = []
    
    for sweep_variable in range(sweep_lenght):
        ket = sweep['result'][sweep_variable][1][:,energy_level]
        double_occupancy.append(double_occupancy_array(ket))

    return double_occupancy






# *********** Thermodynamics

def partition_function(all_evals, beta):
    Z = 0
    for evals in all_evals:
        Z += np.exp(- beta * evals)
    return Z

def thermo_expect(operator, all_evals, all_ekets, beta):
    """
    returns the thermodynamic expectation value of an operator
    """
    Z = partition_function(all_evals, beta)

    tmp = 0
    for i in range(len(all_evals)):
        tmp += np.exp(- beta * all_evals[i]) * expect(operator, all_ekets[:,i])

    return tmp / Z





# ************ Green's function

def Greens_function_ij_w(site1, spin1, site2, spin2, evals, ekets, omega, eta):
    """
    compute one Greens function matrix element
    """
    number_of_ekets = int(len(evals))
    ground_state    = ekets[:,0]
    E0              = evals[0]
    Nsite           = Nsite_from_ket(ground_state)
    cdi = f_create(site1, spin1, Nsite)
    ci  = f_destroy(site1, spin1, Nsite)
    cdj = f_create(site2, spin2, Nsite)
    cj  = f_destroy(site2, spin2, Nsite)

    Ge = 0.+0.j
    Gh = 0.+0.j
    j  = 0.+1.j

    for m in range(number_of_ekets):
        ketm = ekets[:,m]
        Em   = evals[m]

        tmp1 = ground_state.conj() @ ci @ ketm
        tmp2 = ketm.conj() @ cdj @ ground_state
        tmp3 = 1/(omega - eta*j - Em + E0)


        Ge += tmp1 * tmp2 * tmp3

    for m in range(number_of_ekets):
        ketm = ekets[:,m]
        Em   = evals[m]

        tmp1 = ground_state.conj() @ cj @ ketm
        tmp2 = ketm.conj() @ cdi @ ground_state
        tmp3 = 1/(omega - eta*j + Em - E0)

        Gh += tmp1 * tmp2 * tmp3

    return Ge + Gh

def Greens_function_matrix(omega, evals, ekets, eta):
    """
    compute the Greens function matrix at a given energy omega
    """
    Nsite = Nsite_from_ket(ekets[:,0])

    G = np.zeros((Nsite,2,Nsite,2), dtype=complex)

    for site1 in range(Nsite):
        for spin1 in range(2):
            for site2 in range(Nsite):
                for spin2 in range(2):
                    G[site1][spin1][site2][spin2] = Greens_function_ij_w(site1, spin1, site2, spin2, 
                        evals, ekets, omega, eta)

    return G






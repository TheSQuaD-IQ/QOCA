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

import fermionic_toolbox as ft
import matrix_toolbox as mt



def N_particles_basis_vector(Nparticles, Nsite, transpose = False, **params):
    
    def kbits(n, k):
        result = []
        for bits in itertools.combinations(range(n), k):
            s = ['0'] * n
            for bit in bits:
                s[bit] = '1'
            result.append(' '.join(s))
        return result

    n = int(2 * Nsite)
    k = int(Nparticles)
    if (k > n):
        return "you can't place more electrons than there are available spots, dumbass."
    
    basis_vectors_string = kbits(n, k)
    basis_vectors_array  = np.array([np.fromstring(vec, dtype=float, sep=' ') for vec in basis_vectors_string])

    if transpose:
        return basis_vectors_array.T
    else:
        return basis_vectors_array

def number_of_basis_vector(Nparticles, Nsite):
    N = Nparticles
    M = Nsite
    
    return int((math.factorial(2*M))/((math.factorial(N))*(math.factorial(2*M-N))))


def create(site, spin, ket):
    """
    Operator that creates a fermion of spin 'spin' on site 'site'.
    site goes from 0 to Nsite-1 and spin up = 0, down = 1
    """

    Nsite = len(ket)/2
    
    if site>Nsite-1:
        return('error! site number > Nsite')
    elif site<0:
        return('error! site indices start at 1')
    elif spin < 0 or spin > 1:
        return('spin must be 0 for up and 1 for down')
    
    idx   = int(((spin*(Nsite)) + site))
    
    if idx==0:
        exp = 0
    else:
        exp   = int(np.cumsum(ket)[idx-1])
    phase = int(((-1.)**exp)*(1.-ket[idx]))
    
    new_ket = ket
    new_ket[idx]+=1.
    
    return phase , new_ket

def destroy(site, spin, ket):
    """
    Operator that destroys a fermion of spin 'spin' on site 'site'.
    site goes from 0 to Nsite-1 and spin up = 0, down = 1 
    """

    Nsite = len(ket)/2
    
    if site>Nsite-1:
        return('error! site number > Nsite')
    elif site<0:
        return('error! site indices start at 1')
    elif spin < 0 or spin > 1:
        return('spin must be 0 for up and 1 for down')
    
    idx   = int(((spin*(Nsite)) + site))
    
    if idx==0:
        exp = 0
    else:
        exp   = int(np.cumsum(ket)[idx-1])
    phase = int(((-1.)**exp)*(ket[idx]))
    
    new_ket = ket
    new_ket[idx]-=1.
    
    return phase , new_ket

def destroy_mat(site, spin, Nsite, Nparticles, **params):
    
    N_basis_vectors   = N_particles_basis_vector(Nparticles, Nsite, **params)
    Nm1_basis_vectors = N_particles_basis_vector(Nparticles-1, Nsite, **params)
    N_tags            = np.array([ket_to_tag(bv) for bv in N_basis_vectors])
    Nm1_tags          = np.array([ket_to_tag(bv) for bv in Nm1_basis_vectors])
    N_hilbert_space   = len(N_basis_vectors)
    Nm1_hilbert_space = len(Nm1_basis_vectors)
    
    T  = sm((Nm1_hilbert_space, N_hilbert_space))
    v  = 0
    for basis_vector in N_basis_vectors:
        phase, ket = destroy(site, spin, np.array(basis_vector))
        if phase != 0:
            tag     = ket_to_tag(ket)
            u       = int(np.where(Nm1_tags==tag)[0])
            T[u,v] += phase
        v += 1               
    
    return T

def create_mat(site, spin, Nsite, Nparticles, **params):             
    
    return destroy_mat(site, spin, Nsite, Nparticles + 1, **params).conj().transpose()


def ket_to_tag(ket):
    """
    inputs a ket and outputs its tag (base 2 to base 10 conversion)
    """
    ket_int    = ket.astype(int)
    bit_string = ''.join(map(str, ket_int))
    tag        = int(bit_string, 2)
    
    return tag
    
def tag_to_ket(tag):
    """
    inputs a tag and outputs its ket (base 10 to base 2 conversion)
    """
    return np.array([int(digit) for digit in bin(tag)[2:]], dtype=float)

def cd_c(site1, site2, spin, ket):
    """
    operators of the hopping term of the Hubbard Hamiltonian
    """
    ket0 = np.array(ket)
    phase1, ket1 = destroy(site2, spin, ket0)
    phase2, ket2 = create(site1, spin, ket1)
    
    return phase1*phase2, ket2

def H_hopping(Nsite, t, Nparticles, **params):
    
    t_mat         = ft.hopping_matrix(Nsite, t, **params)
    basis_vectors = N_particles_basis_vector(Nparticles, Nsite, **params)
    tags          = np.array([ket_to_tag(bv) for bv in basis_vectors])
    hilbert_space = len(basis_vectors)
    
    T  = sm((hilbert_space, hilbert_space))
    v  = 0
    for basis_vector in basis_vectors:
        for site1 in range(Nsite):
            for site2 in range(Nsite):
                for spin in range(2):
                    if t_mat[site1][site2] != 0:
                        phase, ket = cd_c(site1, site2, spin, np.array(basis_vector))
                        if phase != 0:
                            phase *= -t_mat[site1][site2]
                            tag    = ket_to_tag(ket)
                            u      = int(np.where(tags==tag)[0])
                            T[u,v] = phase
        v += 1               
    
    return T

def H_diag(Nsite, Nparticles, **params):
    
    U  = params['U']
    mu = params['mu']
    
    basis_vectors = N_particles_basis_vector(Nparticles, Nsite, **params)
    tags          = np.array([ket_to_tag(bv) for bv in basis_vectors])
    hilbert_space = len(basis_vectors)
    
    H_U  = sm((hilbert_space, hilbert_space))
    H_mu = sm((hilbert_space, hilbert_space))
    v  = 0
    for basis_vector in basis_vectors:
        U_term  = 0.
        mu_term = 0.
        for site in range(Nsite):
            n_site_up   = basis_vector[site]
            n_site_down = basis_vector[Nsite+site]
            U_term  += n_site_up * n_site_down
            mu_term += n_site_up + n_site_down
        if U_term != 0:
            U_term *= U
            H_U[v,v] = U_term
        if mu_term != 0:
            mu_term *= mu
            H_mu[v,v] = mu_term
        v += 1               
    
    return H_U - H_mu

def Hamiltonian(**params):
    H_kin = H_hopping(**params)
    H_diagonal = H_diag(**params)
    H = H_kin + H_diagonal
    return H


def Gij_w(site1, spin1, site2, spin2, evals_N, ekets_N, evals_Nm1, ekets_Nm1, 
    evals_Np1, ekets_Np1, omega, eta, Nsite, Nparticles, **params):
    """
    compute one Greens function matrix element
    """
    ground_state    = ekets_N[:,0]
    E0              = evals_N[0]
    cdi = create_mat(site1, spin1, Nsite, Nparticles)
    cj  = destroy_mat(site2, spin2, Nsite, Nparticles +1)
    cdj = create_mat(site2, spin2, Nsite, Nparticles)
    cdj2 = create_mat(site2, spin2, Nsite, Nparticles -1)
    ci  = destroy_mat(site1, spin1, Nsite, Nparticles +1)
    ci2  = destroy_mat(site1, spin1, Nsite, Nparticles)

    Ge = 0.+0.j
    Gh = 0.+0.j
    j  = 0.+1.j

    for m in range(len(evals_Nm1)):
        ketm = ekets_Nm1[:,m]
        Em   = evals_Nm1[m]

        tmp1 = ground_state.conj() @ cdj2 @ ketm
        tmp2 = ketm.conj() @ ci2 @ ground_state
        tmp3 = 1/(omega - eta*j + Em - E0)
        
        Ge += tmp1 * tmp2 * tmp3

    for m in range(len(evals_Np1)):
        ketm = ekets_Np1[:,m]
        Em   = evals_Np1[m]

        tmp1 = ground_state.conj() @ ci @ ketm
        tmp2 = ketm.conj() @ cdj @ ground_state
        tmp3 = 1/(omega - eta*j - Em + E0)
        
        Gh += tmp1 * tmp2 * tmp3

    return Ge + Gh



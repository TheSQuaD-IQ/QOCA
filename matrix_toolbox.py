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



# ************ Tools to play with states
def normalize(ket):
    norm = np.linalg.norm(ket)
    return ket/norm


# ************ Tools to do matrix calculations

def diagonalize_all_ekets(operator, zero_energy=False):
    """
    Diagonalizes an operator. Returns all the eignevalues and the eigenvectors.
    I need this function because ft.diagonalise won't give me ALL the eigen vectors.
    """
    evals, ekets = eigh(operator.toarray(), b=None, type=1)
    indx         = evals.argsort()
    evals_s      = np.sort(evals)
    if zero_energy:
        evals_s  = evals_s - evals_s[0]
    ekets_s      = np.zeros(ekets.shape, dtype=ekets.dtype)
    for i in range(len(evals)):
        ekets_s[:,i] = ekets[:,indx[i]]
    return evals_s, ekets_s

def diagonalize(operator, number_of_ekets=10, tol=1e-10, ground_guess=None, zero_energy=False):
    """
    Diagonalizes an operator (scipy csr_matrix format). Returns the eignevalues and the eigenvectors
    """
    evals, ekets = linalg.eigsh(operator, k=number_of_ekets, which='SA', tol=tol, v0=ground_guess)
    indx         = evals.argsort()
    evals_s      = np.sort(evals)
    if zero_energy:
        evals_s  = evals_s - evals_s[0]
    ekets_s      = np.zeros(ekets.shape, dtype=ekets.dtype)
    for i in range(number_of_ekets):
        ekets_s[:,i] = ekets[:,indx[i]]
    return evals_s, ekets_s





# ************ Tools to do parameter sweeps

def sweep(function, sweep_variable, sweep_vector, **params):
    sweep_results = []
    params_copy = params.copy() # copy original params values
    for sweep_point in sweep_vector:
        params_copy[sweep_variable] = sweep_point
        sweep_results.append(function(**params_copy))
    return {'result': sweep_results, 'variable': sweep_variable, 'vector': sweep_vector}






# ************ Tools for plotting quantum quantities


def sliding_mean(data_array, window=50):  

    # This function takes an array of numbers and smoothes them out.  
    # Smoothing is useful for making plots a little easier to read.

    data_array = np.array(data_array)  
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        avg = 0  
        for j in indices:  
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)  
          
    return np.array(new_list)  


def crosscut(curves, step=None):

    """
    returns the crosscut of a bunch of curves at x = step.
    If no x is specified, the last value is taken.
    Curves must be a numpy array of same lenght curves.
    """
    
    if step is None or step==len(curves[0]):
        step = len(curves[0])-1
    
    return curves[:,step]

   
def eff_vs_it(data, theory, convergence_criteria, **criteria_args):

    """
    Returns the 
    'efficiency' = (nb of curves that that converged w.r.t. some convergence_criteria) / (total nb of curves)
    of a collection of convergence curves (data) 
    """
    
    def within_Xperc(data_point, theory, percentage):
        upper_bound = theory + (percentage/100.)*abs(theory)
        lower_bound = theory - (percentage/100.)*abs(theory)

        if lower_bound <= data_point <= upper_bound:
            return True
        else:
            return False
    
    def within_range(data_point, theory, range_value):
        upper_bound = theory + range_value
        lower_bound = theory - range_value

        if lower_bound <= data_point <= upper_bound:
            return True
        else:
            return False

    num_it_tot = len(data[0]) # total number of iteration
    num_traj_tot = len(data)  # total number of curves
    
    nb_that_converged = np.zeros(num_it_tot)
    for it in range(num_it_tot):
        for traj in range(num_traj_tot):
            data_point = data[traj][it]
            if convergence_criteria(data_point, theory, **criteria_args):
                nb_that_converged[it]+=1
    return nb_that_converged/num_traj_tot


def plot_spectrum(sweep, k=10):
    """
    Plots the energy spectrum as a function of a certain sweep variable
    """
    fig, ax = plt.subplots()
    for i in range(k):
        ax.plot(sweep['vector'], [evals[i] for evals, ekets in sweep['result']], 
            linestyle='-', label=r'$%d$' % i, alpha=0.5, linewidth=1., marker='o')

    ax.legend(loc='best')
    ax.set_xlabel(r'%s' % sweep['variable'])
    ax.set_ylabel(r'$E$')
    ax.set_title('Spectrum')
    plt.show()

def plot_occupancy(sweep, energy_level=0, separate_spins = False, **params):
    """
    Plots the occupancy for all the sites as a function of a certain sweep variable
    """
    nups, ndowns = occupancy_sweep(sweep, energy_level, **params)
    
    Nsite = params['Nsite']
    fig, ax = plt.subplots()
    if separate_spins:
        for site in range(Nsite):
            ax.plot(sweep['vector'], [onevalue[site] for onevalue in nups], linestyle='-', 
                    label=r'site $%d$, spin up' % (site), alpha=0.5, linewidth=1., marker='o')
            ax.plot(sweep['vector'], [onevalue[site] for onevalue in ndowns], linestyle='-', 
                    label=r'site $%d$, spin down' % (site), alpha=0.5, linewidth=1., marker='o')
    else:
        for site in range(Nsite):
            ax.plot(sweep['vector'], [nups[U][site] + ndowns[U][site] for U in range(len(sweep['vector']))], 
                linestyle='-', label=r'site $%d$' % site, alpha=0.5, linewidth=1., marker='o')

    ax.legend(loc=1)
    ax.set_xlabel(r'%s' % sweep['variable'])
    ax.set_ylabel(r'$N$')
    ax.set_title('Occupancy')
    plt.show()

def plot_double_occupancy(sweep, energy_level=0, **params):
    """
    Plots the double occupancy (niup*nidown) for all the sites as a function of a certain sweep variable
    """
    double_occupancy = double_occupancy_sweep(sweep, energy_level, **params)
    
    Nsite = params['Nsite']
    fig, ax = plt.subplots()
    for site in range(Nsite):
        ax.plot(sweep['vector'], [onevalue[site] for onevalue in double_occupancy], 
            linestyle='-', label=r'site $%d$' % site, alpha=0.5, linewidth=1., marker='o')

    ax.legend(loc=1)
    ax.set_xlabel(r'%s' % sweep['variable'])
    ax.set_ylabel(r'$n_{i,up}n_{i,down}$')
    ax.set_title('Double occupancy')
    plt.show()

def plot_sweep(sweep, other_curve = None, title = None):
    x       = sweep['vector']
    y       = sweep['result']
    x_label = sweep['variable']
    
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle='-', alpha=0.5, linewidth=1., marker='o', label='sweep results')
    if other_curve:
        ax.plot(x, other_curve, linestyle='-', alpha=0.5, linewidth=1., marker='o', label='other curve')
    ax.set_xlabel(x_label)
    if title:
        ax.set_ylabel(title)
        ax.set_title(title)
    else:
        ax.set_ylabel('Result')
        ax.set_title('Results')
    ax.legend(loc=0)
    ax.grid(True)
    plt.show()

def plot(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle='-', alpha=0.5, linewidth=1., marker='o')
    ax.grid(True)
    plt.show()

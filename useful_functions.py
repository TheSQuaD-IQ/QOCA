#!/usr/bin/env qiskit0.11

import matplotlib.pyplot as plt
import numpy as np 
import pickle
from qutip import *
import matplotlib as mpl
from scipy import sqrt
from qiskit import *
from imports import *
from myvarforms.gates import *
import math
import gzip

# Load a pkl file
def load_obj(file_name):
    # with gzip.open(file_name, 'rb') as f:
    with open(file_name, 'rb') as f:
        return pickle.load(f)


############# PLOT #############

def nice_plots():
    """
    Set pyplot parameters for nice plots
    """ 

    COLOR = '#393e41'   

    fig_width_pt = 2*246.0                  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    # Explicitly set fontsizes:
    font_size = 18
    tick_size = 16
    params = {
              #'backend': 'ps',
              'axes.labelsize':  font_size,
              'axes.labelcolor': COLOR,
              'axes.spines.right': False,    # Toggle right axis
              'axes.spines.top':   False,    # Toggle top axis
              'axes.edgecolor':   COLOR,
              'font.size':       font_size, # UserWarning: text.fontsize is deprecated and replaced with font.size
              'legend.fontsize': 12,
              'xtick.labelsize': tick_size,
              'xtick.color':     COLOR,
              'ytick.labelsize': tick_size,
              'ytick.color':     COLOR,
              'text.color':      COLOR,
              'text.usetex':     False,
              'figure.figsize':  fig_size,
              #'font.family':     'sans-serif',
              #'font.sans-serif': 'Kailasa'
              }
    mpl.rcParams.update(params)
    adjustprops = dict(right=0.95, left=0.14, bottom=0.17, top=0.93, 
                       wspace=0.0, hspace=0.0)  

    figprops = dict(figsize=(1.0*fig_width,1.0*fig_height)) 

    def subplots_adjust(fig, props):
        fig.subplots_adjust(left=props['left'])
        fig.subplots_adjust(right=props['right'])
        fig.subplots_adjust(bottom=props['bottom'])
        fig.subplots_adjust(top=props['top'])
        fig.subplots_adjust(wspace=props['wspace'])
        fig.subplots_adjust(hspace=props['hspace'])

# simple plot
def plot(x, ys):
    nice_plots()
    fig, ax = plt.subplots()
    for y in ys:
        ax.plot(x, y, linestyle='-', alpha=0.5, linewidth=1., marker='o')
    ax.grid(True)
    plt.show()

def loglog_plot(x, ys):
    nice_plots()
    fig, ax = plt.subplots()
    for y in ys:
        ax.loglog(x, y, linestyle='-', alpha=0.5, linewidth=1., marker='o')
    ax.grid(True)
    plt.show()


############# QUTIP #############

# Qutip to Qiskit operator conversion
def qt2qk(op):
    return op.permute(list(range(len(op.dims[0])))[::-1])

# Qutip support functions
def compose(operator, index, dims): 
    '''
    Composes Qobj quantum operators in a multiqubit Hilbert space.
    '''
    #assert isinstance(operator, Qobj), 'Operator is not of type qutip.qobj.Qobj!'
    op_list        = [qeye(dims[mode]) for mode in range(len(dims))]
    op_list[index] = operator

    return tensor(op_list)

def compose_JW_strings(op1, op2, idx1, idx2, dims):
    '''
    Compose a two-qubit operator string with Zs in between
    ex: IIIXZZZYII for op1=X, op2=Y, idx1=3, idx2=7 
    '''
    #assert isinstance(op1, Qobj), 'Operator1 is not of type qutip.qobj.Qobj!'
    #assert isinstance(op2, Qobj), 'Operator2 is not of type qutip.qobj.Qobj!'
    op_list     = [qeye(dims[mode]) for mode in range(len(dims))]
    Zstring     = [sigmaz() for n in range(abs(idx2-idx1)-1)]
    
    if idx1<idx2:
        op_list[idx1+1:idx2] = Zstring
    elif idx1>idx2:
        op_list[idx2+1:idx1] = Zstring
    op_list[idx1] = op1
    op_list[idx2] = op2
    
    return tensor(op_list)

def parity_op(Nsite):
    '''
    Returns the parity operator $\sum_{k=0}^{Nsite-1} \sigma^\dagger \sigma$
    for a system of Nsite spins.
    '''
    sigma = sigmap()
    sigmad = sigma.dag()
    dims = [2 for i in range(Nsite)]
    parity_op = Qobj(dims=[dims for i in range(2)])
    for k in range(Nsite):
        parity_op += compose(sigmad*sigma, k, dims)
    return parity_op


def FFFT(Nqubit):
    id2 = qeye(2)
    cz = csign()
    fswap = cz * swap()
    sigma = sigmap()
    sigmad = sigma.dag()
    X = sigmax()
    Y = sigmay()
    Z = sigmaz()
    XX = tensor(X,X)
    YY = tensor(Y,Y)
    Zp = tensor(Z,id2)
    Zq = tensor(id2,Z)

    F0qb = ((-1.j*np.pi/4*Zq).expm()
            *(-1.j*np.pi/8*XX).expm()
            *(-1.j*np.pi/8*YY).expm()
            *(-1.j*np.pi/4*Zq).expm())

    FFFT4qb = (tensor(id2, fswap, id2)
                *tensor(F0qb,F0qb)
                *tensor(id2,id2,id2,rotation(Z, np.pi/2))
                *tensor(id2, fswap, id2)
                *tensor(F0qb,F0qb)
                *tensor(id2,fswap,id2))

    N = int(Nqubit)

    if N==2: return F0qb
    elif N==4: return FFFT4qb
    else: 
        raise ValueError('You have not coded the general FFFT you dumbass. You can only use N=2 or N=4')


def ket2Norb(ket):
    """
    returns the number of spin orbitals from a statevector (numpy array)
    """
    return int(math.log(len(ket), 2))

def Qobj_op_dims(num_qubits):
    """
    returns the dimensions for a Qobj operator of num_qubits qubits
    """
    return [[2 for j in range(num_qubits)] for i in range(2)]

def Qobj_ket_dims(num_qubits):
    """
    returns the dimensions for a Qobj statevector of num_qubits qubits
    """
    return [[2 for j in range(num_qubits)], [1 for j in range(num_qubits)]]

def toQobj(array, type='ket'):
    """
    converts a numpy array into a Qobj
    """
    if type is 'ket':
        num_qubits = ket2Norb(array)
        dimensions = Qobj_ket_dims(num_qubits)
        return Qobj(array, dims=dimensions)
    elif type is 'operator':
        num_qubits = ket2Norb(array)
        dimensions = Qobj_op_dims(num_qubits)
        return Qobj(array, dims=dimensions)
    else: raise ValueError('Type must be ket or operator')


def tag_to_basis_vector(tag, Norbitals):
    """
    inputs a tag (base 10) and outputs its basis vector (base 10 to base 2 conversion)
    Ex: 4 --> [0. 1. 0. 0.]
    """
    basis_vec = np.zeros(Norbitals) # all zeros for now
    # find binary rep of tag. Ex 4 = 100
    binary_tag = np.array([int(digit) for digit in bin(tag)[2:]], dtype=float)
    
    # expand binary_tag to the correct number of orbitals 100->0100
    basis_vec[-len(binary_tag):] = binary_tag
    return basis_vec
    
def ket2dict(ket):
    """
    returns a dictionary of the nonzero elements of a ket
    in the form {'0011': coeff, '0010': coeff}
    """
    num_qubits = int(np.log2(ket.shape[0]))
    indices = np.nonzero((ket).data.toarray())[0]
    ket_dict = {}
    
    for idx in indices:
        coeff = ket[idx][0][0]
        basis_vector = tag_to_basis_vector(idx, num_qubits)
        
        ket_dict[str(basis_vector)] = coeff
        
    return ket_dict


############# Qiskit #############


def circuit2statevector(circuit, convert2Qobj=True):
    """
    Returns the statevector produced by a Qiskit quantum circuit
    """
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    results = job.result()
    state = results.get_statevector()

    if convert2Qobj: return  toQobj(state)
    else: return state
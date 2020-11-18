import os
import sys
sys.path.insert(0, os.getcwd())

# General
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import useful_functions as uf
from useful_functions import *
from numpy import pi
import timeit
import time

# Qutip
import qutip as qt
from qutip import *


# Classical fermionic
import hubbard as hb
from hubbard import *

# Qiskit
import qiskit as qk
from qiskit import IBMQ, Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
# from qiskit.compiler import transpile
import qiskit.tools.visualization as qv
from qiskit.tools.visualization import circuit_drawer
from qiskit.providers.aer import noise

# Qiskit Aqua
from qiskit.aqua import *
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.optimizers import COBYLA, SPSA, ADAM, L_BFGS_B, P_BFGS
from qiskit.aqua.components.variational_forms import *
# from qiskit.aqua.components.variational_forms.gates import *
from qiskit.aqua.components.initial_states import *
from qiskit.aqua.circuits import FourierTransformCircuits as ftc
from qiskit.aqua.operators import MatrixOperator
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.operators import Z2Symmetries

from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock


from myvarforms import *


# Qiskit Chemistry

import logging
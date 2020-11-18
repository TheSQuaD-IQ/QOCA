import numpy as np
from qiskit import *
from qiskit.circuit import ParameterExpression


def expPauli_circuit(theta, Pauli_string):
    """
    Return the circuit for exp(- I theta [Pauli_string])
    theta (float): angle of rotation
    Pauli_string (list or array): ['X','Y','Z', ...] 
    """

    Pauli_string = np.array(Pauli_string)
    num_qubits = len(Pauli_string)
    nonI_idx = np.where(Pauli_string != 'I')[0]

    qr = QuantumRegister(num_qubits)
    circ = QuantumCircuit(qr, name='exp(- i {} {})'.format(theta,Pauli_string))

    # Basis change
    for i in np.where(Pauli_string == 'X')[0]:
        circ.h(qr[int(i)])

    for i in np.where(Pauli_string == 'Y')[0]:
        circ.rx(np.pi/2., qr[int(i)])

    # CNOTS
    for i in range(len(nonI_idx)-1):
        circ.cx(qr[int(nonI_idx[i])],qr[int(nonI_idx[i+1])])

    # Rz(theta)
    circ.rz(2*theta,qr[int(nonI_idx[-1])])

    # CNOTS
    for i in reversed(range(len(nonI_idx)-1)):
        circ.cx(qr[int(nonI_idx[i])],qr[int(nonI_idx[i+1])])

    # Basis change
    for i in np.where(Pauli_string == 'X')[0]:
        circ.h(qr[int(i)])

    for i in np.where(Pauli_string == 'Y')[0]:
        circ.rx(-np.pi/2., qr[int(i)])

    # convert to gate
    return circ.to_instruction()


def XX_YY(theta, nb_qb):

    qr = QuantumRegister(nb_qb)
    formatted = theta if isinstance(theta, ParameterExpression) else np.round(theta, 3)
    circ = QuantumCircuit(qr, name='exp(- i XX + YY {})'.format(formatted))

    numZ = nb_qb-2
    stringXX = ['X'] + ['Z']*numZ + ['X']
    stringYY = ['Y'] + ['Z']*numZ + ['Y']

    circ.append(expPauli_circuit(theta, stringXX), [qr[i] for i in range(nb_qb)])
    circ.append(expPauli_circuit(theta, stringYY), [qr[i] for i in range(nb_qb)])

    return circ.to_instruction()


def XX_gate(theta):
    """
    returns the circuit for exp[-I \theta XX]
    """
    xx_qr = QuantumRegister(2)
    xx_circ = QuantumCircuit(xx_qr, name='XX({})'.format(theta))

    xx_circ.h(xx_qr[0])
    xx_circ.h(xx_qr[1])
    xx_circ.cx(xx_qr[0],xx_qr[1])
    xx_circ.rz(2*theta,xx_qr[1])
    xx_circ.cx(xx_qr[0],xx_qr[1])
    xx_circ.h(xx_qr[0])
    xx_circ.h(xx_qr[1])

    # convert to gate
    return xx_circ.to_instruction()

def XZsX_gate(theta,L):
    """
    returns the circuit for exp[-I \theta XZ...ZX]
    for a L qubit register.
    """
    xx_qr = QuantumRegister(L)
    xx_circ = QuantumCircuit(xx_qr, name='XZsX({})'.format(theta))

    xx_circ.h(xx_qr[0])
    xx_circ.h(xx_qr[L-1])
    for i in range(L-1):
        xx_circ.cx(xx_qr[i],xx_qr[i+1])
    xx_circ.rz(2*theta,xx_qr[L-1])
    for i in reversed(range(L-1)):
        xx_circ.cx(xx_qr[i],xx_qr[i+1])
    xx_circ.h(xx_qr[0])
    xx_circ.h(xx_qr[L-1])

    # convert to gate
    return xx_circ.to_instruction()

def YY_gate(theta):
    """
    returns the circuit for exp[-I \theta YY]
    """
    yy_qr = QuantumRegister(2)
    yy_circ = QuantumCircuit(yy_qr, name='YY({})'.format(theta))

    yy_circ.rx(np.pi/2., yy_qr[0])
    yy_circ.rx(np.pi/2., yy_qr[1])
    yy_circ.cx(yy_qr[0],yy_qr[1])
    yy_circ.rz(2*theta,yy_qr[1])
    yy_circ.cx(yy_qr[0],yy_qr[1])
    yy_circ.rx(-np.pi/2., yy_qr[0])
    yy_circ.rx(-np.pi/2., yy_qr[1])

    # convert to gate
    return yy_circ.to_instruction()

def YZsY_gate(theta,L):
    """
    returns the circuit for exp[-I \theta YZ...ZY]
    for a L qubit register.
    """
    yy_qr = QuantumRegister(L)
    yy_circ = QuantumCircuit(yy_qr, name='YZsY({})'.format(theta))

    yy_circ.rx(np.pi/2., yy_qr[0])
    yy_circ.rx(np.pi/2., yy_qr[L-1])
    for i in range(L-1):
        yy_circ.cx(yy_qr[i],yy_qr[i+1])
    yy_circ.rz(2*theta,yy_qr[L-1])
    for i in reversed(range(L-1)):
        yy_circ.cx(yy_qr[i],yy_qr[i+1])
    yy_circ.rx(-np.pi/2., yy_qr[0])
    yy_circ.rx(-np.pi/2., yy_qr[L-1])

    # convert to gate
    return yy_circ.to_instruction()

def ZZ_gate(theta):
    """
    returns the circuit for exp[-I \theta ZZ]
    """
    zz_qr = QuantumRegister(2)
    zz_circ = QuantumCircuit(zz_qr, name='ZZ({})'.format(theta))

    zz_circ.cx(zz_qr[0],zz_qr[1])
    zz_circ.rz(2*theta,zz_qr[1])
    zz_circ.cx(zz_qr[0],zz_qr[1])

    # convert to gate
    return zz_circ.to_instruction()

def F0_gate():
    """
    returns the circuit for the F0 gate
    for neighboring qubits
    """
    F0_qr = QuantumRegister(2)
    F0_circ = QuantumCircuit(F0_qr, name='F0')
    F0_circ.rz(np.pi/2,F0_qr[1])
    F0_circ.append(YY_gate(np.pi/8), [F0_qr[0],F0_qr[1]])
    F0_circ.append(XX_gate(np.pi/8), [F0_qr[0],F0_qr[1]])
    F0_circ.rz(np.pi/2,F0_qr[1])

    # convert to gate
    return F0_circ.to_instruction()

def Fk_gate(k,L):
    """
    returns the circuit for the Fk gate
    for neighboring qubits
    """
    Fk_qr = QuantumRegister(2)
    Fk_circ = QuantumCircuit(Fk_qr, name='F{},{}'.format(k,L))

    Fk_circ.rz((-2*np.pi*k)/L,Fk_qr[1])
    Fk_circ.append(F0_gate(),[Fk_qr[0],Fk_qr[1]])

    # convert to gate
    return Fk_circ.to_instruction()

def Fkd_gate(k,L):
    """
    returns the circuit for the Fk gate
    for neighboring qubits
    """
    Fk_qr = QuantumRegister(2)
    Fk_circ = QuantumCircuit(Fk_qr, name='Fd{},{}'.format(k,L))

    Fk_circ.append(F0_gate(),[Fk_qr[0],Fk_qr[1]])
    Fk_circ.rz((2*np.pi*k)/L,Fk_qr[1])

    # convert to gate
    return Fk_circ.to_instruction()

def fswap():
    """
    returns the circuit for the fermionic swap gate
    """
    fswap_qr = QuantumRegister(2)
    fswap_circ = QuantumCircuit(fswap_qr, name='fswap')

    # swap
    #fswap_circ.swap(fswap_qr[0],fswap_qr[1])
    fswap_circ.cx(fswap_qr[0],fswap_qr[1])
    fswap_circ.cx(fswap_qr[1],fswap_qr[0])
    fswap_circ.cx(fswap_qr[0],fswap_qr[1])

    # cz
    fswap_circ.cz(fswap_qr[0],fswap_qr[1])
    #fswap_circ.h(fswap_qr[1])
    #fswap_circ.cx(fswap_qr[0],fswap_qr[1])
    #fswap_circ.h(fswap_qr[1])

    # convert to gate
    return fswap_circ.to_instruction()

def FFFT(Nsite):
    """
    returns the circuit for the Nsite FFFT taking states
    from the position space to the momentum space.

    Nsite must be a power of 2.

    Right now, it is only implemented for Nsite = 2 and 4
    """

    if int(Nsite) == 2:
        return F0_gate()
    elif int(Nsite) == 4:
        return FFFT4()
    else:
        raise ValueError('FFFT is only implemented for Nsite = 2 and 4. Nsite = {} does not work.'.format(Nsite))

def FFFTd(Nsite):
    """
    returns the circuit for the Nsite FFFT taking states
    from the position space to the momentum space.

    Nsite must be a power of 2.

    Right now, it is only implemented for Nsite = 2 and 4
    """

    if int(Nsite) == 2:
        return F0_gate()
    elif int(Nsite) == 4:
        return FFFT4d()
    else:
        raise ValueError('FFFT is only implemented for Nsite = 2 and 4. Nsite = {} does not work.'.format(Nsite))

def FFFT4():
    """
    returns the circuit for the 4-sites FFFT taking states
    from the position space to the momentum space
    """
    FFFT_qr = QuantumRegister(4)
    FFFT_circ = QuantumCircuit(FFFT_qr, name='FFFT4')
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[0],FFFT_qr[1]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[2],FFFT_qr[3]])
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[0],FFFT_qr[1]])
    FFFT_circ.rz(np.pi/2, FFFT_qr[3])
    FFFT_circ.append(F0_gate(), [FFFT_qr[2],FFFT_qr[3]])
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])

    return FFFT_circ.to_instruction()

def FFFT4d():
    """
    returns the circuit for the 4-sites FFFT.dag() taking states
    from the moentum space to the position space
    """
    FFFT_qr = QuantumRegister(4)
    FFFT_circ = QuantumCircuit(FFFT_qr, name='FFFT4d')
    
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[2],FFFT_qr[3]])
    FFFT_circ.rz(-np.pi/2, FFFT_qr[3])
    FFFT_circ.append(F0_gate(), [FFFT_qr[0],FFFT_qr[1]])
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[2],FFFT_qr[3]])
    FFFT_circ.append(F0_gate(), [FFFT_qr[0],FFFT_qr[1]])
    FFFT_circ.append(fswap(), [FFFT_qr[1],FFFT_qr[2]])

    return FFFT_circ.to_instruction()

def expTk(theta, Nsite):
    """
    returns the circuit for exp[I \theta T_k]
    where theta encapsulates all Hamiltonian parameters
    """
    expTk_qr = QuantumRegister(2*Nsite)
    expTk_circ = QuantumCircuit(expTk_qr, name='e^iTk({})'.format(theta))

    for k in range(Nsite):
        expTk_circ.rz(theta[k]*np.cos(2.*np.pi*k/Nsite), expTk_qr[k])
        expTk_circ.rz(theta[k]*np.cos(2.*np.pi*k/Nsite), expTk_qr[k+Nsite])

    # convert to gate
    return expTk_circ.to_instruction()

def expV(theta, Nsite):
    """
    returns the circuit for exp[I \theta V]
    where theta encapsulates all Hamiltonian parameters
    """
    expV_qr = QuantumRegister(2*Nsite)
    expV_circ = QuantumCircuit(expV_qr, name='e^iV({})'.format(theta))

    for i in range(Nsite):
        expV_circ.append(ZZ_gate(-theta[i]), [expV_qr[i],expV_qr[i+Nsite]])

    # convert to gate
    return expV_circ.to_instruction()
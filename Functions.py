
# At start of each file use: 

import matplotlib.pyplot as plt
import numpy as np
from math import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity, StateVector, DensityMatrix
from qiskit import BasicAer
from qiskit.quantum_info.operators import Pauli, Operator

from qiskit.circuit import Gate, Parameter

import scipy.linalg as la

# The following are all modular functions created for the 1 qubit case, which can be applied to the qubit case and more

# Adding unitary matrix as gate 

def ugate(circ,mat,qubits,name='Gate'):
    if not np.allclose(np.eye(mat.shape[0]), mat.conj().T @ mat):
        raise ValueError("Input matrix is not unitary")
    gate = Operator(mat)
    circ.unitary(gate,qubits,label=name)
    return  circ

# Hamiltonian for z direction

def hamiltonian(omega,z):
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    hamiltonian = (omega/2) * z * np.kron(sigma_z,np.eye(2))
    return hamiltonian

# Energy difference 

def edif (ham,circ,state_i,state_f):
    den_mat_i = np.outer(state_i,state_i)
    E_i = np.real(np.trace(ham @ den_mat_i))
    den_mat_f = np.outer(state,state)
    E_f = np.real(np.trace(ham @ den_mat_f))
    return E_i , E_f
    
# Entropy difference

def sdif (state_i,state_f):
    den_mat_i = np.outer(state_i,state_i)
    den_mat_f = np.outer(state_f,state_f)
    s_i = -np.real(np.trace(np.dot(den_mat_i, la.logm(den_mat_i))))
    s_f = -np.real(np.trace(np.dot(den_mat_f, la.logm(den_mat_f))))
    s_dif = s_f - s_i
    return s_i , s_f, s_dif 

# Hamiltonian with multiple pauli components

def Hamiltonian_3(omega,x=0,y=0,z=0):
    coeff = {'X':x,'Y':y,'Z':z}
    ham = np.zeros((4,4), dtype=np.complex128)
    for pauli,coeff in coeff.items():
        paulis = Pauli(pauli).to_matrix()
        ham += coeff * (np.kron(paulis,np.eye(2)))
    ham *= omega/2
    return ham 

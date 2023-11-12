import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Numerical calculation of matrices 

'''
Step 2 - entangle state with unitary operator.
Input: initial unitary matrix and density matrix of the composite system  
Output: intermediate density matrix
'''
    
def unitary_ent (mat,rho):
    mat_dag = mat.conj().T
    rho2 = mat@rho@mat_dag
    return rho2
'''    
Step 3 - measure via projective measurement
Input: intermediate density matrix and memory state the system will be projected onto
Output: measurement density matrix 
'''

def meas (rho,k):
    P_k = np.kron(np.outer([[1,0]],[[1,0]]),[[1,0],[0,1]]) # for a prjective measurement in 0 state 
    rho_k = P_k@rho@P_k
    p_k = np.trace(rho_k)
    rho_m = rho_k/p_k
    return rho_m
'''
Step 4 - feeback on system 
Input: measurement density matrix and feedback unitary matrix 
Output: density matrix of final state 
'''

def feedback (rho,mat):
    mat_dag = mat.getH()
    rho_f= mat@rho@mat_dag
    return rho_f
    
'''
Calculating the energy difference caused by the operation 
Input: the Hamiltonian of the system, the initial density matrix and the final density matrix 
Output: energy change 
'''
    
def delt_e (H,rho1,rho2):
    return (np.trace(H*rho2)-np.trace(H*rho1))
    
'''
Calculating the entropy difference caused by the operation
Input: the intial and final density matrices
Output: the entropy difference
'''
    
def delt_s (rho1,rho4):
    return (rho4*np.log(rho4) - rho1*np.log(rho1))

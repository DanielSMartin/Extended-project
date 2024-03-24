
import sympy as sym
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm

import matplotlib.pyplot as plt

 

def Energy(rho,H):
    '''
    Returns the energy of a given density matrix for a given hamiltonian
    '''
    return np.trace(rho @ H)

# A measurement in the x direction, kappa indicates the strength
# kapppa = 1/2, no information gained,
# kappa = 0,1, maximum binary information gained
def M_plus(kappa):
    A = (np.sqrt(kappa) + np.sqrt(1-kappa)) * I
    B = (np.sqrt(kappa) - np.sqrt(1-kappa)) * sigma_x
    return 1/2*(A+B)

# Same measurement as above except in the minus direction
# sum(M_i ^2) = I
def M_minus(kappa):
    A = (np.sqrt(kappa) + np.sqrt(1-kappa)) * I
    B = (np.sqrt(kappa) - np.sqrt(1-kappa)) * sigma_x
    return 1/2*(A-B)

def BlochCoords(rho):
    '''
    Returns the Bloch sphere coordinates of the density matrix
    '''
    a = rho[0, 0]
    b = rho[1, 0]
    x = 2.0 * b.real
    y = 2.0 * b.imag
    z = 2.0 * a - 1.0
    return([x,y,z])

def Entropy(rho):
    return -np.trace(rho @ logm(rho))


I = np.array([[1,0],[0,1]]) # Identity matrix
#Pauli matrices:
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])


#%% two qubits 

# H = sum of individuak and correlations 
# ea = 0.1 and eb = 2
# most constants set to 1

def H_2 (e_a,e_b,gx,gy,gz):
    H_qs = np.kron(1/2 * e_a * sigma_z,I) + np.kron(I,1/2 * e_b * sigma_z)
    H_corr = (gx*np.kron(sigma_x,sigma_x)) + (gy*np.kron(sigma_y,sigma_y)) +(gz*np.kron(sigma_z,sigma_z))
    return H_qs + H_corr

def U_fb (theta):
    return expm(-1j * theta * sigma_y)

'''
For qubit 1, input strength and q=1, qubit 2 q=2 and both at same time q=3
'''

def Meas (kappa,q):
    A = 1/2 * (np.sqrt(kappa) + np.sqrt(1-kappa)) * np.kron(I,I)
    B = 1/2 * (np.sqrt(kappa) - np.sqrt(1-kappa)) * np.kron(sigma_x,I)
    C = 1/2 * (np.sqrt(kappa) - np.sqrt(1-kappa)) * np.kron(I,sigma_x)
    D = 1/2 * (np.sqrt(kappa) - np.sqrt(1-kappa)) * np.kron(sigma_x,sigma_x)
    
    if  q == 1 :
       return (A+B)
    elif q == 2:
       return (A+C)
    else:
        return (A+D)

    
def rho(beta,H):
    Z = np.trace(expm(-beta*H))
    return expm(-beta*H)/Z



# Test for Q_a = E(rhoT) - E(rhof)

y = 0 
ET_vals = np.zeros([1000,4])
Ef_vals = np.zeros([1000,4])
Q_vals = np.zeros([1000,4])


for k in [0.01,0.4]:
    for f in [-1,1]:
        beta = 0.75
        ea = 0.1/beta
        eb = 2/beta
        x=0
        gz_beta_vals = np.linspace(-0.5,0.5,1000)
        gz_range = (1/beta)*np.ones(len(gz_beta_vals))*gz_beta_vals
        for gz in gz_range:
            
            Ham = H_2(ea,eb,0,0,gz)
            rhoT = rho(beta,Ham)
            
            M = Meas(k,1)
            rhom = M @ rhoT @ np.conj(M).T / np.trace(M @ rhoT @ np.conj(M).T)
            rhoam = np.trace(rhom.reshape(2,2,2,2),axis1=1,axis2=3)
            X , Y , Z = BlochCoords(rhoam)
            
            if gz*beta < 0.05:
                if Z == 0:
                    if f == -1:
                        theta = np.pi/4 + np.pi/2
                    elif f == 1:
                        theta = -np.pi/4 - np.pi/2
            
                else: 
                    if f == -1:
                        theta = -1/2 * np.arctan(X/Z) 
                    elif f == 1:
                        theta = -1/2 * np.arctan(X/Z) + np.pi/2
            elif gz*beta > 0.05:
                if Z == 0:
                    if f == -1:
                        theta = np.pi/4 + np.pi/2
                    elif f == 1:
                        theta = -np.pi/4 - np.pi/2
                else:
                    if f == -1:
                        theta = -1/2 * np.arctan(X/Z) + np.pi/2
                    elif f == 1:
                        theta = -1/2 * np.arctan(X/Z) 
            
                
            U = np.kron(U_fb(theta),I)
            Udag = np.kron(np.conj(U_fb(theta)).T,I)
            rhof = U @ rhom @ Udag
            
            
            
            ET_vals[x,y] = Energy(rhoT,Ham)
            Ef_vals[x,y] = Energy(rhof,Ham)
            Q_vals[x,y] = Energy(rhoT,Ham) - Energy(rhof,Ham)
            x += 1
        y += 1


plt.plot(gz_beta_vals,Q_vals[:,0],linestyle='-.', color='purple',label='$k=0 , F_1 = -1$')
plt.plot(gz_beta_vals,Q_vals[:,1], color='black',label='$k=0 , F_1 = +1$')
plt.plot(gz_beta_vals,Q_vals[:,2],linestyle='--', color='blue',label='$k=0.4 , F_1 = -1$')
plt.plot(gz_beta_vals,Q_vals[:,3],linestyle='dotted', color='red',label='$k=0.4 , F_1 = +1$')

plt.axhline(y=0,linestyle='dashed',color='grey')
#plt.axvline(x=0.06,linestyle='dashed',color='grey')


plt.xlabel('$g_z [kT]$')
plt.ylabel('$Q_A$')
plt.legend()

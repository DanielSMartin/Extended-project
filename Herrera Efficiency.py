import sympy as sym
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm

import matplotlib.pyplot as plt 

def eff (W,Q):
    return -1* (W/Q)

eff_vals = []

for i in e_ratio:
    e_b = i * e_a
    
    H_a = Ham(e_a)
    H_b = Ham(e_b)
    H_ab = np.kron(H_a,np.eye(2)) + np.kron(H_b,np.eye(2)) 
    
    Z_a = partition(beta_a,H_a)
    Z_b = partition(beta_b,H_b)
    a_max = 1/(Z_a * Z_b)
    
    rho_a_i = rho_i(beta_a,H_a,Z_a)
    rho_b_i = rho_i(beta_b,H_b,Z_b)
    rho_ab_i = rho_tot_i(rho_a_i,rho_b_i,chi(a_max))
    
    U_mat = U(np.pi/2)
    rho_ab_f = U_mat @ rho_ab_i @ np.conj(U_mat).T
    
    rho_a_f = np.trace(rho_ab_f.reshape(2,2,2,2),axis1=1,axis2=3)
    
    w_exp = -(Q(rho_a_f,rho_a_i,H_a) + Q(rho_b_f,rho_b_i,H_b))
    q = Q(rho_a_f,rho_a_i,H_a)
    eff_vals.append(eff(w_exp,q))
    
    
plt.figure 
plt.plot(e_ratio,eff_vals,color='purple')
plt.xlabel('$\epsilon_a/\epsilon_B$')
plt.ylabel('$\eta$')

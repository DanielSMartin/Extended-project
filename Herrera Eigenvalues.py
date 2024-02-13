import sympy as sym
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm

from numpy import linalg as L 

eig_vals_ab = []
eig_vals_a = []
eig_vals_b = []

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
    rho_b_f = np.trace(rho_ab_f.reshape(2,2,2,2),axis1=0,axis2=2)
    
    eig_vals_ab.append(L.eigvals(rho_ab_f))
    eig_vals_a.append(L.eigvals(rho_a_f))
    eig_vals_b.append(L.eigvals(rho_b_f))

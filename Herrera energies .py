import sympy as sym
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm

import matplotlib.pyplot as plt


I = np.array([[1,0],[0,1]]) # Identity matrix
#Pauli matrices:
sigX = np.array([[0,1],[1,0]])
sigY = np.array([[0,-1j],[1j,0]])
sigZ = np.array([[1,0],[0,-1]])

ground = np.array([[1],[0]])
excited = np.array([[0],[1]])

#%%

# Defining functions to calclate the work done and energies

def rho_i (beta,H,Z):
    return expm(-beta*H)/Z

def rho_tot_i(rho_a,rho_b,chi):
    return (np.kron(rho_a,rho_b))+chi

def chi (a):
    state_01 = np.kron(ground,excited)
    state_10 = np.kron(excited,ground)
    chi = a*np.outer(state_01,state_10) + np.conj(a)*np.outer(state_10,state_01)
    
    trace_chi = np.trace(chi)
    
    if trace_chi != 0:
        return "Error : Invalid a - trace not equal to zero"
    
    return chi
    
def partition(beta,H):
    return np.trace(expm(-beta*H))

def Ham(e):
    return -1/2 * e * sigZ

def U(Jt):
    sig_sum = np.kron(sigX,sigX) + np.kron(sigY,sigY) + np.kron(sigZ,sigZ)
    return expm(-1j * Jt * 1/2 * sig_sum)

def W(rho_f,rho_i,H):
    return np.trace((rho_f - rho_i) @ H)

def Q(rho_f,rho_i,H):
    return -np.trace((rho_f - rho_i) @ H)

# Setting values 

beta_a = 0.1
beta_b = 0.2
e_a = 10
e_ratio = np.linspace(0,1.5,20)
e_a_vals = e_a * np.ones(len(e_ratio))


# Only shows curved characteristic when epsilon_a is 100 times larger than beta

W_vals= []
Q_a_vals = []
Q_b_vals = []
W_exp = []

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
    
    
    
    W_vals.append(W(rho_ab_f,rho_ab_i,H_ab))
    Q_a_vals.append(Q(rho_a_f,rho_a_i,H_a))
    Q_b_vals.append(Q(rho_b_f,rho_b_i,H_b))
    
    W_exp.append(-(Q(rho_a_f,rho_a_i,H_a) + Q(rho_b_f,rho_b_i,H_b)))

# Plotting results 

plt.figure 
#plt.plot(e_ratio,W_vals/e_a_vals,color='green')
plt.plot(e_ratio,W_exp/e_a_vals,color='green')
plt.plot(e_ratio,Q_a_vals/e_a_vals,color='red')
plt.plot(e_ratio,Q_b_vals/e_a_vals,color='blue')
plt.axhline(y=0,linestyle='--',color='grey')
plt.axvline(x=0.5,linestyle='--',color='grey')
plt.axvline(x=1,linestyle='--',color='grey')
plt.legend(['$W/\epsilon_A$','$Q_A/\epsilon_A$','$Q_B/\epsilon_A$'])
plt.xlabel('$\epsilon_a/\epsilon_b$')
plt.ylabel('Energy')

#%% Using theoretical models to plot energies 

Wc_vals= []
Qc_a_vals = []
Qc_b_vals = []
Wc_exp = []

def w_avg (ea,eb,f):
    return 2 * (eb - ea) * f

def qa_avg (ea,f):
    return 2* ea * f

def qb_avg (eb,f):
    return -2* eb * f

def f (ea,ba,eb,bb,za,zb,lam,a):
    deltv = 1/2 * ((eb*bb)-(ea*ba))
    return lam/(za*zb) * np.sinh(deltv) + 2*a*np.sqrt(lam - lam**2)

beta_a = 0.1
beta_b = 0.2
e_a = 10
e_ratio = np.linspace(0,1.5,20)
e_a_vals = e_a * np.ones(len(e_ratio))
lamb = 0.6

for i in e_ratio:
    e_b = i * e_a 
    
    H_a = Ham(e_a)
    H_b = Ham(e_b)
    H_ab = np.kron(H_a,np.eye(2)) + np.kron(H_b,np.eye(2)) 
    
    Z_a = partition(beta_a,H_a)
    Z_b = partition(beta_b,H_b)
    a_max = 1/(Z_a * Z_b)
    
    F = f(e_a,beta_a,e_b,beta_b,Z_a,Z_b,lamb,a_max)
    
    Wc_vals.append(w_avg(e_a,e_b,F))
    Qc_a_vals.append(0.3*qa_avg(e_a,F))
    Qc_b_vals.append(0.3*qb_avg(e_b,F))
    
    Wc_exp.append(-(0.3*qa_avg(e_a,F) + 0.3*qb_avg(e_b,F)))
    
    # replace 0.3 with 1.6 for uncorrelated graph
    
plt.figure 
#plt.plot(e_ratio,W_vals/e_a_vals,color='green')
plt.plot(e_ratio,Wc_exp/e_a_vals,color='green')
plt.plot(e_ratio,Qc_a_vals/e_a_vals,color='red')
plt.plot(e_ratio,Qc_b_vals/e_a_vals,color='blue')
plt.axhline(y=0,linestyle='--',color='grey')
plt.axvline(x=1,linestyle='--',color='grey')
plt.legend(['$W/\epsilon_A$','$Q_A/\epsilon_A$','$Q_B/\epsilon_A$'])
plt.xlabel('$\epsilon_A/\epsilon_B$')
plt.ylabel('Energy')


'''
Note - the uncorrelated graph can be scaled up by a factorof 1.6 to recreate 
that formed from the matrices. The correlated graph can be scaled by a factor
of 0.3 to recreate that of the paper. These equations used are merely theoretical 
to fit the pattern , exact scaling may depend on the actual matrices  
'''


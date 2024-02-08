#%% Entropy 

def Entropy(rho):
    return -np.trace(rho @ logm(rho))


# Carried out for rho_f after stroke 2


ent_vals = []

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
    
    ent_vals.append(Entropy(rho_ab_f))
    
plt.figure 
plt.plot(e_ratio,ent_vals/e_a_vals,color='green')
plt.xlabel('$\epsilon_a/\epsilon_B$')
plt.ylabel('$S/\epsilon_A$')

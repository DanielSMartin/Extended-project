#%% y Hamiltonian work extraction


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
        gy_beta_vals = np.linspace(-0.5,0.5,1000)
        gy_range = (1/beta)*np.ones(len(gy_beta_vals))*gy_beta_vals
        for gy in gy_range:
            
            Ham = H_2(ea,eb,gx,0,0)
            rhoT = rho(beta,Ham)
            
            M = Meas(k,1)
            rhom = M @ rhoT @ np.conj(M).T / np.trace(M @ rhoT @ np.conj(M).T)
            rhoam = np.trace(rhom.reshape(2,2,2,2),axis1=1,axis2=3)
            X , Y , Z = BlochCoords(rhoam)
            
            if f == -1:
                theta = -1/2 * np.arctan(X/Z) 
            elif f == 1:
                theta =-1/2 * np.arctan(X/Z) + np.pi/2
                
            U = np.kron(U_fb(theta),I)
            Udag = np.kron(np.conj(U_fb(theta)).T,I)
            rhof = U @ rhom @ Udag
            
            
            
            ET_vals[x,y] = Energy(rhoT,Ham)
            Ef_vals[x,y] = Energy(rhof,Ham)
            Q_vals[x,y] = Energy(rhoT,Ham) - Energy(rhof,Ham)
            x += 1
        y += 1


#%%

plt.plot(gy_beta_vals,Q_vals[:,0],linestyle='-.', color='purple',label='$k=0 , F_1 = -1$')
plt.plot(gy_beta_vals,Q_vals[:,1], color='black',label='$k=0 , F_1 = +1$')
plt.plot(gy_beta_vals,Q_vals[:,2],linestyle='--', color='blue',label='$k=0.4 , F_1 = -1$')
plt.plot(gy_beta_vals,Q_vals[:,3],linestyle='dotted', color='red',label='$k=0.4 , F_1 = +1$')

plt.axhline(y=0,linestyle='dashed',color='grey')


plt.xlabel('$g_y [kT]$')
plt.ylabel('$Q_A$')
plt.legend()

#%% Efficiency against kappa for set g vals - y ham

f = -1
beta = 0.75
y = 0

W_ext_vals = np.zeros([100,5])
W_eras_vals = np.zeros([100,5])
efficiency_y_vals = np.zeros([100,5])


for g in [-0.2/beta,-0.1/beta,0,0.1/beta,0.2/beta]:
    ea = 0.1/beta
    eb = 2/beta
    x=0
    Ham = H_2(ea,eb,0,g,0)
    rhoT = rho(beta,Ham)
        
    kappa_range = np.linspace(0.001,0.999,100)
    for kap in kappa_range:
        
        M = Meas(kap,1)
        rhom = M @ rhoT @ np.conj(M).T / np.trace(M @ rhoT @ np.conj(M).T)
        rhoam = np.trace(rhom.reshape(2,2,2,2),axis1=1,axis2=3)
        X , Y , Z = BlochCoords(rhoam)
        
        theta = -1/2 * np.arctan(X/Z)
        U = np.kron(U_fb(theta),I)
        Udag = np.kron(np.conj(U_fb(theta)).T,I)
        rhof = U @ rhom @ Udag
        rhoF = rhof/np.trace(rhof)
        
        W_ext = Energy(rhoT,Ham) - Energy(rhoF,Ham)
        W_eras = 1 * (Entropy(rhoF)-Entropy(rhoT))
        eff = (W_ext + W_eras)/Energy(rhom,Ham)
        
        W_ext_vals[x,y] = W_ext
        W_eras_vals[x,y] = W_eras
        efficiency_y_vals[x,y] = eff
        
        x += 1
    y += 1


#%% 

plt.figure()
plt.plot(kappa_range,efficiency_y_vals[:,0],label='$g_y=-0.2kT$')
plt.plot(kappa_range,efficiency_y_vals[:,1],label='$g_y = -0.1kT$')
plt.plot(kappa_range,efficiency_y_vals[:,2],label='$g_y = 0kT$')
plt.plot(kappa_range,efficiency_y_vals[:,3],label='$g_y = 0.1kT$')
plt.plot(kappa_range,efficiency_y_vals[:,4],label='$g_y = 0.2kT$')


plt.xlabel('$\kappa$')
plt.ylabel('$\eta$')

plt.legend()

#%% work extraction with no correlation - two feedback regimes 

f = -1
beta = 0.75
y = 0

W_ext_vals = np.zeros([100,2])
W_eras_vals = np.zeros([100,2])
efficiency_vals = np.zeros([100,2])

y = 0 
for F in [-1,1]:
        
    ea = 0.1/beta
    eb = 2/beta
    x=0
    Ham = H_2(ea,eb,0,0,0)
    rhoT = rho(beta,Ham)
        
    kappa_range = np.linspace(0.001,0.999,100)
    for kap in kappa_range:
        
        M = Meas(kap,1)
        rhom = M @ rhoT @ np.conj(M).T / np.trace(M @ rhoT @ np.conj(M).T)
        rhoam = np.trace(rhom.reshape(2,2,2,2),axis1=1,axis2=3)
        X , Y , Z = BlochCoords(rhoam)
        
        if F == -1:
            theta = -1/2 * np.arctan(X/Z)
        else:
            theta = -1/2 * np.arctan(X/Z) + np.pi/2
            
        U = np.kron(U_fb(theta),I)
        Udag = np.kron(np.conj(U_fb(theta)).T,I)
        rhof = U @ rhom @ Udag
        rhoF = rhof/np.trace(rhof)
        
        W_ext = Energy(rhoT,Ham) - Energy(rhoF,Ham)
        W_eras = 1 * (Entropy(rhoF)-Entropy(rhoT))
        eff = (W_ext + W_eras)/Energy(rhom,Ham)
        
        W_ext_vals[x,y] = W_ext
        W_eras_vals[x,y] = W_eras
        efficiency_vals[x,y] = eff
        
        x += 1
    y+=1


#%% 

plt.figure()
plt.plot(kappa_range,W_ext_vals[:,0],label='$F_1 = -1$')
plt.plot(kappa_range,W_ext_vals[:,1],label='$F_1 = +1$')

plt.xlabel('$\kappa$',fontsize=12)
plt.ylabel('$W_{ext}$',fontsize=12)
plt.legend()

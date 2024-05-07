'''
Keeping x constant at te value giving max efficiency, z is varied over the appropriate range. Can see at max gx that both z and x 
efficiency graphs merge, following the pattern of z with the jump around k=1/2 similar to x. By changing gx from 0 to the max at 0.2
it can be seen that the effiencies lift up and start forming the jump arounf k=1/2 i.e. a higher x component increases efficiency for all z
component values 
'''
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
        gx = 0.2/beta
        gz_beta_vals = np.linspace(-0.5,0.5,1000)
        gz_range = (1/beta)*np.ones(len(gz_beta_vals))*gz_beta_vals
        for gz in gz_range:
            
            Ham = H_2(ea,eb,gx,0,gz)
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

#%%

plt.plot(gx_beta_vals,Q_vals[:,0],linestyle='-.', color='purple',label='$k=0 , F_1 = -1$')
plt.plot(gx_beta_vals,Q_vals[:,1], color='black',label='$k=0 , F_1 = +1$')
plt.plot(gx_beta_vals,Q_vals[:,2],linestyle='--', color='blue',label='$k=0.4 , F_1 = -1$')
plt.plot(gx_beta_vals,Q_vals[:,3],linestyle='dotted', color='red',label='$k=0.4 , F_1 = +1$')

plt.axhline(y=0,linestyle='dashed',color='grey')


plt.xlabel('$g_z [kT]$')
plt.ylabel('$Q_A$')
plt.legend()

#%% Efficiency against kappa for set g vals - x and z ham

f = -1
beta = 0.75
y = 0

W_ext_vals = np.zeros([100,4])
W_eras_vals = np.zeros([100,4])
efficiency_xz_vals = np.zeros([100,4])


for gz in [-0.2/beta,-0.1/beta,-0.05/beta,0]:
    ea = 0.1/beta
    eb = 2/beta
    x=0
    gx = 0.2/beta
    Ham = H_2(ea,eb,gx,0,gz)
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
        efficiency_xz_vals[x,y] = eff
        
        x += 1
    y += 1


#%% 

plt.figure()
plt.plot(kappa_range,efficiency_xz_vals[:,0],label='$g_z=-0.2kT$')
plt.plot(kappa_range,efficiency_xz_vals[:,1],label='$g_z = -0.1kT$')
plt.plot(kappa_range,efficiency_xz_vals[:,2],label='$g_z = -0.05kT$')
plt.plot(kappa_range,efficiency_xz_vals[:,3],label='$g_z = 0kT$')
#plt.plot(kappa_range,efficiency_xz_vals[:,4],label='$g_x = 0.2kT$')


plt.xlabel('$\kappa$')
plt.ylabel('$\eta$')

plt.legend()

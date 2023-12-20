#%% 

def COP(omega,z0,Q,Td):
    num = omega*(z0 + np.sqrt(1 + (np.exp(-Q)*(z0**2 - 1))))
    den = (omega*z0*(np.exp(Q/2) - 1)) + (Td*np.log(4))
    return num/den

y = 0 # iteration for each z0
C = np.zeros([1000,3])

for j in [-0.01,-0.1,-0.5]:
    omega = 1 # constant for H 
    
    kappa_range = np.linspace(0,1,1000)
    x = 0 # iteration to store values 
    for i in kappa_range:
        kappa = i
        q = Q(kappa)
        t_d = 0.001
        
        COPeff =  COP(omega,j,q,t_d)
        C[x,y] = COPeff
        x += 1
    y+= 1
    
#%% Plotting efficiency varying initial state 
plt.plot(kappa_range,C[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$z_0$ = $-0.01K$')
plt.plot(kappa_range,C[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$z_0$ = -0.1 $K$')
plt.plot(kappa_range,C[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$z_0$ = -0.5 $K$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$C$',fontsize=18)
plt.xlabel('$\kappa$',fontsize=18)
plt.legend(fontsize=10)

'''
PLotting incorrect graph - maybe the definition of the equation for C??
'''

#%% Trying to plot COP using explicit energy terms in equation 

def E_0(omega,z0):
    return 1/2 * omega * (1 + z0)


def E_M(omega,z0,Q):
    return 1/2 * omega * (1 + z0*np.exp(-Q/2))


def E_f(omega,Q,z0):
    return 1/2 * omega * (1 - np.sqrt(1 + np.exp(-Q)*(z0**2 - 1)))

def W_er(Td):
    return Td*np.log(2)

def C(omega,z0,Q,Td):
    num = E_0(omega,z0)-E_f(omega,Q,z0)
    den = E_M(omega,z0,Q)-E_0(omega,z0)+W_er(Td)
    return num/den

y = 0 # iteration for each z0
C_eff = np.zeros([1000,3])

for j in [-0.01,-0.1,-0.5]:
    omega = 0.1 # constant for H  - changed to 0.1 to reproduce paper's graph
    
    kappa_range = np.linspace(0,1,1000)
    x = 0 # iteration to store values 
    for i in kappa_range:
        kappa = i
        q = Q(kappa)
        t_d = 0.001
        
        
        Ceff =  C(omega,j,q,t_d)
        C_eff[x,y] = Ceff
        x += 1
    y+= 1
    
#%% 
plt.plot(kappa_range,C_eff[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$z_0$ = $-0.01K$')
plt.plot(kappa_range,C_eff[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$z_0$ = -0.1 $K$')
plt.plot(kappa_range,C_eff[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$z_0$ = -0.5 $K$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$C$',fontsize=18)
plt.xlabel('$\kappa$',fontsize=18)
plt.legend(fontsize=10)

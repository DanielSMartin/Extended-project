'''
Plotting figure 1 in Yanik paper 
'''
import numpy as np
import matplotlib.pyplot as plt


# Defining equations for q and components of efficiency eta 

def Q(kappa):
    return -2*np.log(2) - np.log(kappa*(1-kappa))

def nbar(omega,beta):
    return 1/(np.exp(omega*beta)-1)

def z0(nbar):
    return -1/(2*nbar + 1)

def eta(z0,Q,omega,Td):
    num  = 1 - (np.sqrt(1 + (np.exp(-Q))*(z0**2 -1))) + ((2/omega) * Td*np.log(2))
    den = 1 + z0*np.exp(-Q/2)
    return 1 - (num/den)
    
y = 0 # iteration for each Td
efficiency = np.zeros([1000,3])

for j in [10**(-3),0.05,0.1]: # values of Td
    omega = 1 # constant for H 
    beta = 0.1 # setting constant for now
    
    kappa_range = np.linspace(0.01,0.99,1000)
    x = 0 # iteration to store values 
    for i in kappa_range:
        kappa = i
        n = nbar(omega,beta)
        z = z0(n)
        q = Q(kappa)
        
        eff = eta(z,q,omega,j)
        efficiency[x,y] = eff
        x += 1
    y+= 1


#%% Plotting efficiency varying demon temp
plt.plot(kappa_range,efficiency[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$T_D$ = $10^{-4}T$')
plt.plot(kappa_range,efficiency[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$T_D$ = 0.005 $T$')
plt.plot(kappa_range,efficiency[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$T_D$ = 0.01 $T$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$\eta$',fontsize=18)
plt.xlabel('$\kappa$',fontsize=18)
plt.legend(fontsize=10)    
    
#%% Now setting Td constant and changing intial state z0

y = 0 # iteration for each z0
efficiency = np.zeros([1000,3])

for j in [-0.01,-0.1,-0.5]:
    omega = 1 # constant for H 
    
    kappa_range = np.linspace(0,1,1000)
    x = 0 # iteration to store values 
    for i in kappa_range:
        kappa = i
        q = Q(kappa)
        t_d = 0.01
        
        eff = eta(j,q,omega,t_d)
        efficiency[x,y] = eff
        x += 1
    y+= 1
    
#%% Plotting efficiency varying initial state 
plt.plot(kappa_range,efficiency[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$z_0$ = $-0.01K$')
plt.plot(kappa_range,efficiency[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$z_0$ = -0.1 $K$')
plt.plot(kappa_range,efficiency[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$z_0$ = -0.5 $K$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$\eta$',fontsize=18)
plt.xlabel('$\kappa$',fontsize=18)
plt.legend(fontsize=10) 

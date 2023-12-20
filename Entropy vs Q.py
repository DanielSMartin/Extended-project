#%% Plotting entropy against Q

def DS(Q,gamma,length):
    return 1/2 * ((Q + gamma) - length*(np.log((1+length)/(1-length))))

def gamma(z0):
    return z0 * np.log((1+z0)/(1-z0))

def length(kappa,z0):
    return np.sqrt(1 + 4*kappa*(1-kappa)*(z0**2 - 1))

# want three different values of z0 so set TD constant 

y = 0 
entropy = np.zeros([1000,3])
for j in [-0.01,-0.1,-0.5]:
    omega = 1 
    beta = 0.1 
    Td = 0.001
    
    kappa_range = np.linspace(0,1,1000)
    x = 0 # iteration to store values 
    for i in kappa_range:
        kappa = i
        q = Q(kappa)
        g = gamma(j)
        l = length(i,j)
        
        ent =  DS(q,g,l)
        entropy[x,y] = ent
        x += 1
    y+= 1
    
Q_vals = []
for k in kappa_range:
    Q_vals.append(Q(k))
    

    
#%% Plotting

plt.plot(Q_vals,entropy[:,0], linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$z_0$ = -0.01')
plt.plot(Q_vals,entropy[:,1], linestyle='-.', color='tomato', linewidth=2, dashes=(10,2,3,2), label = r'$z_0$ = -0.1 ')
plt.plot(Q_vals,entropy[:,2], linestyle=':', color='k', linewidth=2, dashes=(2, 1), label = r'$z_0$ = -0.5 ')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$\Delta S$',fontsize=18)
plt.xlabel('$Q$',fontsize=18)
plt.legend(fontsize=10) 


'''
Slight difference to the graph in the Yanik paper, but can be reproduced for a middle z0 value of arounf -0.35. The overall pattern of entropy holds
but not the exact same graph visual 
'''

#%% Plotting second entropy plot to show the cancellation above certain G value

# z0 = -0.05 so constant 
ent_z = []

for i in kappa_range:
    kappa = i
    q = Q(kappa)
    g = gamma(j)
    l = length(i,-0.05)
    
    ent =  DS(q,g,l)
    ent_z.append(ent)
    
min_ent_z = [-value for value in ent_z]


plt.plot(Q_vals,min_ent_z, linestyle='--', color='steelblue', linewidth=2, dashes=(10, 2), label = r'$-\Delta S_M$')
plt.plot(Q_vals,np.log(2)*np.ones([1000,1]), color='k', linewidth=2, dashes=(10,2,3,2), label = r'$\Delta S_{er}$')
plt.plot(Q_vals,ent_z + np.log(2), linestyle='--', color='tomato', linewidth=2, dashes=(2, 1), label = r'$\Delta S$')


plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('$\Delta S$',fontsize=18)
plt.xlabel('$Q$',fontsize=18)
plt.legend(fontsize=10)

'''
Again there is a slight discrepency with the paper. The overall pattern can be seen and the entropys are still shown to cancel, however the value of Q at which they 
do has decreased down to 1. This may still be proportionate to the decreased scale I have plotted with. The cause of these differences may stem from division by zero
but I am currently unsure if there is an issue in the actual code, since the same conclusions can be drawn but for a shifted range a value might differ from when the 
paper calculated these values.
'''

# Import relevant packages etc

import matplotlib.pyplot as plt
import numpy as np
from math import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity 
from qiskit import BasicAer

# Qiskit implementation of matrices as gates into the quantum circuit of 1 qubit

from qiskit import Aer
from qiskit.quantum_info.operators import Operator
from qiskit.circuit import Gate

'''
Defining gate to make any unitary matrix into a gate and add to the circuit 
Input - circuit, unitary matrix and gate name
Output - circuit and gate
'''

def ugate(circ,mat,qubits,name='Gate'):
    if not np.allclose(np.eye(mat.shape[0]), mat.conj().T @ mat):
        raise ValueError("Input matrix is not unitary")
    gate = Operator(mat)
    circ.unitary(gate,qubits,label=name)
    return  circ 

    
circ = QuantumCircuit(2)
U = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
qc = ugate(circ,U,[0,1],name='Gate')
tran_circ = transpile(qc,optimization_level =1)

backend = Aer.get_backend('statevector_simulator')
job = backend.run(tran_circ)
result = job.result()
state = result.get_statevector(tran_circ, decimals=3)
print(state)

meascirc = QuantumCircuit(2,2)
meascirc.measure(range(2),range(2)) 
tran_circ.add_register(meascirc.cregs[0])
Qc = tran_circ.compose(meascirc)

# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = backend_sim.run(transpile(Qc, backend_sim), shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()

counts = result_sim.get_counts(Qc)
print(counts)

Qc.draw()

# Calculation of the energy and entropy of the system 

'''
Defining definition for the hamiltonian based on pauli marrices
Input: omega and cofficients of pauli matirces
Output: hamiltonian
'''

def hamiltonian(omega,z):
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    hamiltonian = (omega/2) * z * np.kron(sigma_z,np.eye(2))
    return hamiltonian

from qiskit.quantum_info import DensityMatrix 

'''
Defining function to calculate energy difference based on hamiltonian and denisty matrices
Input: omega, coefficient of pauli z, the circuit and the final state vector
Output: energy values
'''

def edif (ham,circ,state_i,state_f):
    den_mat_i = np.outer(state_i,state_i)
    E_i = np.real(np.trace(ham @ den_mat_i))
    den_mat_f = np.outer(state,state)
    E_f = np.real(np.trace(ham @ den_mat_f))
    E_dif = E_f-E_i
    return E_i , E_f, E_dif

'''
Defining function to calculate the diffference in entropy of the system at different points
Input:
Output: entropy values
'''
import scipy.linalg as la

def sdif (state_i,state_f):
    den_mat_i = np.outer(state_i,state_i)
    den_mat_f = np.outer(state_f,state_f)
    s_i = -np.real(np.trace(np.dot(den_mat_i, la.logm(den_mat_i))))
    s_f = -np.real(np.trace(np.dot(den_mat_f, la.logm(den_mat_f))))
    s_dif = s_f - s_i
    return s_i , s_f, s_dif 

# Using functions to calculate the energy and entropy differences

w_vals = np.linspace(0.1,2.0,10)
results = []

for omega in w_vals:
    z =1 
    E_i , E_f , E_dif = edif(hamiltonian(omega,z),tran_circ,[1,0,0,0],state)
    s_i , s_f , s_dif = sdif([1,0,0,0],state)
    results.append((omega, E_i, E_f, E_dif, s_i, s_f, s_dif))
    
print("Results:")
print("w     Initial energy    Final energy  E change    Initial entropy    Final Entropy  s change")
for result in results:
    print("%.2f      %.4f          %.4f       %.4f.        %.4f          %.4f       %.4f" % (result[0],result[1],result[2],result[3],result[4],result[5],result[6]))

# Adding Hadamard gate into circuit, replacing the CNOT gate 

circ_H = QuantumCircuit(2)
U_H = 1 /np.sqrt(2) * np.array([[1,1],[1,-1]]) # hadamard
qc_H = ugate(circ_H,U_H,[0],name='Gate')
tran_circ_H = transpile(qc_H,optimization_level =1)

backend_H = Aer.get_backend('statevector_simulator')
job_H = backend_H.run(tran_circ_H)
result_H = job_H.result()
state_H = result_H.get_statevector(tran_circ_H, decimals=3)
print(state_H)

meascirc_H = QuantumCircuit(2,2)
meascirc_H.measure(range(2),range(2)) 
tran_circ_H.add_register(meascirc_H.cregs[0])
Qc_H = tran_circ_H.compose(meascirc_H)

# Use Aer's qasm_simulator
backend_sim_H = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim_H = backend_sim_H.run(transpile(Qc_H, backend_sim_H), shots=1024)

# Grab the results from the job.
result_sim_H = job_sim_H.result()

counts_H = result_sim_H.get_counts(Qc_H)
print(counts_H)

Qc_H.draw()

# Calculating differences for new circuit 

results_H =[]

for omega in w_vals:
    z =1 
    E_i , E_f , E_dif = edif(hamiltonian(omega,z),tran_circ_H,[1,0,0,0],state_H)
    s_i , s_f , s_dif = sdif([1,0,0,0],state_H)
    results_H.append((omega, E_i, E_f, E_dif, s_i, s_f, s_dif))
    
print("Results:")
print("w     Initial energy    Final energy  E change    Initial entropy    Final Entropy  s change")
for result in results_H:
    print("%.2f      %.4f          %.4f       %.4f.        %.4f           %.4f       %.4f" % (result[0],result[1],result[2],result[3],result[4],result[5],result[6]))
    

# Extract data from the results to plot 

wH = [row[0] for row in results_H]
eH = [row[3] for row in results_H]
sH = [row[5] for row in results_H]

# Plot the differences against a changing omega
plt.figure
plt.plot(wH,eH,marker='o',linestyle='-')
plt.xlabel('$\omega$')
plt.ylabel('$\Delta E$')

# Check
print(eH)

plt.figure
plt.plot(wH,sH,marker='o',linestyle='-',color='r')    
plt.xlabel('$\omega$')
plt.ylabel('$\Delta S$')

# Check 
print(sH)

'''
Find that there is no energy difference bewteen the first and second state vector for any value of omega. The change in entropy is constant for all omega
'''


'''
Want to investigate the effect of changing the Hamiltonian on the values
of energy and entropy 
'''

'''
Define hamiltonian which may have contributions in each direction from each Pauli matrtix
Input: frequency. input components are set default to 0. using function requires explicit values
Output: hamiltonian
'''

def Hamiltonian_3(omega,x=0,y=0,z=0):
    coeff = {'X':x,'Y':y,'Z':z}
    ham = np.zeros((4,4), dtype=np.complex128)
    for pauli,coeff in coeff.items():
        paulis = Pauli(pauli).to_matrix()
        ham += coeff * (np.kron(paulis,np.eye(2)))
    ham *= omega/2
    return ham 

# eg. for previous example where z = 1 : Hamiltonian(omega,z=1)

# testing for single directions of hamiltonian 

results_Hx = []
results_Hy = []
results_Hz = []

for omega in w_vals:
    E_ix , E_fx , E_difx = edif(Hamiltonian_3(omega,x=1),tran_circ_H,[1,0,0,0],state_H)
    s_ix , s_fx , s_difx = sdif([1,0,0,0],state_H)
    results_Hx.append((omega, E_ix, E_fx, E_difx, s_ix, s_fx, s_difx))
    
    E_iy , E_fy , E_dify= edif(Hamiltonian_3(omega,y=1),tran_circ_H,[1,0,0,0],state_H)
    s_iy , s_fy , s_dify= sdif([1,0,0,0],state_H)
    results_Hy.append((omega, E_iy, E_fy, E_dify, s_iy, s_fy, s_dify))
    
    E_iz , E_fz , E_difz= edif(Hamiltonian_3(omega,z=1),tran_circ_H,[1,0,0,0],state_H)
    s_iz , s_fz , s_difz= sdif([1,0,0,0],state_H)
    results_Hz.append((omega, E_iz, E_fz, E_difz, s_iz, s_fz, s_difz))

wHx = [row[0] for row in results_Hx]
eHx = [row[3] for row in results_Hx]
    
wHy = [row[0] for row in results_Hy]
eHy = [row[3] for row in results_Hy]

wHz = [row[0] for row in results_Hz]
eHz = [row[3] for row in results_Hz]

plt.figure 
plt.plot(wHx,eHx,marker='o',linestyle='-')
plt.plot(wHy,eHy,marker='o',linestyle='-')
plt.plot(wHz,eHz,marker='o',linestyle='-')
plt.xlabel('$\omega$')
plt.ylabel('$\Delta E$')

'''
Find that there is no energy change in any direction. Similarly, as S doesn' depend on H then this 
will also be unaffected by the direction
'''

# testing for a combination of directions z and x

results_Hzx = []
    
for omega in w_vals:  
    E_izx , E_fzx , E_difzx = edif(Hamiltonian_3(omega,x=1,z=1),tran_circ_H,[1,0,0,0],state_H)
    s_izx, s_fzx , s_difzx = sdif([1,0,0,0],state_H)
    results_Hzx.append((omega, E_izx, E_fzx, E_difzx, s_izx, s_fzx, s_difzx))

wHzx = [row[0] for row in results_Hzx]
eHzx = [row[3] for row in results_Hzx]

plt.figure
plt.plot(wHx,eHx,marker='o',linestyle='-')
plt.plot(wHzx,eHzx,marker='o',linestyle='-')
plt.plot(wHz,eHz,marker='o',linestyle='-')
plt.xlabel('$\omega$')
plt.ylabel('$\Delta E$')

'''
Appears that the combination of directions haas no affect on the energy difference. The initial 
matrix only has component of 1 in first entry and so the remainder of the hamiltonian matrix is 
obsolete.
'''

'''
Adding feedback to the circuit based on the measurement of system qubit i.e. if the system
qubit is measured as 1 then the gate is applied (rotation in this case), otherwise if measured
as a 0 then the gate isnt applied 
'''

# so far we have Qc_H with a hadamard gate acting on q_o and then measuring each qubit

'''
Want to define a rotation which can be in either the x,y or z direction
Input: rotation angle theta (pi/2) and the direction of the rotation
Output: a rotation in the given direction
'''

from qiskit.circuit import Parameter

theta = Parameter('angle')

def rotation(theta,direction):
    rot = QuantumCircuit(1)
    if direction == 'x':
        rot.rx(theta,0)
    elif direction == 'y':
        rot.ry(theta,0)
    elif direction == 'z':
        rot.rz(theta,0)
    else:
        raise ValueError('Invalid direction')
        
    return rot 

# Want to conditionally apply the rotation if the system qubit is 1 

if '1' in counts_H and counts_H[0] > 1:
    Qc_H_fb = Qc_H.compose(rotation(theta,'x'),0,inplace=True)
    
Qc_H_fb.draw()

'''
Now want to evaluate the new energy and entropy values. As we know there was no change in energy
in directions other than z, we will just use the z Hamiltonian
'''

# Run simlulation of statevector with rotation added

for d in ['x','y','z']:
    if '1' in counts_H and counts_H[0] > 1:
        Qc_H_fb = Qc_H.compose(rotation(np.pi/2,d),0,inplace=True)
        
    tran_circ_H_fb = transpile(Qc_H_fb,optimization_level =1)

    job_H_fb = backend_H.run(tran_circ_H_fb)
    result_H_fb = job_H_fb.result()
    state_H_fb = result_H_fb.get_statevector(tran_circ_H_fb, decimals=3)
    print(state_H_fb)

'''
NOTE: rotation may put q_0 into a superposition and so the statevectors can change when 
rerunning cell They can each be either one of two statevectors. Full statevectors 
noted in goodnotes 
'''



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

def edif (omega,z,circ,state):
    ham = hamiltonian(omega,z)
    E_i = np.real(np.trace(ham @ DensityMatrix.from_label('00').data))
    den_mat_f = np.outer(state,state)
    E_f = np.real(np.trace(ham @ den_mat_f))
    return E_i , E_f

'''
Defining function to calculate the diffference in entropy of the system at different points
Input:
Output: entropy values
'''
import scipy.linalg as la

def sdif (state):
    den_mat_i = DensityMatrix.from_label('00').data
    den_mat_f = np.outer(state,state)
    s_i = -np.real(np.trace(np.dot(den_mat_i, la.logm(den_mat_i))))
    s_f = -np.real(np.trace(np.dot(den_mat_f, la.logm(den_mat_f))))
    return s_i , s_f

w_vals = np.linspace(0.1,2.0,10)
results = []


for omega in w_vals:
    z =1 
    E_i , E_f = edif(omega,z,tran_circ,state)
    s_i , s_f = sdif(state)
    results.append((omega, E_i, E_f, s_i, s_f))
    
print("Results:")
print("    \omega   Initial energy     Final energy     Initial entropy    Final Entropy")
for result in results:
    print("     %.2f        %.4f            %.4f            %.4f.          %.4f" % (result[0],result[1],result[2],result[3],result[4]))

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

for omega in w_vals:
    results_H = []
    z =1 
    E_i , E_f = edif(omega,z,tran_circ_H,state_H)
    s_i , s_f = sdif(state_H)
    results_H.append((omega, E_i, E_f, s_i, s_f))
    
print("Results:")
print("    \omega   Initial energy     Final energy     Initial entropy    Final Entropy")
for result in results_H:
    print("     %.2f        %.4f            %.4f            %.4f.          %.4f" % (result[0],result[1],result[2],result[3],result[4]))

# Extract data from the results to plot 
wH = [row[0] for row in results_H]
eH = [row[2]-row[1] for row in results_H]
sH = [row[4]-row[3] for row in results_H]
    
# Plotting the graphs of E(w) against w and S(w) against w
plt.figure(figsize=(10,6))
plt.figure
plt.plot(wH,eH,marker='o',linestyle='-')
plt.plot(wH,sH,marker='o',linestyle='-',color='r') 

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
    E_ix , E_fx = edif(Hamiltonian_3(omega,x=1),tran_circ_H,state_H)
    s_ix , s_fx = sdif(state_H)
    results_Hx.append((omega, E_ix, E_fx, s_ix, s_fx))
    
    E_iy , E_fy = edif(Hamiltonian_3(omega,y=1),tran_circ_H,state_H)
    s_iy , s_fy = sdif(state_H)
    results_Hy.append((omega, E_iy, E_fy, s_iy, s_fy))
    
    E_iz , E_fz = edif(Hamiltonian_3(omega,z=1),tran_circ_H,state_H)
    s_iz , s_fz = sdif(state_H)
    results_Hz.append((omega, E_iz, E_fz, s_iz, s_fz))

wHx = [row[0] for row in results_Hx]
eHx = [row[2]-row[1] for row in results_Hx]
    
wHy = [row[0] for row in results_Hy]
eHy = [row[2]-row[1] for row in results_Hy]

wHz = [row[0] for row in results_Hz]
eHz = [row[2]-row[1] for row in results_Hz]

plt.figure 
plt.plot(wHx,eHx,marker='o',linestyle='-')
plt.plot(wHy,eHy,marker='o',linestyle='-')
plt.plot(wHz,eHz,marker='o',linestyle='-')

'''
The measurement is along z basis and so only the hamiltonian with component of Pauli Z has any
energy change, the others remaining 0
'''

# testing for a combination of directions z and x

results_Hzx = []
    
for omega in w_vals:  
    E_izx , E_fzx = edif(Hamiltonian_3(omega,x=1,z=1),tran_circ_H,state_H)
    s_izx, s_fzx = sdif(state_H)
    results_Hzx.append((omega, E_izx, E_fzx, s_izx, s_fzx))

wHzx = [row[0] for row in results_Hzx]
eHzx = [row[2]-row[1] for row in results_Hzx]

plt.figure
plt.plot(wHx,eHx,marker='o',linestyle='-')
plt.plot(wHzx,eHzx,marker='o',linestyle='-')
plt.plot(wHz,eHz,marker='o',linestyle='-')

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
    

# Resimulate the new circuit 
job_sim_H_fb = backend_sim_H.run(transpile(Qc_H_fb, backend_sim_H), shots=1024)

# Grab the results from the job.
result_sim_H_fb = job_sim_H_fb.result()

# New counts 
counts_H_fb = result_sim_H_fb.get_counts(Qc_H)
print(counts_H_fb)
Qc_H_fb.draw()



from logical_circuit_bounds.memory_device.tools.kmc import gill_alg
import time
import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap('viridis')

# Some parameters
tfinal = 1e7 # Final Simulation Time
Vd = 4.1     # Cross-Voltage, Vd
Cg = 10.     # Change in readout, constant, dVout/dt = -J(t)/Cg
gamma = 1
beta = 1
max_gate_val = Cg*2
min_gate_val = -Cg*(Vd+2)
params = [tfinal, Vd, Cg, gamma, beta, max_gate_val, min_gate_val]
print('Initial Parameters')
print(f'Vd = {Vd}')
print(f'Cg = {Cg}')
print(f'gamma = {gamma}')
print(f'beta = {beta}')

# Initial State
Y = np.zeros(6) # N1, P1, G1, N2, P2, G2
Y[5]=0.
Y[2]=int(-Vd*Cg/2.)

print('Starting kMC simulations...')

# Run a short initial simulation
params0 = params
params0[0] = params[0]/10
start = time.time()
print(params[0])
data = gill_alg(params,Y)
Y = data[-1, 1:7]
end = time.time()
plt.pause(0.01)
plt.show()
print('The initial simulation takes {0:.4f} s'.format(end-start))

# Run the simulation
start = time.time()
params[0] *= 10
print(params[0])
data = gill_alg(params,Y)
end = time.time()
print('The simulation takes {0:.4f} s'.format(end-start))

# Plot the results
f = plt.figure(figsize=(3, 3))
ax = f.add_subplot(211)

alpha = 0.4
ax.fill_between([data[0, 0], data[-1, 0]], [-100, -100], y2=[alpha, alpha], color = 'k', alpha=0.2)
ax.fill_between([data[0, 0], data[-1, 0]], [1-alpha, 1-alpha], y2=[100, 100], color = 'k', alpha=0.2)
ax.fill_between([data[0, 0], data[-1, 0]], [1-alpha, 1-alpha], y2=[alpha, alpha], color = 'r', alpha=0.2)
ax.plot(data[:, 0], -data[:, 6]/(Cg*Vd), '-', color=cmap(0.3))
ax.plot([data[0, 0], data[-1, 0]], [alpha, alpha], 'k:')
ax.plot([data[0, 0], data[-1, 0]], [1-alpha, 1-alpha], 'k:')

ax.set_xlim(data[0, 0], data[-1, 0])
ax.set_ylim(-0.25, 1.25)

plt.pause(0.01)
plt.savefig('./data/example_evolution.png', dpi=400)

plt.show()

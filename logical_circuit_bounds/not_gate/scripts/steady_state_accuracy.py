"""
This script runs a long time evolution to get the steady state
distribution (by taking large time steps) that are used to
determine the accuracy of a not gate output, as used in
Figure 1 (c) of the paper:
Stochastic thermodynamic bounds on logical circuit operation
"""
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from logical_circuit_bounds.not_gate.tools.ed import *
from logical_circuit_bounds.not_gate.tools.op_tools import *
from matplotlib.colors import ListedColormap
cmap = plt.get_cmap('viridis')

# System Parameters
Cg = [10]
Vd = [np.arange(1, 31)]
alpha = np.linspace(0., 1., 101)
print_inds = [np.argmin(np.abs(alpha-0.05)), np.argmin(np.abs(alpha-0.4))]
dt = 1e15
nsteps = 1
Vin = Vd

# Loop over all parameters to get the steady state accuracy
accuracy = np.zeros((len(Cg), len(Vd[0]), len(alpha)))
for Cgi in range(len(Cg)):
    for Vdi in range(len(Vd[Cgi])):

        # Figure out system parameters
        gate_vals = np.arange(-Vd[Cgi][Vdi]*Cg[Cgi]-Cg[Cgi]*2, Cg[Cgi]*2)
        Ngate = len(gate_vals)

        # Create an initial state
        init_state = np.zeros((2, 2, Ngate))
        ind = np.where(gate_vals == -Cg[Cgi]*Vd[Cgi][Vdi])[0]
        init_state[0, 0, ind] = 1.
        init_state = init_state.reshape(-1)

        # Do the time evolution
        params = {'beta': 1.,
                  'gamma': 1.,
                  'Cg': Cg[Cgi],
                  'Vd': Vd[Cgi][Vdi],
                  'Vin': 0.}

        state = init_state.copy()
        for step in range(nsteps):
            state = time_evolution_step(params, state, gate_vals, dt)

        # Measure occupation
        vl = np.ones(state.shape)
        Vout_densities = measure_gate_occ(state, vl, gate_vals)

        # Check accuracy
        Vout = - gate_vals / (Cg[Cgi]*Vd[Cgi][Vdi])
        for alphai in range(len(alpha)):
            acc_inds = np.where(Vout >= 1.-alpha[alphai])[0]
            accuracy[Cgi, Vdi, alphai] = np.sum(Vout_densities[acc_inds])

        print(f'{Cgi+1}/{len(Cg)}, {Vdi+1}/{len(Vd[Cgi])}, {Cg[Cgi]}, {Vd[Cgi][Vdi]}, {1.-accuracy[Cgi, Vdi, print_inds[0]]}, {1.-accuracy[Cgi, Vdi, print_inds[1]]}')
        np.savez('./data/steady_state_discharging.npz',
                  accuracy = accuracy,
                  Vd = Vd,
                  Cg = Cg,
                  alpha = alpha)

f, ax1 = plt.subplots(1, 1, figsize=(3, 3))
ls = ['--', '-']
for Cgi in range(len(Cg)):
    for Vdi in range(len(Vd[Cgi])):
        plt.plot(alpha, accuracy[Cgi, Vdi, :], ls[Cgi], color=cmap(Vdi/len(Vd[Cgi])))
ax1.set_xlabel(r'$\alpha$', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
plt.tight_layout()
plt.pause(0.01)

f, ax1 = plt.subplots(1, 1, figsize=(3, 3))
ls = ['--', '-']
for Cgi in range(len(Cg)):
    for Vdi in range(len(Vd[Cgi])):
        plt.semilogy(alpha, 1.-accuracy[Cgi, Vdi, :], ls[Cgi], color=cmap(Vdi/len(Vd[Cgi])))
ax1.set_xlabel(r'$\alpha$', fontsize=14)
ax1.set_ylabel('Inaccuracy', fontsize=14)
ax1.set_ylim(1e-15, 1e0)
plt.tight_layout()
plt.pause(0.01)
plt.savefig('./data/output_accuracy_discharging.pdf')

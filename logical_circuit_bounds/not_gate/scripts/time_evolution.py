"""
This script runs a trajectory charging and discharging a single
logical NOT gate, creating Figure 1 (b) from the paper:
Stochastic thermodynamic bounds on logical circuit operation
"""
# Useful Stuff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from logical_circuit_bounds.not_gate.tools.ed import *
from logical_circuit_bounds.not_gate.tools.op_tools import *
cmap = plt.get_cmap('viridis')
cmaps = []
cmap_vals = [0.3, 0.7]
for i in range(len(cmap_vals)):
    cmapi = cmap(cmap_vals[i])
    cmapi = np.array([cmapi for _ in range(256)])
    cmapi[:, -1] = np.linspace(0, 1, 256)
    cmaps.append(ListedColormap(cmapi))

# Plot Settings ###################################################
# Figure Size
figure_width = 3.375 # Inches
figure_height = figure_width*0.6 #Inches
plt.rc('text', usetex=True)
mpl.rc('font',**{'family':'serif','serif':['Arial']})
plt.rc('font', family='serif')
font_size = 11
mpl.rc('xtick', labelsize=font_size)
mpl.rc('ytick', labelsize=font_size)
mpl.rc('axes', titlesize=font_size)
mpl.rc('axes', labelsize=font_size)
mpl.rc('legend', fontsize = font_size)
mpl.rc('legend', borderpad = 0.3)
mpl.rc('legend', labelspacing = 0.0)
mpl.rc('legend', handletextpad = 0.)
mpl.rc('legend', borderaxespad = 0.5)
mpl.rc('legend', columnspacing = 0.)
mpl.rc('legend', handlelength = 1.)
# Plot Settings ###################################################

# System Parameters
# -------------------------------------------------------------------
Vd_vec = np.array([3., 15.])
Cg = 10.
params = [{'beta': 1.,
          'gamma': 1.,
          'Cg': Cg,
          'Vd': Vd,
          'Vin': 0.} for Vd in Vd_vec]
gate_vals = [np.arange(-Vd_vec[_]*Cg - Cg*1, Cg) for _ in range(len(Vd_vec))]
Ngate = [len(gate_vals[_]) for _ in range(len(Vd_vec))]
t0 = 1e-2
tf = 1e10
ntsteps = 500
nsteps_init = 1
dt_init = 1e15

# Set the initial state
# -------------------------------------------------------------------
state = []
for state_ind in range(len(Vd_vec)):
    # Set an initial state for Vin = 1
    init_state = np.zeros((2, 2, Ngate[state_ind]))
    ind = np.argmin(np.abs(gate_vals[state_ind] - 0))
    init_state[0, 0, ind] = 1.
    init_state = init_state.reshape(-1)
    state.append(init_state.copy())

    # Do a time evolution to get to steady state for Vin = 1
    paramsi = {'beta': 1.,
               'gamma': 1.,
               'Cg': Cg,
               'Vd': Vd_vec[state_ind],
               'Vin': Vd_vec[state_ind]}
    Ui = get_time_evolution_operator(paramsi,
                                     gate_vals[state_ind],
                                     dt_init)

    # Take a few time evolution steps
    for step in range(nsteps_init):
        state[state_ind] = time_evolution_step(paramsi,
                                               state[state_ind],
                                               gate_vals[state_ind],
                                               dt_init,
                                               U = Ui)

# Run time evolution
# -------------------------------------------------------------------
# Figure out which gate values are included in absorbing state
absorbing_inds = [None for _ in range(len(Vd_vec))]

# Set up time vector
tvec = np.logspace(np.log10(t0), np.log10(tf), ntsteps)
tvec = np.insert(tvec, 0, 0)

# Set up a figure
f, axes = plt.subplots(1, 2, figsize=(4, 2.5), sharey=True)

# Do an initial measurement
Vout = [[] for _ in range(len(Vd_vec))]
for i in range(len(Vd_vec)):
    vl = np.ones(state[i].shape)
    Vout[i].append(measure_gate_occ(state[i], vl, gate_vals[i]))

# Do the time Evolution
for step in range(ntsteps):

    # Take a time step
    print(f'{step}/{ntsteps}')
    dt = tvec[step+1] - tvec[step]
    state = [time_evolution_step(params[i], state[i], gate_vals[i], dt) for i in range(len(Vd_vec))]

    axes[0].clear()

    for Vdi in range(len(Vd_vec)):

        # Measure the results
        vl = np.ones(state[Vdi].shape)
        Vout[Vdi].append(measure_gate_occ(state[Vdi], vl, gate_vals[Vdi]))

        # Plot the results
        gate_vals_mat, t_mat = np.meshgrid(-gate_vals[Vdi]/Cg/Vd_vec[Vdi], tvec[:step+2])
        axes[0].contourf(t_mat, gate_vals_mat, np.array(Vout[Vdi]), 15, cmap=cmaps[Vdi])
        axes[0].plot([t0, tf], [0.95, 0.95], 'r:', linewidth=1)
        axes[0].set_xlabel(r'$t \ / \ \beta\hbar$', fontsize=11)
        axes[0].set_ylabel(r'$V_\mathrm{out}/V_\mathrm{d}$', fontsize=11)
        axes[0].set_ylim(-0.2, 1.2)
        axes[0].set_xscale('log')
        axes[0].set_xlim(t0, tf)
        axes[0].set_xticks([1e0, 1e4, 1e8])
        plt.tight_layout()
        plt.pause(0.01)

# Run time evolution in the opposite direction
# -------------------------------------------------------------------
params = [{'beta': 1.,
          'gamma': 1.,
          'Cg': Cg,
          'Vd': Vd,
          'Vin': Vd} for Vd in Vd_vec]
# Set up time vector
tvec = np.logspace(np.log10(t0), np.log10(tf), ntsteps)
tvec = np.insert(tvec, 0, 0)

# Do an initial measurement
Vout = [[] for _ in range(len(Vd_vec))]
for i in range(len(Vd_vec)):
    vl = np.ones(state[i].shape)
    Vout[i].append(measure_gate_occ(state[i], vl, gate_vals[i]))

# Do the time Evolution
for step in range(ntsteps):

    # Take a time step
    print(f'{step}/{ntsteps}')
    dt = tvec[step+1] - tvec[step]
    state = [time_evolution_step(params[i], state[i], gate_vals[i], dt) for i in range(len(Vd_vec))]

    axes[1].clear()

    for Vdi in range(len(Vd_vec)):

        # Measure the results
        vl = np.ones(state[Vdi].shape)
        Vout[Vdi].append(measure_gate_occ(state[Vdi], vl, gate_vals[Vdi]))

        # Plot the results
        gate_vals_mat, t_mat = np.meshgrid(-gate_vals[Vdi]/Cg/Vd_vec[Vdi], tvec[:step+2])
        axes[1].contourf(t_mat, gate_vals_mat, np.array(Vout[Vdi]), 15, cmap=cmaps[Vdi])
        axes[1].plot([t0, tf], [0.05, 0.05], 'r:', linewidth=1)
        axes[1].set_xlabel(r'$t \ / \ \beta\hbar$', fontsize=11)
        axes[1].set_ylim(-0.2, 1.2)
        axes[1].set_xscale('log')
        axes[1].set_xlim(t0, tf)
        axes[1].set_xticks([1e0, 1e4, 1e8])
        plt.tight_layout()
        plt.pause(0.01)

plt.subplots_adjust(wspace=0)
plt.pause(0.01)
plt.savefig('./data/time_evolution.pdf')

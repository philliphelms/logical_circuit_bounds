"""
This is a script to calculate all of the properties needed to analyze
the thermodynamic uncertainty relationships for a single NOT gate operating
in the charging (0 in -> 1 out) direction.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logical_circuit_bounds.not_gate.tools.ed import *
from logical_circuit_bounds.not_gate.tools.op_tools import *
cmap = plt.get_cmap('viridis') # Standard Cmap

# System Parameters
# -------------------------------------------------------------------
# System parameters
Vd_vec = np.arange(1., 31.)
Cg = 10.
alpha = 0.4
alpha_tol = 1e-16
t0 = 1e-2
tf = 1e16
ntsteps = 500

# Calculation Parameters
t0_init = 1e3
tf_init = 1e12
ntsteps_init = 100
plot = True

gate_vals = [np.arange(-Vd_vec[_]*Cg - Cg*2, Cg*2) for _ in range(len(Vd_vec))]
Ngate = [len(gate_vals[_]) for _ in range(len(Vd_vec))]
params = [{'beta': 1.,
          'gamma': 1.,
          'Cg': Cg,
          'Vd': Vd,
          'Vin': 0.} for Vd in Vd_vec]

# Set the initial state
# (here we use the steady-state of the NOT gate with the opposite input)
# -------------------------------------------------------------------
dt_init = 1e15
nsteps_init = 1
state = []
initial_density = [None for _ in range(len(Vd_vec))]
if plot:
    init_density_fig = plt.figure()
    init_density_ax = init_density_fig.add_subplot(111)
tvec = np.logspace(np.log10(t0_init), np.log10(tf_init), ntsteps_init)
tvec = np.insert(tvec, 0, 0)
absorbing_inds_rev = [np.where(-gate_vals[_]/Vd_vec[_]/Cg < (alpha))[0] for _ in range(len(Vd_vec))]

# Loop over all cross-voltages
for state_ind in range(len(Vd_vec)):

    # Set an initial state for Vin = 0
    init_state = np.zeros((2, 2, Ngate[state_ind]))
    ind = np.where(gate_vals[state_ind] == -Cg*Vd_vec[state_ind])[0]
    init_state[0, 0, ind] = 1.
    init_state = init_state.reshape(-1)
    state.append(init_state.copy())

    # Do a time evolution to get to steady state for Vin = 0
    paramsi = {'beta': 1.,
               'gamma': 1.,
               'Cg': Cg,
               'Vd': Vd_vec[state_ind],
               'Vin': 0.}
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

    # Now start doing an evolution back in the other direction for Vin = 1
    # and continue this evolution until we the probability of not being within
    # alpha of the correct measurement is less than alpha_tol
    for step in range(ntsteps_init):

        # Take the time step
        dt = tvec[step+1]-tvec[step]
        paramsi = {'beta': 1.,
                   'gamma': 1.,
                   'Cg': Cg,
                   'Vd': Vd_vec[state_ind],
                   'Vin': Vd_vec[state_ind]}
        state[state_ind] = time_evolution_step(paramsi, state[state_ind], gate_vals[state_ind], dt)

        # Do a measurement to see what the state looks like
        vl = np.ones(state[state_ind].shape)
        initial_density[state_ind] = measure_gate_occ(state[state_ind],
                                                      vl,
                                                      gate_vals[state_ind])

        # If we have passed the threshold, then end evolution
        if np.sum(initial_density[state_ind][absorbing_inds_rev[state_ind]]) > 1.-alpha_tol:
            break

    # Plot the resulting gate occupation distribution
    if plot:
        init_density_ax.semilogy(-gate_vals[state_ind]/(Cg*Vd_vec[state_ind]),
                             initial_density[state_ind],
                             '-', color=cmap(state_ind/len(Vd_vec)))
        init_density_ax.set_xlim(-0.2, 1.2)
        init_density_ax.set_ylim(1e-16, 1e0)
        plt.tight_layout()
        plt.pause(0.01)

# Run the Absorbing Calculation to Calculate FPT Distribution
# -------------------------------------------------------------------
# Figure out which gate values are included in absorbing state
absorbing_inds = [np.where(-gate_vals[_]/Vd_vec[_]/Cg >= (1-alpha))[0] for _ in range(len(Vd_vec))]

# Set up time vector
tvec = np.logspace(np.log10(t0), np.log10(tf), ntsteps)
tvec = np.insert(tvec, 0, 0)

# Set up a figure
if plot:
    f = plt.figure(figsize=(3, 8))
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(615)
    ax4 = f.add_subplot(616)

# Set up containers
S = [np.array([]) for _ in range(len(Vd_vec))]
dS_flow = [np.array([]) for _ in range(len(Vd_vec))]
current = [None for _ in range(len(Vd_vec))]
key_order = [('N1', 'S1'), ('S1', 'N1'), ('N1', 'P1'), ('P1', 'N1'), ('P1', 'D1'), ('D1', 'P1'), ('P1', 'G1'), ('G1', 'P1'), ('N1', 'G1'), ('G1', 'N1')]

# Do initial measurements
for i in range(len(Vd_vec)):

    # Get Flat state for contraction
    vl = np.ones(state[i].shape)

    # Calculate Absorption Variable
    S[i] = np.append(S[i], 1.-measure_target_probability_density(state[i], vl, gate_vals[i], absorbing_inds[i]))
    if plot:
        ax2.semilogx(tvec[i], S[i], '-', color=cmap((i+1)/len(Vd_vec)))

    # Measure Current Flow
    currenti = measure_current(state[i], vl, params[i], gate_vals[i], absorbing_inds=absorbing_inds[i])
    currenti = np.array([currenti[key] for key in key_order])
    if current[i] is None:
        current[i] = np.expand_dims(currenti, axis=0)
    else:
        current[i] = np.concatenate((current[i], np.expand_dims(currenti, axis=0)), axis=0)

    # Calculate first passage time distribution
    if len(S[i]) > 1:
        fS = -np.gradient(S[i], tvec[i])
        if plot:
            ax1.loglog(tvec[i], fS, '-', color=cmap((i+1)/len(Vd_vec)))

    # Calculate Entropy Production
    dS_flow[i] = np.append(dS_flow[i], measure_entropy_production(state[i], vl, params[i], gate_vals[i], absorbing_inds=absorbing_inds[i]))
    if plot:
        ax3.loglog(tvec[i],  dS_flow[i],  '-', color=cmap((i+1)/len(Vd_vec)))
        ax4.loglog(tvec[i], -dS_flow[i],  '-', color=cmap((i+1)/len(Vd_vec)))

if plot:
    ax1.set_ylabel(r'f(t)', fontsize=16)
    ax1.set_ylim(10**(-12), 10**-2)
    ax2.set_ylabel(r'$S(t)$', fontsize=16)
    ax3.set_ylabel(r'$\dot{S}_{flow}(t)$', fontsize=16)
    ax4.set_xlabel(r'$t/kT$', fontsize=16)
    ax4.set_ylabel(r'$-\dot{S}_{flow}(t)$', fontsize=16)
    ax4.invert_yaxis()
    plt.tight_layout()
    plt.pause(0.01)

# Do the time Evolution
for step in range(ntsteps):

    # Take a time step
    print(f'{step}/{ntsteps}')
    dt = tvec[step+1] - tvec[step]
    state = [absorbing_time_evolution_step(params[i], state[i], gate_vals[i], dt, absorbing_inds[i]) for i in range(len(Vd_vec))]

    # Do measurements on the state and add to plot
    if plot:
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

    for i in range(len(Vd_vec)):

        # Get Flat state for contraction
        vl = np.ones(state[i].shape)

        # Calculate Absorption Variable
        S[i] = np.append(S[i], 1.-measure_target_probability_density(state[i].copy(), vl.copy(), gate_vals[i], absorbing_inds[i]))
        if plot:
            ax2.semilogx(tvec[:step+2], S[i], '-', color=cmap((i+1)/len(Vd_vec)))

        # Measure Current Flow
        currenti = measure_current(state[i].copy(), vl.copy(), params[i], gate_vals[i], absorbing_inds=absorbing_inds[i])
        currenti = np.array([currenti[key] for key in key_order])
        if current[i] is None:
            current[i] = np.expand_dims(currenti, axis=0)
        else:
            current[i] = np.concatenate((current[i], np.expand_dims(currenti, axis=0)), axis=0)

        # Calculate first passage time distribution
        if len(S[i]) > 1:
            fS = -np.gradient(S[i], tvec[:step+2])
            if plot:
                ax1.loglog(tvec[:step+2], fS, '-', color=cmap((i+1)/len(Vd_vec)))

        # Calculate Entropy Production
        dS_flow[i] = np.append(dS_flow[i], measure_entropy_production(state[i].copy(), vl.copy(), params[i], gate_vals[i], absorbing_inds=absorbing_inds[i]))
        if plot:
            ax3.loglog(tvec[:step+2],  dS_flow[i],  '-', color=cmap((i+1)/len(Vd_vec)))
            ax4.loglog(tvec[:step+2], -dS_flow[i],  '-', color=cmap((i+1)/len(Vd_vec)))

    if plot:
        ax1.set_ylabel(r'f(t)', fontsize=16)
        ax1.set_ylim(10**(-12), 10**-2)
        ax2.set_ylabel(r'$S(t)$', fontsize=16)
        ax3.set_ylabel(r'$\dot{S}_{flow}(t)$', fontsize=16)
        ax4.set_xlabel(r'$t/kT$', fontsize=16)
        ax4.set_ylabel(r'$-\dot{S}_{flow}(t)$', fontsize=16)
        ax4.invert_yaxis()
        plt.tight_layout()
        plt.pause(0.01)
        plt.savefig(f'./data/first_passage_time_distribution_forward_alpha{alpha}_Cg{Cg}.pdf')

    np.savez(f'./data/first_passage_time_distribution_forward_alpha{alpha}_Cg{Cg}.npz',
             t=tvec[:step+2],
             S=S,
             state = state,
             current = current,
             dS_flow=dS_flow,
             Vd=Vd_vec,
             Cg=Cg,
             gate_vals=gate_vals,
             initial_density=initial_density,
             alpha=alpha)
plt.show()

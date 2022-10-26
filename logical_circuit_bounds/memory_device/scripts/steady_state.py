from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from logical_circuit_bounds.memory_device.tools.ed import *
cmap = plt.get_cmap('viridis')

# Set up evolution parameters
Cg = float(argv[1])
Vd = float(argv[2])
dt = float(argv[3])
initial_state = 'split' # Can also be steady, biased, or transition
evo_method = 'rk4'
lognsteps = 10

gate1_vals = np.arange(-(Vd+1)*Cg, 1*Cg)
gate2_vals = np.arange(-(Vd+1)*Cg, 1*Cg)
params = {'beta': 1.,
          'gamma': 1.,
          'Cg': Cg,
          'Vd': Vd}
nstep = int(10**lognsteps)
nsample = np.logspace(0, lognsteps, 5000)
nsample = [int(_) for _ in nsample]
sparse = True
plot = True
add_noise = 0.

# Set the initial state
Ngate1, Ngate2 = len(gate1_vals), len(gate2_vals)
init_state = np.zeros((2, 2, Ngate1, 2, 2, Ngate2))
if initial_state == 'split':
    ind1a = (np.abs(gate1_vals - 0)).argmin()
    ind1b = (np.abs(gate1_vals - (-Cg*Vd))).argmin()
    ind2a = (np.abs(gate2_vals - 0)).argmin()
    ind2b = (np.abs(gate2_vals - (-Cg*Vd))).argmin()
    init_state[0, 0, ind1a, 0, 0, ind2b] = 1/2
    init_state[0, 0, ind1b, 0, 0, ind2a] = 1/2
elif initial_state == 'transition':
    ind1a = (np.abs(gate1_vals - (-Cg*Vd/2.))).argmin()
    ind2b = (np.abs(gate2_vals - (-Cg*Vd/2.))).argmin()
    init_state[0, 0, ind1a, 0, 0, ind2b] = 1.
elif initial_state == 'biased':
    ind1a = (np.abs(gate1_vals - 0)).argmin()
    ind2b = (np.abs(gate2_vals - (-Cg*Vd))).argmin()
    init_state[0, 0, ind1a, 0, 0, ind2b] = 1.
elif initial_state == 'steady':
    init_state = get_steady_initial_state(params, gate1_vals, gate2_vals, biased=False)
else:
    raise ValueError('Invalid initial state')

# Figure out which gate values are included in absorbing state
absorbing_inds1 = []
absorbing_inds2 = []
plot_ind = np.argmin(np.abs(-gate1_vals/Vd/Cg - 0.5))

# Plot gate occupations
vl = np.ones(init_state.shape)
rhoG1i, rhoG2i = measure_gate_occs(init_state, vl, gate1_vals, gate2_vals)

if plot:
    f, ax = plt.subplots(1, 2, figsize=(5, 3))
    ax[0].semilogy(-gate1_vals/Cg/Vd, rhoG1i, '-', color=cmap(0))
    ax[0].semilogy(-gate2_vals/Cg/Vd, rhoG2i, '--', color=cmap(0))
    ax[0].set_ylim(1e-25, 1e0)
    ax[1].semilogx(0, -gate1_vals[plot_ind]/Cg/Vd, 'o', color=cmap(0))
    plt.pause(0.01)

# Add some random noise
init_state += add_noise * np.random.random(init_state.shape)
init_state /= np.sum(init_state)
rk4_state = init_state.reshape(-1)

# Get the time evolution operator
W = get_generator(params, gate1_vals, gate2_vals, sparse=sparse,
                  absorbing_inds1 = absorbing_inds1,
                  absorbing_inds2 = absorbing_inds2)
if evo_method == 'exact':
    U = get_time_evolution_operator(params, gate1_vals, gate2_vals, dt, W=W,
                                    absorbing_inds1 = absorbing_inds1,
                                    absorbing_inds2 = absorbing_inds2)
else:
    U = None

# Do the time evolution
tvec = np.array([])
S = np.array([])
dS_flow = np.array([])

for step in range(nstep):

    # Take a time step
    print(f'{step}/{nstep}')
    rk4_state = time_evolution_step(params, rk4_state, gate1_vals, gate2_vals, dt,
                                    W=W,
                                    U=U,
                                    method=evo_method,
                                    absorbing_inds1 = absorbing_inds1,
                                    absorbing_inds2 = absorbing_inds2)

    # Add to the plot occassionally
    if step in nsample:

        # Save time
        tvec = np.append(tvec, (step+1)*dt)

        # Get flat state for contraction
        vl = np.ones(rk4_state.shape)

        # Measure Gate Occupation
        rhoG1i, rhoG2i = measure_gate_occs(rk4_state, vl, gate1_vals, gate2_vals)
        if plot:
            ax[0].semilogy(-gate1_vals/Cg/Vd, rhoG1i, '-', color=cmap(np.log(step)/np.log(nstep)))
            ax[0].semilogy(-gate2_vals/Cg/Vd, rhoG2i, '--', color=cmap(np.log(step)/np.log(nstep)))
            ax[0].set_ylim(1e-25, 1e0)
            ax[1].loglog(step, rhoG1i[plot_ind], 'x', color=cmap(np.log(step)/np.log(nstep)))
            ax[1].loglog(step, rhoG2i[plot_ind], '+', color=cmap(np.log(step)/np.log(nstep)))
            plt.tight_layout()
            plt.pause(0.01)

        # Calculate absorption variable
        S = np.append(S, 1. - measure_target_probability_density(rk4_state, vl, gate1_vals, gate2_vals, absorbing_inds1, absorbing_inds2))

        # Calculate first passage time distribution
        if len(S) > 1:
            fS = -np.gradient(S, tvec)

        # Measure Entropy Flow
        dS_flow = np.append(dS_flow, measure_entropy_production(rk4_state, vl, params, gate1_vals, gate2_vals,
                                                                absorbing_inds1=absorbing_inds1,
                                                                absorbing_inds2=absorbing_inds2))
        # Format plot
        if plot:
            plt.pause(0.01)
            plt.savefig(f'./data/steady_state_evo_dt{dt}_Cg{Cg}.pdf')
        np.savez(f'./data/steady_state_Cg{Cg}_Vd{Vd}_dt{dt}_{initial_state}_initial_state.npz',
                 t=tvec,
                 S=S,
                 dS_flow=dS_flow,
                 Vd=Vd,
                 Cg=Cg,
                 gate1_vals = gate1_vals,
                 gate2_vals = gate2_vals)

plt.show()

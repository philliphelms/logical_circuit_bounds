from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from logical_circuit_bounds.memory_device.tools.ed import *
cmap = plt.get_cmap('viridis')

# Set up evolution parameters
Cg = float(argv[1])
Vd = float(argv[2])
dt = float(argv[3])
alpha = 0.4
lognsteps = 10
initial_state = 'steady' # other options are biased, transition, split
evo_method = 'rk4'

params = {'beta': 1.,
          'gamma': 1.,
          'Cg': Cg,
          'Vd': Vd}
gate1_vals = np.arange(-(Vd+1)*Cg, 1*Cg)
gate2_vals = np.arange(-(Vd+1)*Cg, 1*Cg)
nstep = int(10**lognsteps)
nsample = np.logspace(0, lognsteps, 5000)
nsample = [int(_) for _ in nsample]
add_noise = 0.
sparse = True
plot = True

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
    init_state = get_steady_initial_state(params, gate1_vals, gate2_vals)
else:
    raise ValueError('Invalid initial state')

# Plot gate occupations
if plot:
    f = plt.figure(figsize=(4, 6))
    ax0 = f.add_subplot(411)
    ax1 = f.add_subplot(412)
    ax2 = f.add_subplot(413)
    ax3 = f.add_subplot(817)
    ax4 = f.add_subplot(818)
vl = np.ones(init_state.shape)
rhoG1i, rhoG2i = measure_gate_occs(init_state, vl, gate1_vals, gate2_vals)
ax0.semilogy(-gate1_vals/Cg/Vd, rhoG1i, '+-')
ax0.semilogy(-gate2_vals/Cg/Vd, rhoG2i, 'x:')
ax0.set_ylim(1e-20, 1e0)
plt.pause(0.01)

# Add some random noise
init_state += add_noise * np.random.random(init_state.shape)
init_state /= np.sum(init_state)
rk4_state = init_state.reshape(-1)

# Figure out which gate values are included in absorbing state
absorbing_inds1 = [np.where(-gate1_vals/Vd/Cg > (1-alpha))[0][-1]]
absorbing_inds2 = [np.where(-gate2_vals/Vd/Cg <  (alpha))[0][0]]

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
        if plot:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

        # Save time
        tvec = np.append(tvec, (step+1)*dt)

        # Get flat state for contraction
        vl = np.ones(rk4_state.shape)

        # Calculate absorption variable
        S = np.append(S, 1. - measure_target_probability_density(rk4_state, vl, gate1_vals, gate2_vals, absorbing_inds1, absorbing_inds2))
        if plot:
            ax2.semilogx(tvec, S, 'bo-')

        # Calculate first passage time distribution
        if len(S) > 1:
            fS = -np.gradient(S, tvec)
            if plot:
                ax1.loglog(tvec, fS, 'bo-')

        # Measure Entropy Flow
        dS_flow = np.append(dS_flow, measure_entropy_production(rk4_state, vl, params, gate1_vals, gate2_vals,
                                                                absorbing_inds1=absorbing_inds1,
                                                                absorbing_inds2=absorbing_inds2))
        if plot:
            ax3.loglog(tvec,  dS_flow,  'bo-')
            ax4.loglog(tvec, -dS_flow,  'bo-')

        # Format plot
        if plot:
            ax1.set_ylabel(r'f(t)', fontsize=16)
            ax2.set_ylabel(r'$S(t)$', fontsize=16)
            ax3.set_ylabel(r'$\dot{S}$', fontsize=16)
            ax4.set_xlabel(r'$t$', fontsize=16)
            ax4.set_ylabel(r'$-\dot{S}$', fontsize=16)
            ax4.invert_yaxis()
            plt.tight_layout()
            plt.pause(0.01)
            plt.savefig(f'./data/first_passage_time_distribution_dt{dt}_Cg{Cg}.pdf')
        np.savez(f'./data/fpt_Cg{Cg}_Vd{Vd}_dt{dt}_{initial_state}_initial_state.npz',
                 t=tvec,
                 S=S,
                 dS_flow=dS_flow,
                 Vd=Vd,
                 Cg=Cg,
                 gate1_vals = gate1_vals,
                 gate2_vals = gate2_vals,
                 alpha=alpha)

plt.show()

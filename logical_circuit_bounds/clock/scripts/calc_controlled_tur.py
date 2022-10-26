import time
import numpy as np
import matplotlib.pyplot as plt
from logical_circuit_bounds.clock.tools.kmc import gill_alg
from sys import argv
cmap = plt.get_cmap('viridis')

# Set up parameters
Cg = 10.
Vd = float(argv[1])
plot_conv = True if len(argv) > 2 else False
plot = False # Plots trajectories in gill alg
tf = 1e20
maxiter = int(1e10)
max_markers = 5000
gamma = 1.
beta = 1.
alpha = 0.4
alpha_marker = 0.4
tsample = np.linspace(0, tf, 1000)
nsim = 100
params = {'Cg': Cg,
          'Vd': Vd,
          'beta': beta,
          'gamma': gamma,
          'alpha': alpha,
          'gate1_min': -Vd*Cg - Cg*2,
          'gate1_max': Cg*2,
          'gate2_min': -Vd*Cg - Cg*2,
          'gate2_max': Cg*2,
          'gate3_min': -Vd*Cg - Cg*2,
          'gate3_max': Cg*2}
signal1 = 'threshold_signal3'
signal2 = 'threshold_signal1'
signal3 = 'threshold_signal2'
max_corr = None

# Write Function to see if absorbing conditions are satisfied
TARGET_STATE = [1, 0, 1]
def time_marker_func(state):
    G1, G2, G3 = -state[2]/(Vd*Cg), -state[5]/(Vd*Cg), -state[8]/(Vd*Cg)
    mark_time = [False, False, False]
    # Check first gate
    if TARGET_STATE[0] == 1:
        if (G1 >= 1.-alpha_marker):
            mark_time[0] = True
            TARGET_STATE[0] = 0
    else:
        if (G1 < alpha_marker):
            TARGET_STATE[0] = 1
    # Check second gate
    if TARGET_STATE[1] == 1:
        if (G2 >= 1.-alpha_marker):
            mark_time[1] = True
            TARGET_STATE[1] = 0
    else:
        if (G2 < alpha_marker):
            TARGET_STATE[1] = 1
    # Check third gate
    if TARGET_STATE[2] == 1:
        if (G3 >= 1.-alpha_marker):
            mark_time[2] = True
            TARGET_STATE[2] = 0
    else:
        if (G3 < alpha_marker):
            TARGET_STATE[2] = 1
    # Return result
    return mark_time

# Create figure
if plot_conv:
    f, ax = plt.subplots(4, 1, figsize=(3, 5))

# Run many simulations
clock_times = []
clock_times_avg = []
clock_times_std = []
dQ_avg = []
Q_avg = []
for simi in range(nsim):

    # Set up an initial state
    state = np.array([1, 0, 0, # N1, P1, G1
                      1, 0, -Vd*Cg, # N2, P2, G2
                      1, 0, 0]) # N3, P3, G3
    t = []
    entropy = []
    tmax = 0.

    # Run a short simulation to get to equilibrate
    out = gill_alg(tf, params, state,
                   plot_markers = False,
                   plot_density = False,
                   maxiter = maxiter,
                   time_marker_func = time_marker_func,
                   time_marker_size = 3,
                   max_markers = 30,
                   plot_every = np.inf,
                   signal1 = signal1,
                   signal2 = signal2,
                   signal3 = signal3)
    state = np.array([1, 0, out[1][-1, 0],
                      1, 0, out[1][-1, 1],
                      1, 0, out[1][-1, 2]])

    # Run the simulation
    out = gill_alg(tf, params, state,
                   plot_markers = plot,
                   plot_density = plot,
                   maxiter = maxiter,
                   time_marker_func = time_marker_func,
                   time_marker_size = 3,
                   max_markers = max_markers,
                   signal1 = signal1,
                   signal2 = signal2,
                   signal3 = signal3)
    t_vec, gate_densities, entropy_vec, marked_times = out
    t_vec = list(t_vec)

    # Get FPT statistics
    clock_times_i  = list(np.array(marked_times[0][1:]) - np.array(marked_times[0][:-1]))
    clock_times_i += list(np.array(marked_times[1][1:]) - np.array(marked_times[1][:-1]))
    clock_times_i += list(np.array(marked_times[2][1:]) - np.array(marked_times[2][:-1]))

    clock_times.append(np.array(clock_times_i))
    clock_times_avg.append(np.nanmean(clock_times_i))
    clock_times_std.append(np.nanstd(clock_times_i))

    # Get Heat Dissipation Info
    Q_i = []
    dQ_i = []
    heat_inds_i = [np.abs(t_vec - _).argmin() for _ in marked_times[0]]
    for _ in range(len(heat_inds_i)-1):
        ind2, ind1 = heat_inds_i[_+1], heat_inds_i[_]
        Q_i.append(entropy_vec[ind2] - entropy_vec[ind1])
        dQ_i.append((entropy_vec[ind2] - entropy_vec[ind1])/
                    (t_vec[ind2] - t_vec[ind1]))
        print(dQ_i[-1])
    heat_inds_i = [np.abs(t_vec - _).argmin() for _ in marked_times[1]]
    for _ in range(len(heat_inds_i)-1):
        ind2, ind1 = heat_inds_i[_+1], heat_inds_i[_]
        Q_i.append(entropy_vec[ind2] - entropy_vec[ind1])
        dQ_i.append((entropy_vec[ind2] - entropy_vec[ind1])/
                    (t_vec[ind2] - t_vec[ind1]))
        print(dQ_i[-1])
    heat_inds_i = [np.abs(t_vec - _).argmin() for _ in marked_times[2]]
    for _ in range(len(heat_inds_i)-1):
        ind2, ind1 = heat_inds_i[_+1], heat_inds_i[_]
        Q_i.append(entropy_vec[ind2] - entropy_vec[ind1])
        dQ_i.append((entropy_vec[ind2] - entropy_vec[ind1])/
                    (t_vec[ind2] - t_vec[ind1]))
        print(dQ_i[-1])

    dQ_avg.append(np.nanmean(dQ_i))
    Q_avg.append(np.nanmean(Q_i))

    if plot_conv:
        ax[0].plot(simi, np.nanmean(Q_avg), 'bo')
        ax[1].plot(simi, np.nanmean(dQ_avg), 'bo')
        ax[2].plot(simi, np.nanmean(clock_times_avg), 'bo')
        ax[3].plot(simi, np.nanmean(clock_times_std), 'bo')
        plt.tight_layout()
        plt.pause(0.01)

    # Save results
    np.savez(f'./data/controlled_tur_data_Vd{Vd}_Cg{Cg}.npz',
             Cg = Cg,
             Vd = Vd,
             alpha_marker = alpha_marker,
             t_vec = t_vec,
             gate_densities = gate_densities,
             entropy_vec = entropy_vec,
             tsample=tsample,
             clock_times=clock_times,
             clock_times_avg = clock_times_avg,
             clock_times_std = clock_times_std,
             dQ_avg = dQ_avg,
             Q_avg = Q_avg)


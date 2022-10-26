import time
import numpy as np
import matplotlib.pyplot as plt
from logical_circuit_bounds.clock.tools.kmc import gill_alg
from sys import argv
cmap = plt.get_cmap('viridis')

# Set up parameters
Cg = 10.
Vd = float(argv[1])
plot = True if len(argv) > 2 else False
tf = 1e20
maxiter = int(1e10)
max_markers = 5000
gamma = 1.
beta = 1.
alpha = np.inf
alpha_test = 0.4
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
signal1 = 'signal3'
signal2 = 'signal1'
signal3 = 'signal2'
max_corr = None

# Write Function to see if absorbing conditions are satisfied
TARGET_STATE = [1, 0, 1]
def time_marker_func(state):
    G1, G2, G3 = -state[2]/(Vd*Cg), -state[5]/(Vd*Cg), -state[8]/(Vd*Cg)
    mark_time = [False, False, False]
    # Check first gate
    if TARGET_STATE[0] == 1:
        if (G1 >= 1.-alpha_test):
            mark_time[0] = True
            TARGET_STATE[0] = 0
    else:
        if (G1 < alpha_test):
            TARGET_STATE[0] = 1
    # Check second gate
    if TARGET_STATE[1] == 1:
        if (G2 >= 1.-alpha_test):
            mark_time[1] = True
            TARGET_STATE[1] = 0
    else:
        if (G2 < alpha_test):
            TARGET_STATE[1] = 1
    # Check third gate
    if TARGET_STATE[2] == 1:
        if (G3 >= 1.-alpha_test):
            mark_time[2] = True
            TARGET_STATE[2] = 0
    else:
        if (G3 < alpha_test):
            TARGET_STATE[2] = 1
    # Return result
    return mark_time

# Create a function to calc autocorrelation func
def calc_acf(var, max_corr=np.inf):
    acf = np.zeros(min(len(var), max_corr))
    last_el = np.where(np.isnan(var))[0]
    if len(last_el) == 0:
        last_el = len(var)
    else:
        last_el = last_el[0]
    for i1 in range(min(last_el, max_corr)):
        if ((i1 % int(max_corr/20)) == 0):
            print(i1/min(len(var), max_corr)*100, '% Progress on ACF Calc')
        acf[i1] = np.dot(var[i1:last_el], var[:last_el-i1])/(last_el-i1)
    return acf

# Set up a figure
f, ax = plt.subplots(1, 1, figsize=(6, 3))

# Run the simulation many times
signal_fft = []
signal_acf = []
signal_acf_fft = []
max_signal = 100
tsample = None
for simi in range(nsim):
    t = []
    entropy = []
    tmax = 0.

    # Run a short simulation to get to equilibrate
    if simi == 0:

        # Set up an initial state
        state = np.array([1, 0, 0, # N1, P1, G1
                          1, 0, -Vd*Cg, # N2, P2, G2
                          1, 0, 0]) # N3, P3, G3

        # Run a short trajectory
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

        # Save the resulting state
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
                   plot_every = 5000,
                   signal1 = signal1,
                   signal2 = signal2,
                   signal3 = signal3)

    t_vec, gate_densities, entropy_vec, marked_times = out

    state = np.array([1, 0, out[1][-1, 0],
                      1, 0, out[1][-1, 1],
                      1, 0, out[1][-1, 2]])

    # Interpolate to get signals at even temporal spacing
    if tsample is None:
        tsample = np.linspace(0, t_vec[-1], len(t_vec/5))
        if max_corr is None:
            max_corr = len(tsample/10000)
    sampled_gate_densities = np.zeros((len(tsample), 3))
    sampled_gate_densities[:, 0] = np.interp(tsample, t_vec, gate_densities[:, 0], right=np.nan)
    sampled_gate_densities[:, 1] = np.interp(tsample, t_vec, gate_densities[:, 1], right=np.nan)
    sampled_gate_densities[:, 2] = np.interp(tsample, t_vec, gate_densities[:, 2], right=np.nan)
    sampled_gate_densities = -sampled_gate_densities/(Cg*Vd)

    # Calculate a time correlation function
    acf1 = calc_acf(sampled_gate_densities[:, 0], max_corr=max_corr)
    acf2 = calc_acf(sampled_gate_densities[:, 1], max_corr=max_corr)
    acf3 = calc_acf(sampled_gate_densities[:, 2], max_corr=max_corr)
    avg_acf = np.nanmean(np.array([acf1, acf2, acf3]), axis=0)
    signal_acf.append(avg_acf)

    # Plot ACF
    ax.clear()
    avg = np.nanmean(np.array(signal_acf), axis=0)
    std = np.nanstd(np.array(signal_acf), axis=0)
    x = np.arange(max_corr)*(tsample[1]-tsample[0])
    ax.plot(x, avg, 'b', linewidth=2)
    ax.fill_between(x, avg-std, y2=avg+std, color='b', alpha=0.2)
    ax.set_xlim(0, tsample[-1]/5)

    plt.pause(0.01)
    np.savez(f'./data/acf_data_Vd{Vd}_Cg{Cg}.npz',
             Cg = Cg,
             Vd = Vd,
             t_vec = t_vec,
             gate_densities = gate_densities,
             entropy_vec = entropy_vec,
             tsample=tsample,
             signal_acf=signal_acf)

import time
import numpy as np
import matplotlib.pyplot as plt

# Define the gillespie algorithm
def gill_alg(tf, params, state,
             absorbing_inds1 = [],
             absorbing_inds2 = [],
             absorbing_inds3 = [],
             signal1 = 'threshold_signal3',
             signal2 = 'threshold_signal1',
             signal3 = 'threshold_signal2',
             maxiter = 100000000,
             absorbing_func = None,
             plot_every = 1000,
             time_marker_func = None,
             time_marker_size = None,
             max_markers = np.inf,
             calc_signal_acf = True,
             plot_density = False,
             plot_markers = False):

    # Get parameters
    beta = params['beta']
    gamma = params['gamma']
    Vd = params['Vd']
    Cg = params['Cg']
    if 'alpha' in params:
        alpha = params['alpha']
    else:
        alpha = None

    gate1_min = params['gate1_min']
    gate1_max = params['gate1_max']
    gate2_min = params['gate2_min']
    gate2_max = params['gate2_max']
    gate3_min = params['gate3_min']
    gate3_max = params['gate3_max']

    # How the particle count changes for each transition above
    #              N1 P1 G1 N2 P2 G2 N3 P3 G3
    V = np.array([[-1, 0, 0, 0, 0, 0, 0, 0, 0], # N1 -> S1
                  [+1, 0, 0, 0, 0, 0, 0, 0, 0], # S1 -> N1
                  [-1,+1, 0, 0, 0, 0, 0, 0, 0], # N1 -> P1
                  [+1,-1, 0, 0, 0, 0, 0, 0, 0], # P1 -> N1
                  [ 0,-1, 0, 0, 0, 0, 0, 0, 0], # P1 -> D1
                  [ 0,+1, 0, 0, 0, 0, 0, 0, 0], # D1 -> P1
                  [ 0,-1,+1, 0, 0, 0, 0, 0, 0], # P1 -> G1
                  [ 0,+1,-1, 0, 0, 0, 0, 0, 0], # G1 -> P1
                  [-1, 0,+1, 0, 0, 0, 0, 0, 0], # N1 -> G1
                  [+1, 0,-1, 0, 0, 0, 0, 0, 0], # G1 -> N1
                  [ 0, 0, 0,-1, 0, 0, 0, 0, 0], # N2 -> S2
                  [ 0, 0, 0,+1, 0, 0, 0, 0, 0], # S2 -> N2
                  [ 0, 0, 0,-1,+1, 0, 0, 0, 0], # N2 -> P2
                  [ 0, 0, 0,+1,-1, 0, 0, 0, 0], # P2 -> N2
                  [ 0, 0, 0, 0,-1, 0, 0, 0, 0], # P2 -> D2
                  [ 0, 0, 0, 0,+1, 0, 0, 0, 0], # D2 -> P2
                  [ 0, 0, 0, 0,-1,+1, 0, 0, 0], # P2 -> G2
                  [ 0, 0, 0, 0,+1,-1, 0, 0, 0], # G2 -> P2
                  [ 0, 0, 0,-1, 0,+1, 0, 0, 0], # N2 -> G2
                  [ 0, 0, 0,+1, 0,-1, 0, 0, 0], # G2 -> N2
                  [ 0, 0, 0, 0, 0, 0,-1, 0, 0], # N3 -> S3
                  [ 0, 0, 0, 0, 0, 0,+1, 0, 0], # S3 -> N3
                  [ 0, 0, 0, 0, 0, 0,-1,+1, 0], # N3 -> P3
                  [ 0, 0, 0, 0, 0, 0,+1,-1, 0], # P3 -> N3
                  [ 0, 0, 0, 0, 0, 0, 0,-1, 0], # P3 -> D3
                  [ 0, 0, 0, 0, 0, 0, 0,+1, 0], # D3 -> P3
                  [ 0, 0, 0, 0, 0, 0, 0,-1,+1], # P3 -> G3
                  [ 0, 0, 0, 0, 0, 0, 0,+1,-1], # G3 -> P3
                  [ 0, 0, 0, 0, 0, 0,-1, 0,+1], # N3 -> G3
                  [ 0, 0, 0, 0, 0, 0,+1, 0,-1]])# G3 -> N3

    # Containers for storing results
    t_vec = np.zeros((maxiter))
    gate_densities = np.zeros((maxiter, 3))
    entropy_vec = np.zeros((maxiter))
    entropy = 0.
    gate_densities[0, 0] = state[2]
    gate_densities[0, 1] = state[5]
    gate_densities[0, 2] = state[8]
    t = 0.
    niter = 0
    if time_marker_func is not None:
        marked_times = [[] for _ in range(time_marker_size)]

    # Set up figure
    if plot_density:
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(t, -gate_densities[niter, 0]/(Cg*Vd), 'k.')
        ax.plot(t, -gate_densities[niter, 1]/(Cg*Vd), 'r.')
        ax.plot(t, -gate_densities[niter, 2]/(Cg*Vd), 'b.')
        ax.set_xlabel(r'$t$', fontsize=14)
        ax.set_ylabel(r'$\sigma_{out}$', fontsize=14)
        plt.tight_layout()
        plt.pause(0.01)
    if plot_markers:
        f2, ax2 = plt.subplots(2, 1, figsize=(3, 3))

    # Run Gillespie Simulation
    while (t < tf) and (niter < maxiter-1):

        # Determine Energies
        G1 = state[2]
        G2 = state[5]
        G3 = state[8]
        if signal1[-7:] == 'signal3':
            if signal1[:9] == 'threshold' and -G3/(Cg*Vd) > 1.-alpha:
                epsilon_P1 = 0.0 - Vd
                epsilon_N1 = -1.5*Vd + Vd
            elif signal1[:9] == 'threshold' and -G3/(Cg*Vd) < alpha:
                epsilon_P1 = 0.0
                epsilon_N1 = -1.5*Vd
            else:
                epsilon_P1 = 0.0 + G3/Cg
                epsilon_N1 = -1.5*Vd - G3/Cg
        elif signal1[-7:] == 'signal2':
            if signal1[:9] == 'threshold' and -G2/(Cg*Vd) > 1.-alpha:
                epsilon_P1 = 0.0 - Vd
                epsilon_N1 = -1.5*Vd + Vd
            elif signal1[:9] == 'threshold' and -G2/(Cg*Vd) < alpha:
                epsilon_P1 = 0.0
                epsilon_N1 = -1.5*Vd
            else:
                epsilon_P1 = 0.0 + G2/Cg
                epsilon_N1 = -1.5*Vd - G2/Cg
        elif signal1[-7:] == 'signal1':
            if signal1[:9] == 'threshold' and -G1/(Cg*Vd) > 1.-alpha:
                epsilon_P1 = 0.0 - Vd
                epsilon_N1 = -1.5*Vd + Vd
            elif signal1[:9] == 'threshold' and -G1/(Cg*Vd) < alpha:
                epsilon_P1 = 0.0
                epsilon_N1 = -1.5*Vd
            else:
                epsilon_P1 = 0.0 + G1/Cg
                epsilon_N1 = -1.5*Vd - G1/Cg
        else:
            epsilon_P1 = 0.0 - signal1
            epsilon_N1 = -1.5*Vd + signal1

        if signal2[-7:] == 'signal3':
            if signal2[:9] == 'threshold' and -G3/(Cg*Vd) > 1.-alpha:
                epsilon_P2 = 0.0 - Vd
                epsilon_N2 = -1.5*Vd + Vd
            elif signal2[:9] == 'threshold' and -G3/(Cg*Vd) < alpha:
                epsilon_P2 = 0.0
                epsilon_N2 = -1.5*Vd
            else:
                epsilon_P2 = 0.0 + G3/Cg
                epsilon_N2 = -1.5*Vd - G3/Cg
        elif signal2[-7:] == 'signal2':
            if signal2[:9] == 'threshold' and -G2/(Cg*Vd) > 1.-alpha:
                epsilon_P2 = 0.0 - Vd
                epsilon_N2 = -1.5*Vd + Vd
            elif signal2[:9] == 'threshold' and -G2/(Cg*Vd) < alpha:
                epsilon_P2 = 0.0
                epsilon_N2 = -1.5*Vd
            else:
                epsilon_P2 = 0.0 + G2/Cg
                epsilon_N2 = -1.5*Vd - G2/Cg
        elif signal2[-7:] == 'signal1':
            if signal2[:9] == 'threshold' and -G1/(Cg*Vd) > 1.-alpha:
                epsilon_P2 = 0.0 - Vd
                epsilon_N2 = -1.5*Vd + Vd
            elif signal2[:9] == 'threshold' and -G1/(Cg*Vd) < alpha:
                epsilon_P2 = 0.0
                epsilon_N2 = -1.5*Vd
            else:
                epsilon_P2 = 0.0 + G1/Cg
                epsilon_N2 = -1.5*Vd - G1/Cg
        else:
            epsilon_P2 = 0.0 - signal1
            epsilon_N2 = -1.5*Vd + signal1

        if signal3[-7:] == 'signal3':
            if signal3[:9] == 'threshold' and -G3/(Cg*Vd) > 1.-alpha:
                epsilon_P3 = 0.0 - Vd
                epsilon_N3 = -1.5*Vd + Vd
            elif signal3[:9] == 'threshold' and -G3/(Cg*Vd) < alpha:
                epsilon_P3 = 0.0
                epsilon_N3 = -1.5*Vd
            else:
                epsilon_P3 = 0.0 + G3/Cg
                epsilon_N3 = -1.5*Vd - G3/Cg
        elif signal3[-7:] == 'signal2':
            if signal3[:9] == 'threshold' and -G2/(Cg*Vd) > 1.-alpha:
                epsilon_P3 = 0.0 - Vd
                epsilon_N3 = -1.5*Vd + Vd
            elif signal3[:9] == 'threshold' and -G2/(Cg*Vd) < alpha:
                epsilon_P3 = 0.0
                epsilon_N3 = -1.5*Vd
            else:
                epsilon_P3 = 0.0 + G2/Cg
                epsilon_N3 = -1.5*Vd - G2/Cg
        elif signal3[-7:] == 'signal1':
            if signal3[:9] == 'threshold' and -G1/(Cg*Vd) > 1.-alpha:
                epsilon_P3 = 0.0 - Vd
                epsilon_N3 = -1.5*Vd + Vd
            elif signal3[:9] == 'threshold' and -G1/(Cg*Vd) < alpha:
                epsilon_P3 = 0.0
                epsilon_N3 = -1.5*Vd
            else:
                epsilon_P3 = 0.0 + G1/Cg
                epsilon_N3 = -1.5*Vd - G1/Cg
        else:
            epsilon_P3 = 0.0 - signal1
            epsilon_N3 = -1.5*Vd + signal1

        # Compute rates at which particles hop between sites
        c = np.zeros(30)
        c[0]  = gamma / (np.exp(beta*epsilon_N1)+1.)                                    # N1 -> S1
        c[1]  = gamma * (1. - 1. / (np.exp(beta*epsilon_N1)+1.))                        # S1 -> N1
        c[2]  = gamma * ( 1. / (np.exp(beta*(abs(epsilon_P1-epsilon_N1)))-1.+1e-6))     # N1 -> P1
        c[3]  = gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P1-epsilon_N1)))-1.+1e-6)) # P1 -> N1
        c[4]  = gamma * ( 1. / (np.exp(beta*(epsilon_P1-Vd))+1.))                       # P1 -> D1
        c[5]  = gamma * (1. - 1. / (np.exp(beta*(epsilon_P1-Vd))+1.))                   # D1 -> P1
        c[6]  = gamma*1./(np.exp(beta*(epsilon_P1+G1/Cg))+1.)                           # P1 -> G1
        c[7]  = gamma*(1-1./(np.exp(beta*(epsilon_P1+G1/Cg))+1.))                       # G1 -> P1
        c[8]  = gamma*1./(np.exp(beta*(epsilon_N1+G1/Cg))+1.)                           # N1 -> G1
        c[9]  = gamma*(1-1./(np.exp(beta*(epsilon_N1+G1/Cg))+1.))                       # G1 -> N1

        c[10] = gamma / (np.exp(beta*epsilon_N2)+1.)                                    # N2 -> S2
        c[11] = gamma * (1. - 1. / (np.exp(beta*epsilon_N2)+1.))                        # S2 -> N2
        c[12] = gamma * ( 1. / (np.exp(beta*(abs(epsilon_P2-epsilon_N2)))-1.+1e-6))     # N2 -> P2
        c[13] = gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P2-epsilon_N2)))-1.+1e-6)) # P2 -> N2
        c[14] = gamma * ( 1. / (np.exp(beta*(epsilon_P2-Vd))+1.))                       # P2 -> D2
        c[15] = gamma * (1. - 1. / (np.exp(beta*(epsilon_P2-Vd))+1.))                   # D2 -> P2
        c[16] = gamma*1./(np.exp(beta*(epsilon_P2+G2/Cg))+1.)                           # P2 -> G2
        c[17] = gamma*(1-1./(np.exp(beta*(epsilon_P2+G2/Cg))+1.))                       # G2 -> P2
        c[18] = gamma*1./(np.exp(beta*(epsilon_N2+G2/Cg))+1.)                           # N2 -> G2
        c[19] = gamma*(1-1./(np.exp(beta*(epsilon_N2+G2/Cg))+1.))                       # G2 -> N2

        c[20] = gamma / (np.exp(beta*epsilon_N3)+1.)                                    # N3 -> S3
        c[21] = gamma * (1. - 1. / (np.exp(beta*epsilon_N3)+1.))                        # S3 -> N3
        c[22] = gamma * ( 1. / (np.exp(beta*(abs(epsilon_P3-epsilon_N3)))-1.+1e-6))     # N3 -> P3
        c[23] = gamma * (1. + 1. / (np.exp(beta*(abs(epsilon_P3-epsilon_N3)))-1.+1e-6)) # P3 -> N3
        c[24] = gamma * ( 1. / (np.exp(beta*(epsilon_P3-Vd))+1.))                       # P3 -> D3
        c[25] = gamma * (1. - 1. / (np.exp(beta*(epsilon_P3-Vd))+1.))                   # D3 -> P3
        c[26] = gamma*1./(np.exp(beta*(epsilon_P3+G3/Cg))+1.)                           # P3 -> G3
        c[27] = gamma*(1-1./(np.exp(beta*(epsilon_P3+G3/Cg))+1.))                       # G3 -> P3
        c[28] = gamma*1./(np.exp(beta*(epsilon_N3+G3/Cg))+1.)                           # N3 -> G3
        c[29] = gamma*(1-1./(np.exp(beta*(epsilon_N3+G3/Cg))+1.))                       # G3 -> N3

        # Get the probability of each possible transition
        a = np.zeros(30)
        a[0] = c[0] * state[0]
        a[1] = c[1] * (1.-state[0])
        a[2] = c[2] * (state[0]) * (1.-state[1])
        a[3] = c[3] * (1.-state[0]) * (state[1])
        a[4] = c[4] * state[1]
        a[5] = c[5] * (1.-state[1])
        a[6] = c[6] * state[1] if state[2] < gate1_max else 0.
        a[7] = c[7] * (1.-state[1]) if state[2] > gate1_min else 0.
        a[8] = c[8] * state[0] if state[2] < gate1_max else 0.
        a[9] = c[9] * (1.-state[0]) if state[2] > gate1_min else 0.

        a[10] = c[10] * state[3]
        a[11] = c[11] * (1.-state[3])
        a[12] = c[12] * (state[3]) * (1.-state[4])
        a[13] = c[13] * (1.-state[3]) * (state[4])
        a[14] = c[14] * state[4]
        a[15] = c[15] * (1.-state[4])
        a[16] = c[16] * state[4] if state[5] < gate2_max else 0.
        a[17] = c[17] * (1.-state[4]) if state[5] > gate1_min else 0.
        a[18] = c[18] * state[3] if state[5] < gate1_max else 0.
        a[19] = c[19] * (1.-state[3]) if state[5] > gate1_min else 0.

        a[20] = c[20] * state[6]
        a[21] = c[21] * (1.-state[6])
        a[22] = c[22] * (state[6]) * (1.-state[7])
        a[23] = c[23] * (1.-state[6]) * (state[7])
        a[24] = c[24] * state[7]
        a[25] = c[25] * (1.-state[7])
        a[26] = c[26] * state[7] if state[8] < gate1_max else 0.
        a[27] = c[27] * (1.-state[7]) if state[8] > gate1_min else 0.
        a[28] = c[28] * state[6] if state[8] < gate1_max else 0.
        a[29] = c[29] * (1.-state[6]) if state[8] > gate1_min else 0.

        asum = np.sum(a)

        # Get the time step size
        tau = np.log(1./np.random.rand())/asum
        t = t + tau

        # Determine which reaction to do
        asum = np.cumsum(a/asum)
        transition_ind = np.nonzero(np.random.rand() < asum)
        transition_ind = np.min(transition_ind[0])

        # Make the transition
        state = state + V[transition_ind]

        # Calculate the entropy generation
        prev_entropy = entropy
        if transition_ind == 0:
            entropy += -beta*epsilon_N1
        elif transition_ind == 1:
            entropy +=  beta*epsilon_N1
        elif transition_ind == 2:
            entropy += -beta*abs(epsilon_P1-epsilon_N1)
        elif transition_ind == 3:
            entropy +=  beta*abs(epsilon_P1-epsilon_N1)
        elif transition_ind == 4:
            entropy += -beta*(epsilon_P1-Vd)
        elif transition_ind == 5:
            entropy +=  beta*(epsilon_P1-Vd)
        elif transition_ind == 6:
            entropy += -beta*(epsilon_P1+G1/Cg)
        elif transition_ind == 7:
            entropy +=  beta*(epsilon_P1+G1/Cg)
        elif transition_ind == 8:
            entropy += -beta*(epsilon_N1+G1/Cg)
        elif transition_ind == 9:
            entropy +=  beta*(epsilon_N1+G1/Cg)
        elif transition_ind == 10:
            entropy += -beta*epsilon_N2
        elif transition_ind == 11:
            entropy +=  beta*epsilon_N2
        elif transition_ind == 12:
            entropy += -beta*abs(epsilon_P2-epsilon_N2)
        elif transition_ind == 13:
            entropy +=  beta*abs(epsilon_P2-epsilon_N2)
        elif transition_ind == 14:
            entropy += -beta*(epsilon_P2-Vd)
        elif transition_ind == 15:
            entropy +=  beta*(epsilon_P2-Vd)
        elif transition_ind == 16:
            entropy += -beta*(epsilon_P2+G2/Cg)
        elif transition_ind == 17:
            entropy +=  beta*(epsilon_P2+G2/Cg)
        elif transition_ind == 18:
            entropy += -beta*(epsilon_N2+G2/Cg)
        elif transition_ind == 19:
            entropy +=  beta*(epsilon_N2+G2/Cg)
        elif transition_ind == 20:
            entropy += -beta*epsilon_N3
        elif transition_ind == 21:
            entropy +=  beta*epsilon_N3
        elif transition_ind == 22:
            entropy += -beta*abs(epsilon_P3-epsilon_N3)
        elif transition_ind == 23:
            entropy +=  beta*abs(epsilon_P3-epsilon_N3)
        elif transition_ind == 24:
            entropy += -beta*(epsilon_P3-Vd)
        elif transition_ind == 25:
            entropy +=  beta*(epsilon_P3-Vd)
        elif transition_ind == 26:
            entropy += -beta*(epsilon_P3+G3/Cg)
        elif transition_ind == 27:
            entropy +=  beta*(epsilon_P3+G3/Cg)
        elif transition_ind == 28:
            entropy += -beta*(epsilon_N3+G3/Cg)
        elif transition_ind == 29:
            entropy +=  beta*(epsilon_N3+G3/Cg)

        # Save data
        niter += 1
        t_vec[niter] = t
        gate_densities[niter, 0] = state[2]
        gate_densities[niter, 1] = state[5]
        gate_densities[niter, 2] = state[8]
        entropy_vec[niter] = entropy

        # Print Progress
        if niter % plot_every == 0:
            if t/tf > niter/maxiter:
                if (time_marker_func is not None) and \
                        (len(marked_times[0])+len(marked_times[1])+len(marked_times[2]))/max_markers > t/tf:
                    print(f'{(len(marked_times[0])+len(marked_times[1])+len(marked_times[2]))/max_markers*100} % marked times')
                else:
                    print(f'{t/tf*100} % time')
            else:
                if (time_marker_func is not None) and \
                        (len(marked_times[0])+len(marked_times[1])+len(marked_times[2]))/max_markers > niter/maxiter:
                    print(f'{(len(marked_times[0])+len(marked_times[1])+len(marked_times[2]))/max_markers*100} % marked times')
                else:
                    print(f'{niter/maxiter*100} % iter')

        # Update plot
        if (niter % plot_every == 0):
            if plot_density:

                ax.clear()
                ax.plot(t_vec[:niter], -gate_densities[:niter, 0]/(Cg*Vd), 'r-')
                ax.plot(t_vec[:niter], -gate_densities[:niter, 1]/(Cg*Vd), 'b-')
                ax.plot(t_vec[:niter], -gate_densities[:niter, 2]/(Cg*Vd), 'k-')
                for tt_i in range(len(marked_times)):
                    ax.plot([marked_times[tt_i], marked_times[tt_i]],
                            [0, 1], 'k-')

                plt.sca(ax)
                plt.tight_layout()
                plt.pause(0.01)

            if plot_markers and all([len(_)>2 for _ in marked_times]): #(len(marked_times[0]) > 2) and (len(transition_times[1]) > 2) and (len(transition_times[2]) > 2):
                transition_times = marked_times.copy()
                for _ in range(3):
                    transition_times[_] = np.array(transition_times[_])
                    transition_times[_] = transition_times[_][1:] - transition_times[_][:-1]
                transition_times = [list(_) for _ in transition_times]
                all_transition_times = []
                for _ in range(len(transition_times)):
                    all_transition_times += transition_times[_]
                transition_times = all_transition_times
                bins = np.linspace(min(transition_times), max(transition_times), 50)
                log_bins = np.logspace(np.log10(min(transition_times)), np.log10(max(transition_times)), 50)
                hist, bins = np.histogram(transition_times, bins=bins, density=True)
                log_hist, log_bins = np.histogram(transition_times, bins=log_bins, density=True)

                ax2[0].clear()
                ax2[0].plot(log_bins[1:], log_hist, '-')
                ax2[0].set_xlabel(r'$t$', fontsize=14)
                ax2[0].set_yscale('log')
                ax2[0].set_xscale('log')
                ax2[1].clear()
                ax2[1].plot(log_bins[1:], log_hist, '-')
                ax2[1].set_xlabel(r'$\dot{t}$', fontsize=14)
                ax2[1].set_xscale('log')

                plt.sca(ax2[0])
                plt.tight_layout()
                plt.pause(0.01)

                # Save transition times
                np.savez(f'./data/clock_kmc_transition_time_Cg{Cg}_Vd{Vd}_alpha{alpha}.npz',
                         transition_times = transition_times,
                         Cg = Cg,
                         Vd = Vd)


        # Check if absorbing condition is met
        if absorbing_func is not None:
            if absorbing_func(state):
                # Update plot
                if plot_density:

                    ax.clear()
                    ax.plot(t_vec[:niter], -gate_densities[:niter, 0]/(Cg*Vd), 'r-')
                    ax.plot(t_vec[:niter], -gate_densities[:niter, 1]/(Cg*Vd), 'b-')
                    ax.plot(t_vec[:niter], -gate_densities[:niter, 2]/(Cg*Vd), 'k-')

                    plt.tight_layout()
                    plt.pause(0.01)
                break

        # Check to see if the transition being made needs marked
        if time_marker_func is not None:
            marker_bool = time_marker_func(state)
            if any(marker_bool):
                for marker_ind in range(time_marker_size):
                    if marker_bool[marker_ind]:
                        marked_times[marker_ind].append(t)
                if len(marked_times[0])+len(marked_times[1])+len(marked_times[2]) > max_markers:
                    break



    # Return results
    if time_marker_func is not None:
        return  t_vec[:niter], gate_densities[:niter, :], entropy_vec[:niter], marked_times
    else:
        return  t_vec[:niter], gate_densities[:niter, :], entropy_vec[:niter]

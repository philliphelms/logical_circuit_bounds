# Imports necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt

# Define Gillespie algorithm
def gill_alg(params, pop, plot=False):

    # Set initial time
    t = 0

    # Get parameters
    tfinal=params[0]
    Vd=params[1]
    Cg=params[2]
    gamma=params[3]
    beta = params[4]
    max_gate_val = params[5]
    min_gate_val = params[6]

    # How the particle count changes for each transition above
    V = np.array([[-1,0,0,0,0,0],
                  [1,0,0,0,0,0],
                  [-1,1,0,0,0,0],
                  [1,-1,0,0,0,0],
                  [0,-1,0,0,0,0],
                  [0,1,0,0,0,0],
                  [0,-1,1,0,0,0],
                  [0,1,-1,0,0,0],
                  [-1,0,1,0,0,0],
                  [1,0,-1,0,0,0],
                  [0,0,0,-1,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,-1,1,0],
                  [0,0,0,1,-1,0],
                  [0,0,0,0,-1,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,-1,1],
                  [0,0,0,0,1,-1],
                  [0,0,0,-1,0,1],
                  [0,0,0,1,0,-1.]])

    # Set up containers for results
    c=np.zeros(20)
    data=np.zeros((100000000,8))
    entropy=0.
    data[0,0]=0.
    data[0,1]=pop[0]
    data[0,2]=pop[1]
    data[0,3]=pop[2]
    data[0,4]=pop[3]
    data[0,5]=pop[4]
    data[0,6]=pop[5]
    jumps=0
    stat=0

    if plot:
        f = plt.figure()
        ax = f.add_subplot(211)
        ax1 = f.add_subplot(212)

    # Run Gillespie Simulation
    while t < tfinal:

        epsilonP1=pop[5]/Cg
        epsilonN1=-1.5*Vd-pop[5]/Cg

        epsilonP2=pop[2]/Cg
        epsilonN2=-1.5*Vd-pop[2]/Cg

        # Compute rates at which particles hop between sites
        c[0] = gamma*1./(np.exp(beta*(epsilonN1))+1)                           # N1 -> S1
        c[1] = gamma*(1-1./(np.exp(beta*(epsilonN1))+1))                       # S1 -> N1

        c[2] = gamma*1./(np.exp(beta*(abs(epsilonN1-epsilonP1)))-1.+1e-6)      # N1 -> P1
        c[3] = gamma*(1+1./(np.exp(beta*(abs(epsilonN1-epsilonP1)))-1.+1e-6))  # P1 -> N1

        c[4] = gamma*1./(np.exp(beta*(epsilonP1-Vd))+1.)                       # P1 -> D1
        c[5] = gamma*(1-1./(np.exp(beta*(epsilonP1-Vd))+1.))                   # D1 -> P1

        c[6] = gamma*1./(np.exp(beta*(epsilonP1+(pop[2])/Cg))+1.)              # P1 -> G1
        c[7] = gamma*(1-1./(np.exp(beta*(epsilonP1+pop[2]/Cg))+1.))            # G1 -> P1

        c[8] = gamma*1./(np.exp(beta*(epsilonN1+(pop[2])/Cg))+1.)              # N1 -> G1
        c[9] = gamma*(1-1./(np.exp(beta*(epsilonN1+pop[2]/Cg))+1.))            # G1 -> N1

        # Rates
        c[10] = gamma*1./(np.exp(beta*(epsilonN2))+1)                          # N2 -> S2
        c[11] = gamma*(1-1./(np.exp(beta*(epsilonN2))+1))                      # S2 -> N2

        c[12] = gamma*1./(np.exp(beta*(abs(epsilonN2-epsilonP2)))-1.+1e-6)     # N2 -> P2
        c[13] = gamma*(1+1./(np.exp(beta*(abs(epsilonN2-epsilonP2)))-1.+1e-6)) # P2 -> N2

        c[14] = gamma*1./(np.exp(beta*(epsilonP2-Vd))+1.)                      # P2 -> D2
        c[15] = gamma*(1-1./(np.exp(beta*(epsilonP2-Vd))+1.))                  # D2 -> P2

        c[16] = gamma*1./(np.exp(beta*(epsilonP2+(pop[5])/Cg))+1.)             # P2 -> G2
        c[17] = gamma*(1-1./(np.exp(beta*(epsilonP2+pop[5]/Cg))+1.))           # G2 -> P2

        c[18] = gamma*1./(np.exp(beta*(epsilonN2+(pop[5])/Cg))+1.)             # N2 -> G2
        c[19] = gamma*(1-1./(np.exp(beta*(epsilonN2+pop[5]/Cg))+1.))           # G2 -> N2


        # Get the probability of each possible transition
        a=np.zeros((20,1))
        a[0] = c[0]*pop[0];
        a[1] = c[1]*(1-pop[0]);
        a[2] = c[2]*pop[0]*(1-pop[1]);
        a[3] = c[3]*(1-pop[0])*pop[1];
        a[4] = c[4]*pop[1];
        a[5] = c[5]*(1-pop[1]);
        a[6] = c[6]*pop[1] if pop[2] < max_gate_val else 0;
        a[7] = c[7]*(1-pop[1]) if pop[2] > min_gate_val else 0;
        a[8] = c[8]*pop[0] if pop[2] < max_gate_val else 0;
        a[9] = c[9]*(1-pop[0]) if pop[2] > min_gate_val else 0;
        a[10] = c[10]*pop[3];
        a[11] = c[11]*(1-pop[3]);
        a[12] = c[12]*pop[3]*(1-pop[4]);
        a[13] = c[13]*(1-pop[3])*pop[4];
        a[14] = c[14]*pop[4];
        a[15] = c[15]*(1-pop[4]);
        a[16] = c[16]*pop[4] if pop[5] < max_gate_val else 0;
        a[17] = c[17]*(1-pop[4]) if pop[5] > min_gate_val else 0;
        a[18] = c[18]*pop[3] if pop[5] < max_gate_val else 0;
        a[19] = c[19]*(1-pop[3]) if pop[5] > min_gate_val else 0;

        asum = np.sum(a)

        # Get the time step size
        tau = np.log(1/np.random.rand())/asum
        t = t + tau

        # Determine which reaction to do
        atmp = np.cumsum(a/asum)
        jtmp=np.nonzero(np.random.rand()<atmp)
        j= np.min(jtmp[0])

        # Update Population
        pop = pop + V[j]
        if j%2==0: entropy = entropy + np.log(c[j]/c[j+1])
        else: entropy = entropy + np.log(c[j]/c[j-1])

        # Save data
        data[stat,0]=t
        data[stat,1]=pop[0]
        data[stat,2]=pop[1]
        data[stat,3]=pop[2]
        data[stat,4]=pop[3]
        data[stat,5]=pop[4]
        data[stat,6]=pop[5]
        data[stat,7]=entropy

        stat += 1
        jumps=jumps+1
        if jumps%10000==0:
            print(t/tfinal*100, '%')
            if plot:
                ax.clear()
                ax.plot(data[:stat, 0], -data[:stat, 3]/(Cg*Vd), 'r-')
                ax.plot(data[:stat, 0], -data[:stat, 6]/(Cg*Vd), 'b-')
                ax1.clear()
                minval = min(np.min(data[:stat,3]), np.min(data[:stat, 6]))
                maxval = max(np.max(data[:stat,3]), np.max(data[:stat, 6]))
                bins = np.arange(minval, maxval+1)-0.5
                ax1.hist(data[:stat, 3], color='r', bins=bins, alpha=0.5, density=True)
                ax1.hist(data[:stat, 6], color='b', bins=bins, alpha=0.5, density=True)
                ax1.set_yscale('log')
                plt.pause(0.01)

    return data[:stat]


"""
This script processes data from trajectories, creating
Figure 1 (c) from the paper:
Stochastic thermodynamic bounds on logical circuit operation
"""
import numpy as np
from sys import argv
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('plasma')
greycmap = plt.get_cmap('Greys')
import matplotlib as mpl

# Plot Settings ###################################################
# Figure Size
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
mpl.rc('legend', labelspacing = 0.3)
mpl.rc('legend', handletextpad = 0.5)
mpl.rc('legend', borderaxespad = 0.5)
mpl.rc('legend', columnspacing = 0.)
mpl.rc('legend', handlelength = 1.)
# Plot Settings ###################################################

# Create the figure
f, ax1 = plt.subplots(1, 1, figsize=(4, 2.5))

##############################################################################
# Forward, alpha=0.05
##############################################################################
Cg = 10.
fname = 'data/first_passage_time_distribution_forward_Cg{}.npz'.format(Cg)
data = np.load(fname)
Vd = data['Vd']
S = data['S']
dS = [None for _ in range(len(Vd))]
tau_avg = [None for _ in range(len(Vd))]
dtau2_avg = [None for _ in range(len(Vd))]
t = data['t']

for Vdi in range(len(Vd)):

    # Show fpt probability distribution
    dS[Vdi] = -np.gradient(S[Vdi], t)
    if Vd[Vdi] > 18:
        keep_inds = np.where(t > 1e8)[0]
    else:
        keep_inds = np.where(t >= 0)[0]

    # Calculate mean first passage time
    sig_inds = np.where(dS[Vdi][keep_inds]/max(dS[Vdi][keep_inds]) > 1e-8)[0]
    y = (t[keep_inds]*dS[Vdi][keep_inds])[sig_inds]
    x = t[keep_inds][sig_inds]
    tau_avg[Vdi] = np.trapz(y, x=x)

    # Calculate std first passage time
    y = (dS[Vdi]*(t - tau_avg[Vdi])**2.)[keep_inds][sig_inds]
    x = t[keep_inds][sig_inds]
    dtau2_avg[Vdi] = np.trapz(y, x=x)

dS_flow = data['dS_flow']
S_flow= [None for _ in range(len(Vd))]
t = data['t']

for Vdi in range(len(Vd)):
    S_flow[Vdi] = [np.trapz(dS_flow[Vdi][:ti], t[:ti]) for ti in range(1, len(t))]

# Get steady state accuracies
x = Vd

ms = 5.
y = [tau_avg[_]**2/(S_flow[_][-1] * dtau2_avg[_]) for _ in range(len(Vd))]
ax1.semilogy(x, y, '-o', color=cmap(0.05/0.3),
             markeredgecolor = cmap(0.05/0.3),
             markerfacecolor = (1, 1, 1, 1),
             markersize=ms)
ax1.semilogy(-10, -10, '^', color=cmap(0.05/0.3),
             markeredgecolor = cmap(0.05/0.3),
             markerfacecolor = (1, 1, 1, 1),
             markersize=ms, label='Discharging')
ax1.semilogy(-10, -10, 'o', color=cmap(0.05/0.3),
             markeredgecolor = cmap(0.05/0.3),
             markerfacecolor = (1, 1, 1, 1),
             markersize=ms, label='Charging')
y = [1./2. for _ in range(len(Vd))]
ax1.fill_between([0, 30], [1./2., 1./2.], [100, 100], hatch='\\\\', color='k', alpha=0.2)


##############################################################################
# reverse, alpha=0.05
##############################################################################
Cg = 10.
fname = 'data/first_passage_time_distribution_reverse_Cg{}.npz'.format(Cg)
data = np.load(fname)
Vd = data['Vd']
S = data['S']
dS = [None for _ in range(len(Vd))]
tau_avg = [None for _ in range(len(Vd))]
dtau2_avg = [None for _ in range(len(Vd))]
t = data['t']

for Vdi in range(len(Vd)):

    # Show fpt probability distribution
    dS[Vdi] = -np.gradient(S[Vdi], t)
    if Vd[Vdi] > 100:
        keep_inds = np.where(t > 1e8)[0]
    else:
        keep_inds = np.where(t >= 0)[0]

    # Calculate mean first passage time
    sig_inds = np.where(dS[Vdi][keep_inds]/max(dS[Vdi][keep_inds]) > 1e-8)[0]
    y = (t[keep_inds]*dS[Vdi][keep_inds])[sig_inds]
    x = t[keep_inds][sig_inds]
    tau_avg[Vdi] = np.trapz(y, x=x)

    # Calculate std first passage time
    y = (dS[Vdi]*(t - tau_avg[Vdi])**2.)[keep_inds][sig_inds]
    x = t[keep_inds][sig_inds]
    dtau2_avg[Vdi] = np.trapz(y, x=x)

dS_flow = data['dS_flow']
S_flow= [None for _ in range(len(Vd))]
t = data['t']

for Vdi in range(len(Vd)):
    S_flow[Vdi] = [np.trapz(dS_flow[Vdi][:ti], t[:ti]) for ti in range(1, len(t))]

# Get steady state accuracies
x = Vd

# Plot results
y = [tau_avg[_]**2/(S_flow[_][-1] * dtau2_avg[_]) for _ in range(len(Vd))]
ax1.semilogy(x, y, '-^', color=cmap(0.05/0.3),
             markeredgecolor = cmap(0.05/0.3),
             markerfacecolor = (1, 1, 1,1),
             markersize=ms)

# Do some formatting of the plot
plt.tight_layout()
ax1.set_xlim(0, 30)
ax1.set_ylim(1e-4, 1e0)
ax1.set_xlabel(r'$V_\mathrm{d} \ / \ V_\mathrm{T}$', fontsize=11)
ax1.set_ylabel(r'$ \chi_\mathrm{p}^2 \ / \ \beta \langle Q\rangle$', fontsize=11)
plt.legend(loc='lower left', fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig(f'./figs/fig1c.png', dpi=500)

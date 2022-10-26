import numpy as np
from sys import argv
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
cmap2 = plt.get_cmap('plasma')
greycmap = plt.get_cmap('Greys')
import matplotlib as mpl

# Plot Settings ###################################################
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
gridon = False
# Plot Settings ###################################################

# Create the figure
f, ax1 = plt.subplots(1, 1, figsize=(4, 2.5))
Cg = 10.
alpha_vec = [0.0, 0.05, 0.1, 0.2]

# Figure out accuracy in steady state
acc_data = np.load('./data/steady_state_loading.npz')
accuracy = acc_data['accuracy']
acc_alpha = acc_data['alpha']
acc_Vd = acc_data['Vd']
acc_Cg = acc_data['Cg']
Cgind = np.where(acc_Cg == Cg)[0]

# Loop over accuracy thresholds
for alphai, alpha in enumerate(alpha_vec):

    if alpha == 0.05:
        fname = 'data/first_passage_time_distribution_forward_Cg{}.npz'.format(Cg)
    else:
        fname = 'data/first_passage_time_distribution_forward_alpha{}_Cg{}.npz'.format(alpha, Cg)
    data = np.load(fname, allow_pickle=True)
    Vd = data['Vd']
    S = data['S']
    dS = [None for _ in range(len(Vd))]
    tau_avg = [None for _ in range(len(Vd))]
    dtau2_avg = [None for _ in range(len(Vd))]
    t = data['t']

    for Vdi in range(len(Vd)):

        # Show fpt probability distribution
        dS[Vdi] = -np.gradient(S[Vdi], t)
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
    alphaind = np.argmin(np.abs(acc_alpha-alpha))
    x = accuracy[Cgind, :, alphaind]
    x = 1.-x

    ms = 5.
    y = [tau_avg[_]**2/(S_flow[_][-1] * dtau2_avg[_]) for _ in range(len(Vd))]
    ax1.semilogy(x[0,:], y, '-o', color=cmap(alpha/0.3),
                 markeredgecolor = cmap(alpha/0.3),
                 markerfacecolor = (1, 1, 1, 1),
                 markersize=ms,
                 linewidth=1)
y = [1./2. for _ in range(len(Vd))]
ax1.fill_between([0, 30], [1./2., 1./2.], [100, 100], hatch='\\\\', color='k', alpha=0.2)

# Do some formatting of the plot
plt.tight_layout()
ax1.set_xscale('log')
ax1.set_xlim(1e0, 1e-15)
ax1.set_ylim(1e-4, 1e0)
ax1.set_xlabel(r'$1-P_\mathrm{correct}$', fontsize=11)
ax1.set_ylabel(r'$ \chi_\mathrm{p}^2 \ / \ \beta \langle Q\rangle$', fontsize=11)
plt.tight_layout()
plt.savefig(f'./figs/fig1d.png', dpi=500)

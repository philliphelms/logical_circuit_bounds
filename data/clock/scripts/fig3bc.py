import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
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
mpl.rc('legend', labelspacing = 0.0)
mpl.rc('legend', handletextpad = 0.)
mpl.rc('legend', borderaxespad = 0.5)
mpl.rc('legend', columnspacing = 0.)
mpl.rc('legend', handlelength = 1.)
gridon = False
ftype = 'png'
dpi = 500
ms = 5.0
# Plot Settings ###################################################

# System parameters
Cg = 10.
Vd_vec = [1.0, 2.1, 3.0, 4.1, 5.0, 6.1, 7.0, 8.1, 9.0,
          10.1, 11.0, 12.1, 13.0, 14.1, 15.0, 16.1, 17.0, 18.1, 19.0, 20.1]
beta = 1.
q = 1.
V_T = 1.
max_cmap = 1e10

# Loop through results (collect & process)
tau_clock_avg = []
tau_clock_std = []
tau_clock2_avg = []
tau_clock2_std = []
dQ_avg = []
dQ_std = []
Q_avg = []
Q_std = []
for Vdi in range(len(Vd_vec)):
    Vd = Vd_vec[Vdi]
    fname = f'./data/tur_data_Vd{Vd}_Cg{Cg}.npz'
    data = np.load(fname, allow_pickle=True)

    dQi = data['dQ_avg']
    dQ_avg.append(np.nanmean(dQi))
    dQ_std.append(np.nanstd(dQi))

    Qi = data['Q_avg']
    Q_avg.append(np.nanmean(Qi))
    Q_std.append(np.nanstd(Qi))

    tau_i = data['clock_times_avg']
    tau_clock_avg.append(np.nanmean(tau_i))
    tau_clock_std.append(np.nanstd(tau_i))

    tau2_i = data['clock_times_std']
    tau_clock2_avg.append(np.nanmean(tau2_i))
    tau_clock2_std.append(np.nanstd(tau2_i))

tau_clock_avg = np.array(tau_clock_avg)
tau_clock_std = np.array(tau_clock_std)
tau_clock2_avg = np.array(tau_clock2_avg)
tau_clock2_std = np.array(tau_clock2_std)
Q_avg = np.array(Q_avg)
Q_std = np.array(Q_std)
dQ_avg = np.array(dQ_avg)
dQ_std = np.array(dQ_std)

tau_clock_avg_cont = []
tau_clock_std_cont = []
tau_clock2_avg_cont = []
tau_clock2_std_cont = []
dQ_avg_cont = []
dQ_std_cont = []
Q_avg_cont = []
Q_std_cont = []
for Vdi in range(len(Vd_vec)):
    Vd = Vd_vec[Vdi]
    fname = f'./data/controlled_tur_data_Vd{Vd}_Cg{Cg}.npz'
    data = np.load(fname, allow_pickle=True)

    dQi = data['dQ_avg']
    dQ_avg_cont.append(np.nanmean(dQi))
    dQ_std_cont.append(np.nanstd(dQi))

    Qi = data['Q_avg']
    Q_avg_cont.append(np.nanmean(Qi))
    Q_std_cont.append(np.nanstd(Qi))

    tau_i = data['clock_times_avg']
    tau_clock_avg_cont.append(np.nanmean(tau_i))
    tau_clock_std_cont.append(np.nanstd(tau_i))

    tau2_i = data['clock_times_std']
    tau_clock2_avg_cont.append(np.nanmean(tau2_i))
    tau_clock2_std_cont.append(np.nanstd(tau2_i))

tau_clock_avg_cont = np.array(tau_clock_avg_cont)
tau_clock_std_cont = np.array(tau_clock_std_cont)
tau_clock2_avg_cont = np.array(tau_clock2_avg_cont)
tau_clock2_std_cont = np.array(tau_clock2_std_cont)
Q_avg_cont = np.array(Q_avg_cont)
Q_std_cont = np.array(Q_std_cont)
dQ_avg_cont = np.array(dQ_avg_cont)
dQ_std_cont = np.array(dQ_std_cont)

# Create the figure
f, ax1 = plt.subplots(1, 1, figsize=(4, 2.5), sharex=True)
f, ax2 = plt.subplots(1, 1, figsize=(4, 2.5), sharex=True)

# Data for thresholded calculations
# Plot actual results
ax2.semilogx(tau_clock_avg_cont, (tau_clock_avg_cont**2/tau_clock2_avg_cont**2)**(1/2), 's-',
             color=greycmap(0.2), markeredgecolor=cmap(0.7), markerfacecolor='w', markersize=ms)

# Plot Lower bound
y = np.sqrt( (beta * tau_clock_avg_cont) / (dQ_avg_cont * tau_clock2_avg_cont**2) )
ax2.semilogx(tau_clock_avg_cont, y, '^-', color=greycmap(0.2),
             markeredgecolor='k', markerfacecolor='w', markersize=ms)
ax2.fill_between(tau_clock_avg_cont, 0.*tau_clock_avg_cont, y2=y, color='k', hatch='\\\\', alpha=0.2)

# Plot Upper Bound
y = (Q_avg_cont/2)**(1/2)
ax2.semilogx(tau_clock_avg_cont, y, 'o-', color=greycmap(0.2),
             markeredgecolor='k', markerfacecolor='w', markersize=ms)
ax2.fill_between(tau_clock_avg_cont, 1e20*np.ones(tau_clock_avg_cont.shape), y2=y, color='k', hatch='\\\\', alpha=0.2)

ax2.semilogx([0, 1e100], [1, 1], 'r:', linewidth=1.5)

# Data for first set of calculations
# Plot actual results
ax1.semilogx(tau_clock_avg, (tau_clock_avg**2/tau_clock2_avg**2)**(1/2), 's-',
             color=greycmap(0.2), markeredgecolor=cmap(0.7), markerfacecolor='w', markersize=ms)

# Plot Lower bound
y = np.sqrt( (beta * tau_clock_avg) / (dQ_avg * tau_clock2_avg**2) )
ax1.semilogx(tau_clock_avg, y, '^-', color=greycmap(0.2),
             markeredgecolor='k', markerfacecolor='w', markersize=ms)
ax1.fill_between(tau_clock_avg, 0.*tau_clock_avg, y2=y, color='k', hatch='\\\\', alpha=0.2)

# Plot Upper Bound
y = (Q_avg/2)**(1/2)
ax1.semilogx(tau_clock_avg, y, 'o-', color=greycmap(0.2),
             markeredgecolor='k', markerfacecolor='w', markersize=ms)
ax1.fill_between(tau_clock_avg, 1e20*np.ones(tau_clock_avg.shape), y2=y, color='k', hatch='\\\\', alpha=0.2)

ax1.semilogx([0, 1e100], [1, 1], 'r:', linewidth=1.5)

ax1.semilogx([0, 1e100], [1, 1], 'r:', linewidth=1.5)

# Format plot
ax1.set_ylabel(r'$\chi_\mathrm{mem}$', fontsize=11)
ax2.set_ylabel(r'$\chi_\mathrm{mem}$', fontsize=11)
ax1.set_xlabel(r'$\langle \tau_\mathrm{c}\rangle \ / \ \beta\hbar$', fontsize=11)
ax2.set_xlabel(r'$\langle \tau_\mathrm{c}\rangle \ / \ \beta\hbar$', fontsize=11)
ax1.set_yscale('log')
ax1.set_ylim(1e-2, 100)
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 100)
ax1.set_xlim(min(tau_clock_avg), max(tau_clock_avg))
ax2.set_xlim(min(tau_clock_avg_cont), max(tau_clock_avg_cont))
plt.sca(ax1)
_ax1 = ax1.twiny()
_ax1.set_xscale('log')
_ax1.set_xlim(ax1.get_xlim())
_ax1.set_xticks([tau_clock_avg[_] for _ in [0, 4, 8, 12, 16]])
_ax1.set_xticklabels([f'{Vd_vec[_]:0.0f}' for _ in [0, 4, 8, 12, 16]])
_ax1.set_xlabel(r'$V_\mathrm{d} \ / \ V_\mathrm{T}$')
plt.tight_layout()
plt.pause(0.01)
plt.savefig('./figs/fig3b.'+ftype, dpi=500)
plt.sca(ax2)
_ax2 = ax2.twiny()
_ax2.set_xscale('log')
_ax2.set_xlim(ax2.get_xlim())
_ax2.set_xticks([tau_clock_avg_cont[_] for _ in [0, 4, 8, 12, 16]])
_ax2.set_xticklabels([f'{Vd_vec[_]:0.0f}' for _ in [0, 4, 8, 12, 16]])
_ax2.set_xlabel(r'$V_\mathrm{d} \ / \ V_\mathrm{T}$')
plt.tight_layout()
plt.pause(0.01)
plt.savefig('./figs/fig3c.'+ftype, dpi=500)


#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = plt.get_cmap('viridis')

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
# Plot Settings ###################################################

# Parameters
Cg = 10.
Vd_vec = [1.0, 1.25, 1.5, 1.75,
          2.1, 2.35, 2.6, 2.85,
          3.0, 3.25, 3.5, 3.75,
          4.1, 4.35, 4.6, 4.85,
          5.0]
beta = 1

# Load the data
fnames = [f'./data/fpt_Cg{Cg}_Vd{Vd}_dt0.1_steady_initial_state.npz' for Vd in Vd_vec]
dat = [np.load(fname) for fname in fnames]
S = [dati['S'] for dati in dat]
t = [dati['t'] for dati in dat]
dS_flow_trans = [dati['dS_flow'] for dati in dat]
dS_flow_tot = [np.trapz(dS_flow_trans[_], x=t[_]) for _ in range(len(Vd_vec))]

# Calculate the first passage times
fS = [None for _ in range(len(Vd_vec))]
filter_strength = [1, 1, 1, 1,
                   1, 1, 1, 1,
                   1, 1, 1, 1,
                   5, 5, 5, 5,
                   5, 30, 30, 30,
                   30, 30, 30, 30]
for Vdi in range(len(Vd_vec)):
    S[Vdi] = S[Vdi][::filter_strength[Vdi]]
    t[Vdi] = t[Vdi][::filter_strength[Vdi]]
    fS[Vdi] = -np.gradient(S[Vdi], t[Vdi])
    dS_flow_trans[Vdi] = dS_flow_trans[Vdi][::filter_strength[Vdi]]

# Calculate the heat dissipation over first passage trajectory
# Create a plot to help with fitting exponential decay
fit_inds = [(500, 1100), (750, 1200), (750, 1400), (800, 1500),
            (1200, 1600), (1300, 1800), (1400, 1900), (1800, 2200),
            (2000, 2300), (2000, 2350), (2100, -2), (2200, -2),
            (400, -2), (400, -2), (400, -2), (400, -2),
            (400, -2), (-10, -2), (-100, -2), (-100, -2),
            (-100, -2), (-100, -2), (-100, -2), (-100, -2)]

# Fit results to an exponential
k = [None for _ in range(len(Vd_vec))]
b = [None for _ in range(len(Vd_vec))]
for Vdi in range(len(Vd_vec)):
    if fit_inds[Vdi] is not None:
        fit_val1 = fS[Vdi][fit_inds[Vdi][0]]
        fit_val2 = fS[Vdi][fit_inds[Vdi][1]]
        tref1 = t[Vdi][fit_inds[Vdi][0]]
        tref2 = t[Vdi][fit_inds[Vdi][1]]
        k[Vdi] = (np.log(fit_val1) - np.log(fit_val2))/(tref1 - tref2)
        b[Vdi] = np.log(fit_val2)-k[Vdi]*tref2

# Get the exponential Fits
logt = [None for _ in range(len(Vd_vec))]
fS_fit = [None for _ in range(len(Vd_vec))]
for Vdi in range(len(Vd_vec)):
    if k[Vdi] is not None:
        logt[Vdi] = np.logspace(np.log10(t[Vdi][fit_inds[Vdi][1]]), 14)
        fS_fit[Vdi] = np.exp(k[Vdi]*logt[Vdi]+b[Vdi])

# Calculate the mean first passage time
tau_avg = [None for _ in range(len(Vd_vec))]
for Vdi in range(len(Vd_vec)):
    y = np.append(fS[Vdi], fS_fit[Vdi])
    x = np.append(t[Vdi], logt[Vdi])
    y *= x
    tau_avg[Vdi] = np.trapz(y, x=x)

# Calculate the heat dissipation
ss_fnames = [f'data/steady_state_Cg{Cg}_Vd{Vdi}_dt0.1_steady_initial_state.npz' for Vdi in Vd_vec]
ss_data = [np.load(ss_fname) for ss_fname in ss_fnames]
ss_t = [datai['t'] for datai in ss_data]
dS_flow = [datai['dS_flow'] for datai in ss_data]

# Get the steady state occupations
ss_occ_well = [None for _ in range(len(Vd_vec))]
ss_occ_tran = [None for _ in range(len(Vd_vec))]
for Vdi in range(len(Vd_vec)):
    gate_vals = -ss_data[Vdi]['gate1_vals']/(Cg*Vd_vec[Vdi])
    ind_tran = np.argmin(np.abs(gate_vals - 0.5))
    ind_well = np.argmin(np.abs(gate_vals - 0.6))
    ss_occ_well[Vdi] = np.sum(ss_data[Vdi]['rhoG1'][-1, :ind_well])
    ss_occ_tran[Vdi] = ss_data[Vdi]['rhoG1'][-1, ind_tran]

# Get relaxation time data (alpha=0.4, forward)
tau_p_Vd = np.arange(1, 31)
tau_p_mfpt = np.array([40.77799570446843,
                    112.4943951589994,
                    224.09906676500162,
                    433.0939460310747,
                    829.3880518887512,
                    1571.595628086609,
                    2560.7363964732526,
                    4363.722109862616,
                    6996.136972962748,
                    11657.049325567254,
                    19365.204965017347,
                    33794.77970275343,
                    58792.988309949025,
                    101997.49545305599,
                    191636.71972264923,
                    292225.71868274076,
                    545759.0391610757,
                    868916.2752292857,
                    1550351.506527732,
                    2750278.4807169787,
                    4395257.091800109,
                    8026944.459624171,
                    12438776.483910253,
                    21942974.443792053,
                    36279069.89827788,
                    61847664.76199401,
                    108400497.71784827,
                    189325027.85732526,
                    296155566.9316994,
                    517099278.4051299])
tau_p_mfpt = np.interp(Vd_vec, tau_p_Vd, tau_p_mfpt)

# Create the figure
f = plt.figure(figsize=(4, 2.5))
ax = f.add_subplot(111)

# Populate the figure
# memory time data
ax.loglog(tau_p_mfpt, tau_avg/tau_p_mfpt, 's',
          markeredgecolor=cmap(0.3),
          markerfacecolor=(0, 0, 0, 0),
          label='Observed')

# Add Transition State Theory Estimate
ax.loglog(tau_p_mfpt, np.array(ss_occ_well)/np.array(ss_occ_tran),
          '^',
          markeredgecolor=cmap(0.7),
          markerfacecolor=(0, 0, 0, 0),
          label='Noneq. TST')

# Esposito's limit
for Vdi in range(len(Vd_vec)):
    if Vdi == 0:
        ax.loglog(tau_p_mfpt[Vdi], 1./(beta*dS_flow[Vdi][-1]*tau_p_mfpt[Vdi]), 'o',
                  markeredgecolor='k',
                  markerfacecolor='w',
                  label='Dissipation Bound')
    else:
        ax.loglog(tau_p_mfpt[Vdi], 1./(beta*dS_flow[Vdi][-1]*tau_p_mfpt[Vdi]), 'o',
                  markeredgecolor='k',
                  markerfacecolor='w')

# Espositos limit
y2 = [1./(beta*dS_flow[Vdi][-1]*tau_p_mfpt[Vdi]) for Vdi in range(len(Vd_vec))]
ax.fill_between(tau_p_mfpt, 0*tau_p_mfpt, y2=y2, hatch='//', color='k', alpha=0.2)

# Lasting memory line
ax.plot(tau_p_mfpt, np.ones(len(tau_p_mfpt)), 'r:')

ax.set_xlabel(r'$\langle \tau_\mathrm{p} \rangle \ / \ \beta \hbar$')
ax.set_ylabel(r'$\langle \tau_\mathrm{err} \rangle \ / \ \langle \tau_\mathrm{p} \rangle$')
ax.set_xscale('linear')
ax.set_xlim(tau_p_mfpt[0], tau_p_mfpt[-1])
ax.set_xticks([250, 500, 750])
plt.legend(frameon=False)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([tau_p_mfpt[_] for _ in [0, 4, 8, 12, 16]])
ax2.set_xticklabels([f'{Vd_vec[_]}' for _ in [0, 4, 8, 12, 16]])
ax2.set_xlabel(r'$V_\mathrm{d} \ / \ V_\mathrm{T}$')
plt.tight_layout()

plt.savefig('./figs/fig2b.'+ftype, dpi=dpi)

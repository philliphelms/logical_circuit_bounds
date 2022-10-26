import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap('seismic')
cmap = plt.get_cmap('PRGn')
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
ftype = 'pdf'
dpi = 500
# Plot Settings ###################################################

# Load the data
Cg = 10.
Vd_vec = np.array([1., 1.5, 2.1, 3., 4.1, 5., 6.1, 7., 8.1, 9., 10.1,
                   11., 12.1, 13., 14.1, 15., 16.1, 17., 18.1, 19., 20.1])
dat = [np.load(f'./data/acf_data_Vd{Vd}_Cg{Cg}.npz') for Vd in Vd_vec]
acf = [dati['signal_acf'] for dati in dat]
t = [dati['tsample'] for dati in dat]

# Use tau_clock_avg as reference
tau_c_ref_Vd = np.array([1.0,
                        2.1,
                        3.0,
                        4.1,
                        5.0,
                        6.1,
                        7.0,
                        8.1,
                        9.0,
                        10.1,
                        11.0,
                        12.1,
                        13.0,
                        14.1,
                        15.0,
                        16.1,
                        17.0,
                        18.1,
                        19.0,
                        20.1])
tau_c_avg = np.array([113.67686180246503,
                    572.411127181981,
                    1874.359243465036,
                    5554.273119338497,
                    13254.972004761561,
                    36956.66915031168,
                    85789.99010193403,
                    236853.8339793844,
                    539382.308474078,
                    1456732.9243685189,
                    3245804.953823093,
                    8717218.573237555,
                    19139326.638096366,
                    51167768.79845609,
                    109191487.12000673,
                    282487569.8196812,
                    614044333.7228686,
                    1616339029.1985044,
                    3405036960.509863,
                    8723036692.022135,])

# Do interpolation to get values
tau_c_avg = np.interp(Vd_vec, tau_c_ref_Vd, tau_c_avg)

# Create a contour plot
f, ax = plt.subplots(1, 1, figsize=(4, 2.5))

# Interpolate to get grid of times
tmax = 5
t_grid = np.linspace(0, tmax, 10000)
acf_mat = [np.interp(t_grid, t[_]/tau_c_avg[_], np.nanmean(acf[_], axis=0)) for _ in range(len(Vd_vec))]
acf_mat = np.array(acf_mat)
for Vdi in range(len(Vd_vec)):
    acf_mat[Vdi, :] -= acf_mat[Vdi, -1]
    acf_mat[Vdi, :] /= acf_mat[Vdi, 0]
t_grid_mat, Vd_mat = np.meshgrid(t_grid, Vd_vec)
con = ax.contourf(Vd_mat, t_grid_mat, acf_mat, 20, cmap=cmap, vmin=-0.5, vmax=0.5)
cbar = f.colorbar(con, ticks=[-1, -0.5, 0, 0.5, 1])

ax.set_xlim(1, 20)
ax.set_xlabel(r'$V_\mathrm{d} \ / \ V_\mathrm{T}$', fontsize=11)
ax.set_ylabel(r'$t \ / \ \langle \tau_\mathrm{c} \rangle$', fontsize=11)
ax.text(22, -0.5, r'$C_{V_iV_i}^*$', fontsize=11)
plt.tight_layout()

plt.savefig(f'./figs/fig3a.'+ftype, dpi=dpi)

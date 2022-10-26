import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sys import argv
cmap = plt.get_cmap('Blues')

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

# Load all of the data
Cg = 10.
dt = 0.1
Vd_vec = np.array([1.0, 1.5, 2.1, 2.6, 3.0, 3.5, 4.1, 4.6, 5.0, 5.5,
                   6.1, 6.6, 7.0, 7.5, 8.1])
fnames = [f'./data/steady_state_Cg{Cg}_Vd{Vdi}_dt{dt}_steady_initial_state.npz' for Vdi in Vd_vec]
data = [np.load(fname) for fname in fnames]
t = [datai['t'] for datai in data]
rhoG1 = [datai['rhoG1'][-1, :] for datai in data]
rhoG2 = [datai['rhoG2'][-1, :] for datai in data]
gate1_vals = [datai['gate1_vals'] for datai in data]
gate2_vals = [datai['gate2_vals'] for datai in data]

# Interpolate onto the finest grid
gate1_vals = [-gate1_vals[_]/(Cg*Vd_vec[_]) for _ in range(len(data))]
rhoG1_int = [np.interp(gate1_vals[-1][::-1], gate1_vals[_][::-1], rhoG1[_][::-1])[::-1] for _ in range(len(data))]
rhoG2_int = [np.interp(gate2_vals[-1][::-1], gate2_vals[_][::-1], rhoG2[_][::-1])[::-1] for _ in range(len(data))]
rhoG1_int = [(rhoG1_int[_]+rhoG2_int[_])/2 for _ in range(len(data))]
rhoG1_int = np.array(rhoG1_int)

# Create a the contour plot
f, ax = plt.subplots(1, 1, figsize=(4, 2.5))
gate_vals_mat, Vd_mat = np.meshgrid(gate1_vals[-1], Vd_vec)
con = ax.contourf(Vd_mat, gate_vals_mat, rhoG1_int, 10, cmap=cmap)

# Add data points
for Vdi in range(len(Vd_vec)):
    if Vdi > 2:
        max_ind1 = np.argsort(rhoG1[Vdi][:int(len(rhoG1[Vdi])/2)])[-1]
        max_ind2 = np.argsort(rhoG1[Vdi][int(len(rhoG1[Vdi])/2):])[-1] + int(len(rhoG1[Vdi])/2)
        ax.plot(Vd_vec[Vdi], gate1_vals[Vdi][max_ind1], 'kx', markersize=4)
        ax.plot(Vd_vec[Vdi], gate1_vals[Vdi][max_ind2], 'kx', markersize=4)
    else:
        max_ind1 = np.argsort(rhoG1[Vdi])[-1]
        ax.plot(Vd_vec[Vdi], gate1_vals[Vdi][max_ind1], 'kx', markersize=4)


# Format the plot
ax.set_xlabel(r'$V_{\mathrm{d}} \ / \ V_\mathrm{T}$')
ax.set_ylabel(r'$V_{\mathrm{out}} \ / \ V_\mathrm{d}$')
ax.set_yticks([0, 0.5, 1.0])

plt.tight_layout()
plt.savefig(f'./figs/fig2a.'+ftype, dpi=dpi)

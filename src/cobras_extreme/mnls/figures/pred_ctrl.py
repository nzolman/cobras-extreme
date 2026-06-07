import os
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from cobras_extreme import _mnls_data_dir

from cobras_extreme.mnls.plot_utils import pred_3D_test

from cobras_extreme.mnls.waves import MNLS1D
from cobras_extreme.mnls.shift_utils import get_snap_shift
from cobras_extreme.plotting import set_defaults


colors = set_defaults()



# Setup solver with default params
N = 2**10
L_domain = 256 * jnp.pi
Tf = 200
dt = 0.025
frac = 2/3
solver = MNLS1D(N, L_domain, dt, M_contour=128, dealias_frac = frac)
x = solver.x

# grab shifter
shift_snap_v = get_snap_shift(solver)


# Grab Long data
long_data_path = os.path.join(_mnls_data_dir, 
                              'forward_Tf_4000', 
                              'mnls_fwd_long-4000.pkl'
                              )
import pickle
with open(long_data_path, 'rb') as f:
    long_data = pickle.load(f)
    
long_data.shape
aligned_long = shift_snap_v(long_data, jnp.arange(4000), 0.5, 1, 0)




n_modes = 8
svm_dir = os.path.join(_mnls_data_dir, 'svm')


# grab SVM datasets
param_dict = jnp.load(f'{svm_dir}/svm_r-{n_modes}-svm_params.npy', allow_pickle=True).item()
ctrl_dict = jnp.load(f'{svm_dir}/svm_r-{n_modes}-ctrl.npy', allow_pickle=True).item()
pred_dict = jnp.load(f'{svm_dir}/svm_r-{n_modes}-decs.npy', allow_pickle=True).item()

all_pos_quants = pred_dict['all_pos_quants']

all_snaps = ctrl_dict['snaps']
all_ctrl = ctrl_dict['ctrls']
all_snaps_ctrl_aligned = shift_snap_v(all_snaps, jnp.arange(len(all_snaps)), 0.5, 1, 0)
all_ctrl_aligned = shift_snap_v(all_ctrl, jnp.arange(len(all_ctrl)), 0.5, 1, 0)



# Zoom out for SI
fig = plt.figure(figsize=(3,3))
ax, fig = pred_3D_test(solver, aligned_long, 
                       fig=fig, 
             preds = all_pos_quants,
             z_downsample = 1, 
             cobras_kws = {'vmin':0, 'vmax':1, 's': 10},
             max_t = 401,
             skip =50,
             min_x = 0*jnp.pi, 
             max_x = 128 * jnp.pi,
            extr_kws= dict(s = 50, vmin = 0.15, vmax = 0.2, alpha = 1.0, 
                zorder=2, depthshade=False, edgecolor = 'k', lw=0.5),
            cobras_cmap = sns.light_palette(colors[-1], as_cmap=True),
                x_bounds_nudge=5,
             )

ax.tick_params(axis='both', which='both', bottom=False, top=False, labelsize=20)

plt.savefig('half_domain_pred_3D.png',dpi=300, transparent=True)


# Zoom in for Figure 1
fig = plt.figure(figsize=(15,5))
ax, fig = pred_3D_test(solver, aligned_long, 
                       fig=fig, 
             preds = all_pos_quants,
             z_downsample = 1, 
             cobras_kws = {'vmin':0, 'vmax':1, 's': 10},
             max_t = 401,
             skip =50,
             min_x = 0*jnp.pi, 
             max_x = 256 * jnp.pi,
            extr_kws= dict(s = 35, vmin = 0.15, vmax = 0.2, alpha = 1.0, 
                zorder=2, depthshade=False, edgecolor = 'k', lw=0.5),
            cobras_cmap = sns.light_palette(colors[-1], as_cmap=True),
            # cobras_cmap = 'Spectral_r',
            # sns.light_palette(colors[-1], as_cmap=True),
                x_bounds_nudge=5,
             )
plt.savefig('full_domain_pred_3D.png',dpi=300, transparent=True)


# max amplitude over time
q_thresh = 0.207
plt.figure(figsize=(5,1.5))
plt.plot(jnp.abs(all_snaps).max(axis=-1), lw = 2, label = 'controlled')
plt.plot(jnp.abs(aligned_long).max(axis=-1), lw = 2, label = 'uncontrolled')

plt.axhline(q_thresh, ls = '--', c = 'k')
plt.xlim(0, len(all_snaps))
plt.yticks([0.15, 0.2, 0.25, 0.3]);

plt.legend(fontsize=8)
plt.savefig('ctrl.png',dpi=300)


# Heatmap of sparse control

pal = ["white", colors[-1]]

cmap = mcolors.LinearSegmentedColormap.from_list("custom", pal)

plt.figure(figsize=(10,2))
plt.imshow(0.01*jnp.abs(all_ctrl_aligned), origin = 'lower', 
           cmap = cmap, 
           vmin = 0, vmax = 0.002, aspect='auto', extent=[0, 256 * jnp.pi, 0, len(all_ctrl_aligned)])
plt.xticks([0, 64*jnp.pi, 128*jnp.pi, 192*jnp.pi, 256*jnp.pi],
           labels = ['0', r'$64\pi$', r'$128\pi$', r'$192\pi$', r'$256\pi$'])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelsize=20)
plt.savefig('ctrl_heatmap.png', dpi=300, transparent=True)
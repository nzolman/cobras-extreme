import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import seaborn as sns

from cobras_extreme import _mnls_data_dir
import matplotlib.pyplot as plt

from cobras_extreme.plotting import set_defaults
colors = set_defaults(style = 'science', dpi=100
                      )
from cobras_extreme import _mnls_data_dir

from cobras_extreme.mnls.waves import MNLS1D

from cobras_extreme.mnls.shift_utils import get_snap_shift
from cobras_extreme.plotting import remaining_svd_energy

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



# Grab long data
long_data_path = os.path.join(_mnls_data_dir, 
                              'forward_Tf_4000', 
                              'mnls_fwd_long-4000.pkl'
                              )
import pickle
with open(long_data_path, 'rb') as f:
    long_data = pickle.load(f)
    
long_data.shape
aligned_long = shift_snap_v(long_data, jnp.arange(4000), 0.5, 1, 0)


t_max = 400
u_sq = jnp.abs(aligned_long)[:t_max]**2

(u_sq).max()
xc = jnp.pi * 32
L_gauss = jnp.pi/2
gauss = jnp.exp(-(x - xc)**2 / (2*L_gauss**2))

cmap = sns.light_palette(colors[3], as_cmap=True)
cmap.set_under("white")



fpath = os.path.join(_mnls_data_dir, 'modes.npy')
modes = jnp.load(fpath, allow_pickle=True).item()


plt.rcParams['xtick.top'] = False 
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['ytick.right'] = False 


def get_waves_plot(ax):
    ax.imshow(jnp.abs(long_data[:t_max]), 
            extent=(x[0], x[-1], 0, t_max), 
            cmap=cmap, 
            origin = 'lower', 
            aspect='auto', 
            vmin = 0,
            vmax = 0.2,)

    lw = 2.5
    lc = 'k'
    linestyles = [':', '--', '-']
    for i, n in enumerate([32, 96, 160]):
        xc = n*jnp.pi

        
        
        ax.plot(x, 2*(x - xc), linestyle = linestyles[i], color = lc,  lw = lw)
        
    ax.set_xticks(jnp.arange(0,5) *64* jnp.pi)
    ax.set_xticklabels([f'{64*i}' + r'$\pi$' for i in range(5)])
    ax.set_ylim(0,t_max)
    return ax


def get_qoi_ts(ax):
    L_gauss = jnp.pi/2
    lc = 'k'
    t_max = 400
    u_sq = jnp.abs(aligned_long)[:t_max]**2
    ls = [':', '--', '-']
    for i, n in enumerate([32, 96, 160]):
        xc = n*jnp.pi
        gauss = jnp.exp(-(x - xc)**2 / (2*L_gauss**2))
        gauss /= gauss.sum()
        q = jnp.einsum('i,Ni->N', gauss, u_sq)
        ax.plot( jnp.arange(t_max), q, ls[i], color = lc, lw = 3)
        
    ax.set_xticks(jnp.arange(0,500,100))
    ax.set_yticks(jnp.arange(0,6)/100)

    ax.tick_params(labelsize=15)
    return ax



# ------------------------------------------
# Plot Figure grid
# ------------------------------------------
fig = plt.figure(figsize=(10, 1.5))
gs = gridspec.GridSpec( 2, 2, height_ratios=[2.5, 12], # top row for colorbar, bottom for imshow 
                       width_ratios=[7.5, 2.5], # imshow wider than timeseries
                       hspace=0.05, 
                       wspace=0.25 )

# Axes 
ax_cbar = fig.add_subplot(gs[0, 0]) # colorbar above imshow 
ax_im = fig.add_subplot(gs[1, 0]) # imshow 
ax_ts = fig.add_subplot(gs[1, 1]) # timeseries spans both rows

ax_im = get_waves_plot(ax_im)
ax_ts = get_qoi_ts(ax_ts)
ax_ts.set_xticks([0, 200, 400])

ax_ts.set_yticks([0.0, 0.02, 0.04])
ax_ts.set_ylim(0, 0.05)

# Standalone colorbar
cbar = fig.colorbar(ax_im.images[0], cax=ax_cbar, orientation='horizontal', aspect = 40, pad = 0.2)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_ticks([0, 0.1, 0.2])


fontsize=10
ax_im.tick_params(axis='both', labelsize=fontsize)
ax_ts.tick_params(axis='both', labelsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

plt.savefig('mnls_intro.png', dpi=300, bbox_inches='tight', transparent=True)



# ------------------------------------------
# Plot SVD energy
# ------------------------------------------
S_cobras = modes['S']
S_pod = modes['S_pod']
S_pod_local = modes['S_pod_local']

fig, ax = plt.subplots(1,1,figsize=(3.,3))

n_mode_cobras = len(S_cobras)
n_mode_pod = len(S_pod)
n_mode_pod_local = len(S_pod_local)
ms = 5

ax.plot(jnp.arange(1, n_mode_cobras + 1), remaining_svd_energy(S_cobras),  color=colors[0], marker = 'o', ms = ms, label = 'CoBRAS')
ax.plot(jnp.arange(1, n_mode_pod + 1),remaining_svd_energy(S_pod),  color=colors[1], marker = '^', ms = ms, label = 'POD')
ax.plot(jnp.arange(1, n_mode_pod_local + 1),remaining_svd_energy(S_pod_local),  color=colors[2], marker = 'v', ms = ms, label = 'Local POD')

ax.set_yscale('log')
ax.set_xlim(0,20)
ax.set_ylim(1e-3,2)
ax.axhline(1e-2, color='k', ls='--', lw=1)
ax.set_xticks([0,5,10,15,20])
ax.tick_params(labelsize=15)
ax.legend( fontsize=10)
plt.savefig('mnls_svd_energy.png', dpi=300, bbox_inches='tight', transparent=True)




# ------------------------------------------
# Plot modes
# ------------------------------------------
Vh_pod = modes['Vh_pod']
Vh_pod_local = modes['Vh_pod_local']
Phi_cobras = modes['Phi']
Psi_cobras = modes['Psi']


fig, axes = plt.subplots(3,6, figsize=(6.5,3), sharex=True, sharey=True)
x = solver.x
xc = solver.L_domain / 2

x_centered = x - xc
win_size = 16*jnp.pi
re_style = '-'
im_style = ':'
lw = 2
lw2 = 2

axes = axes.T
for i in range(6): 
    axes[i,0].plot(x_centered[N//2 - 64:N//2 + 64], Vh_pod_local.T[:128,i]/jnp.linalg.norm(Vh_pod_local.T[:,i]), re_style, c = colors[i//2], lw = lw)
    axes[i,0].plot(x_centered[N//2 - 64:N//2 + 64], Vh_pod_local.T[128:,i]/jnp.linalg.norm(Vh_pod_local.T[:,i]), im_style, c = colors[i//2], lw = lw2)
    
    axes[i,1].plot(x_centered,Phi_cobras[:1024,i]/jnp.linalg.norm(Phi_cobras[:,i]), re_style, c = colors[i//2], lw = lw)
    axes[i,1].plot(x_centered,Phi_cobras[1024:,i]/jnp.linalg.norm(Phi_cobras[:,i]), im_style, c = colors[i//2], lw = lw2)
    
    axes[i,2].plot(x_centered,Psi_cobras[:1024,i]/jnp.linalg.norm(Psi_cobras[:,i]), re_style, c = colors[i//2], lw = lw)
    axes[i,2].plot(x_centered,Psi_cobras[1024:,i]/jnp.linalg.norm(Psi_cobras[:,i]), im_style, c = colors[i//2], lw = lw2)
    
    



pi_ticks = [-8,0, 8]
labels = [r'$-8\pi\phantom{-}$', r'$0$', r'$8\pi$']

for ax in axes.flatten():
    ax.set_xlim( - win_size,   win_size)
    ax.set_xticks(jnp.pi*jnp.array(pi_ticks), labels=labels)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([-0.2,0.0, 0.2])
    
    ax.tick_params('both', labelsize = 15)
    ax.tick_params('x', rotation=0)
    ax.axvline(0, color='gray', ls='-', lw=1, zorder = 0, alpha = 0.5)
    



fig.tight_layout()

for ax in axes.T[-1, :]:
    dx = -5/72
    offset = ScaledTranslation(dx, 0, fig.dpi_scale_trans)
    label = ax.xaxis.get_major_ticks()[0].label1
    label.set_transform(label.get_transform() + offset)

plt.savefig('mnls_modes_intro.png', dpi=300, bbox_inches='tight', transparent=True)
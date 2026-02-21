import matplotlib.pyplot as plt
import seaborn as sns

import jax.numpy as jnp

from cobras_extreme.plotting import set_defaults
    
def format_flow(snapshot, ax, **imshow_kwargs):
    im = ax.imshow(snapshot.T, **imshow_kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()
    return im, ax


def plot_modes(modes_list, n_modes = 8, scale = 2, clim = 7e-3, cmap = 'icefire', orientation = 'horizontal'):
    res = int(jnp.sqrt(modes_list[0].shape[0]))
    n_modes_list = len(modes_list)
    n_plots = n_modes
    
    if orientation == 'horizontal':
        fig, axes = plt.subplots(n_modes_list,n_plots, figsize=(n_plots*scale, n_modes_list*scale))
    else:
        fig, axes = plt.subplots(n_plots,n_modes_list,figsize=(n_modes_list*scale,n_plots*scale))
        axes = axes.T
    mode_specs = dict(vmin = -clim, vmax = clim, cmap = cmap)

    for i in range(n_plots): 
        for j, mode_list in enumerate(modes_list):
            ax = axes[j,i]
            mode = mode_list[:,i]
            mode /= jnp.linalg.norm(mode)
            format_flow(mode.reshape(res,res), ax, **mode_specs)
            
    fig.tight_layout()

    return fig, axes


def remaining_svd_energy(S): 
    return 1 - jnp.cumsum(S**2) / jnp.sum(S**2)
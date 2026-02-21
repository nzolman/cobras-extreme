import matplotlib.pyplot as plt
import seaborn as sns

import scienceplots

import jax.numpy as jnp

def set_defaults(style=None, dpi=100):
    from matplotlib import rcParams
    if style is not None:
        plt.style.use(style)
    else:
        
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 15
    
    colors = sns.color_palette('colorblind')
    sns.set_palette('colorblind')

    rcParams['figure.dpi'] = dpi
    return colors


def remaining_svd_energy(S): 
    return 1 - jnp.cumsum(S**2) / jnp.sum(S**2)
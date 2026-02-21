import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from waves import MNLS1D, NLS1D

from solver_step import iterative_func
from jax import grad, jit
from jax import random, lax, vmap

from mnls_utils import amplitude_hat
from cobras_extreme import _mnls_data_dir

# Default params
N = 2**10
L = 256 * jnp.pi
Tf = 4000
dt = 0.025
frac = 2/3
solver = MNLS1D(N, L, dt, M_contour=128, dealias_frac = frac)

num_steps = int(Tf/dt)
save_freq  = int(1/solver.dt)


def gen_data(seed): 
    key = random.PRNGKey(seed)

    xis = random.uniform(key, minval= 0.0, maxval = 2*jnp.pi, shape=solver.k.shape)

    u0_wave_hat = amplitude_hat(solver.k2pi, xis, dk = jnp.diff(solver.k2pi)[0],
                            eps=0.05, sigma = 0.1) * solver.N_grid
        
    _, traj_waves_hat = iterative_func(solver.step, u0_wave_hat, steps=num_steps, save_n = save_freq)
    traj_waves = jnp.fft.ifft(traj_waves_hat, axis=-1)
    
    return traj_waves



gen_data_v = vmap(gen_data)

if __name__ == '__main__': 
    import pickle
    from tqdm import tqdm
    import os
    

    seed = 4200
    
    data_dir = os.path.join(_mnls_data_dir, f'forward_Tf_{Tf}')
    os.makedirs(data_dir, exist_ok=True)
    fpath = os.path.join(data_dir, f'mnls_fwd_long-{Tf}.pkl')
    
    

    data = gen_data(seed)
        
    with open(fpath, 'wb') as f: 
        pickle.dump(data, f)



import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from waves import MNLS1D

from solver_step import iterative_func
from jax import grad, jit, vjp
from jax import random, lax, vmap

from mnls_utils import amplitude_hat, group_speed, load_data
from cobras_extreme import _mnls_data_dir

# Default params
N = 2**10
L = 256 * jnp.pi
Tf = 200
dt = 0.025
frac = 2/3
solver = MNLS1D(N, L, dt, M_contour=128, dealias_frac = frac)

L_gauss = jnp.pi # 4 * jnp.pi # 2 *
v_g = 0.5
xc = solver.x.mean()

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


from hermite import gauss_hermite
from jax import vmap 
def gh_shift(x, t, vg, L, L_domain):
    x_eval = (x - vg*t) 
    return gauss_hermite(x_eval/L) + gauss_hermite((L_domain -x_eval)/L)

gh_shift_v = vmap(gh_shift, in_axes = (None, 0, None, None, None))

# def qoi(traj, x, t, vg, L, L_domain): 
#     '''
#     traj: ndarray(complex128)
#         shape: n_t, n_x
#     '''
    
#     traj_amps_sq = jnp.abs(traj)**2
#     t_peak_idx = traj_amps_sq.max(axis=-1).argmax()
#     x_peak_idx = traj_amps_sq[t_peak_idx].argmax()
#     t_peak = t[t_peak_idx] 
#     x_peak = x[x_peak_idx]
    
#     x_centered = x - x_peak
#     t_centered = t - t_peak
    
#     # shape (n_t, n_x, n_basis)
#     gh_basis = gh_shift_v(x_centered, t_centered, vg, L, L_domain)
    
#     return jnp.einsum('tx,txb->tb', traj_amps_sq, gh_basis)

# def qoi_fwd(u0):
#     u0_hat = jnp.fft.fft(u0)
#     _, traj_hat = iterative_func(solver.step, u0_hat, steps=num_steps, save_n = save_freq)
    
#     traj = jnp.fft.ifft(traj_hat)
    
#     n_t, n_x = traj.shape
    
#     return qoi(traj, solver.x, jnp.arange(n_t), group_speed(char_length), char_length, solver.L_domain).flatten()

def shift_snap(snap, snap_t_idx,
               g_speed = 0.5,
               dt=1,
               offset_x = 0
               ):
    k = solver.k
    t_shift = (snap_t_idx)*dt
    
    x_shift = -g_speed * t_shift + offset_x
    
    snap_hat = jnp.fft.fft(snap)
    snap_shift_hat = snap_hat * jnp.exp(-(2.0j) * jnp.pi * x_shift * k)
    snap_shift = jnp.fft.ifft(snap_shift_hat)
    
    return snap_shift

shift_snap_v = vmap(shift_snap, in_axes=(0,0,None,None,None))

def gauss_shift(x, t, vg, L, x0):
    x_shift = (x - x0 - t * vg)/L
    bump = jnp.exp(-x_shift**2)
    return bump

gauss_shift_v = vmap(gauss_shift, in_axes=(None,0,None, None, None))


def qoi(traj, x, t, vg, L, x0, dx): 
    '''
    traj: ndarray(complex128)
        shape: n_t, n_x
    '''   
    traj_amps_sq = jnp.abs(traj)**2
    
    time_bump = gauss_shift_v(x, t,vg,L,x0)
    
    projection = dx * jnp.einsum('tx,tx->t', traj_amps_sq, time_bump)
    return projection

def qoi_fwd(u0):
    u0_hat = jnp.fft.fft(u0)
    _, traj_hat = iterative_func(solver.step, u0_hat, steps=num_steps, save_n = save_freq)
    
    traj = jnp.fft.ifft(traj_hat)
    
    n_t, n_x = traj.shape
    
    return qoi(traj, solver.x, t=jnp.arange(n_t), vg = v_g, L = L_gauss, x0 =xc,  dx = solver.dx)


    
    
def get_IC(key): 
    # key = random.PRNGKey(seed)
    xis = random.uniform(key, minval= 0.0, maxval = 2*jnp.pi, shape=solver.k.shape)
    u0_hat = amplitude_hat(solver.k2pi, xis, dk = jnp.diff(solver.k2pi)[0],
                            eps=0.05, sigma = 0.1) * solver.N_grid

    u0 = jnp.fft.ifft(u0_hat)
    return u0



def get_subICs(key, trajs, n_traj, traj_len):
    _, n_t_max, n_x = trajs.shape
    traj_key, start_key, shift_key = random.split(key, num=3)
    traj_idx = random.randint(traj_key, shape=(n_traj,), minval=0, maxval=len(trajs)-1)
    start_idx = random.randint(start_key, shape=(n_traj,), minval=0, maxval=traj_len-1)
    shift_idx = random.randint(start_key, shape=(n_traj,), minval=0, maxval=n_x-1)
    
    def get_subtraj(t_idx, s_idx, shift_idx):
        snap =  trajs[t_idx, s_idx]
        return jnp.roll(snap, shift_idx, axis=-1)
    get_subtraj_v = vmap(get_subtraj)
    return (traj_idx, start_idx, shift_idx), get_subtraj_v(traj_idx, start_idx, shift_idx)    
   
    
def get_jvp(u0, cotangent):
    y, vjp_fn = vjp(qoi_fwd, u0)
    grad_sample = vjp_fn(cotangent)[0]
    return grad_sample


get_IC_v = vmap(get_IC)
get_jvp_v = vmap(get_jvp, in_axes=(0, 0))


if __name__ == '__main__': 
    import pickle
    from tqdm import tqdm
    import os
    import time
    
    batch_size = 50
    num_samples = 5000
    n_batches = num_samples // batch_size
    
    # output_dim = 11 * Tf  # 11 basis functions if using all GH
    output_dim = 1 * Tf  # 1 gauss function
    
    seeds = jnp.arange(num_samples)
    
    Tf_load = 400
    Tf_grad = Tf
    load_dir = os.path.join(_mnls_data_dir, f'forward_Tf_{Tf_load}')
    data_dir = os.path.join(_mnls_data_dir, f'backward_Tf-load_{Tf_load}_grad_{Tf_grad}-gauss_{L_gauss:.02f}')
    os.makedirs(data_dir, exist_ok=True)
    
    tic = time.time()
    data_trajs = load_data(load_dir, device_type='cpu')
    
    toc = time.time()
    print('Data Loaded!', toc-tic)
    key = random.PRNGKey(num_samples + 1) # since IC seeds go from 0 to num_samples -1
    cotangets = random.normal(key, shape=(num_samples, output_dim))
    key = random.split(key)[0]
    
    tic = time.time()
    (traj_idxes, start_idxes, shift_idxes), all_u0s = get_subICs(key, data_trajs, num_samples, Tf)
    toc = time.time()
    print('Acquired ICs', toc-tic)
    
    for i in tqdm(range(n_batches)):
        
        traj_idx = traj_idxes[i*batch_size:(i+1)*batch_size]
        start_idx = start_idxes[i*batch_size:(i+1)*batch_size]
        shift_idx = shift_idxes[i*batch_size:(i+1)*batch_size]
        
        cots = cotangets[i*batch_size:(i+1)*batch_size]
        u0s = all_u0s[i*batch_size:(i+1)*batch_size]
        u0s = u0s.to_device(jax.devices()[0]) # put on the correct device
        grad_data = get_jvp_v(u0s, cots)
        
        data = {'traj_idx': traj_idx,
                'start_idx': start_idx,
                'shift_idx': shift_idx,
                'grad_data': grad_data}
        
        fpath_i = os.path.join(data_dir, f'mnls_bwd_{i:04}.pkl')
        with open(fpath_i, 'wb') as f: 
            pickle.dump(data, f)

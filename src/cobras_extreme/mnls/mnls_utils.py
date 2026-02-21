import jax.numpy as jnp
import glob
import pickle
import os
from tqdm import tqdm
from jax import devices

def surface_elevation(u,x,t):
    u_tmp = u * jnp.exp(1j * (x - t))
    return jnp.real(u_tmp)

def gauss(k, eps=0.05, sigma=0.1):
    coef = (eps**2) / (sigma * jnp.sqrt(2 * jnp.pi))
    exp = jnp.exp(-(k**2)/(2*sigma **2))
    return coef * exp

def amplitude_hat(k,xi,dk, eps=0.05, sigma = 0.1):
    A_k = jnp.sqrt(2 * dk *  gauss(k, eps, sigma)) * jnp.exp(1j * xi)
    return A_k

def group_speed_k(k):
    return 0.5 - 1/4 * k + 3/16 * k**2

def group_speed(L):
    k = 2 * jnp.pi / L
    return group_speed_k(k)


def load_data(dir_path, device_type = 'cpu'):
    # to-do: maybe parallel loading? 
    device = devices(device_type)[0]
    fpaths = glob.glob(os.path.join(dir_path, '*.pkl'))
    fpaths.sort()
    
    all_trajs = []
    for fpath in tqdm(fpaths):
        with open(fpath, 'rb') as f:
            all_trajs.append(pickle.load(f).to_device(device))
            
    all_trajs = jnp.concatenate(all_trajs)
    return all_trajs


def load_bwd_data(data_dir):
    fpaths = glob.glob(os.path.join(data_dir, '*.pkl'))
    fpaths.sort()
    all_grads = []
    all_traj_idxs = []
    all_start_idxs = []
    all_shift_idxs = []
    for fpath in fpaths:
        with open(fpath, 'rb') as f:
            data_i = pickle.load(f)
        all_grads.append(data_i['grad_data'])
        all_traj_idxs.append(data_i['traj_idx'])
        all_start_idxs.append(data_i['start_idx'])
        all_shift_idxs.append(data_i['shift_idx'])
        
    all_grads = jnp.concatenate(all_grads, axis=0)
    all_traj_idxs = jnp.concatenate(all_traj_idxs, axis=0)
    all_start_idxs = jnp.concatenate(all_start_idxs, axis=0)    
    all_shift_idxs = jnp.concatenate(all_shift_idxs, axis=0)
    return all_grads, all_traj_idxs, all_start_idxs, all_shift_idxs
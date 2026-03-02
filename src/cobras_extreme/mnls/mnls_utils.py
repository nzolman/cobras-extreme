import jax.numpy as jnp
import glob
import pickle
import os
from tqdm import tqdm
from jax import devices, random, lax, vmap

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


def load_data(dir_path, device_type = 'cpu', n_files = None, max_t = None):
    # to-do: maybe parallel loading? 
    device = devices(device_type)[0]
    fpaths = glob.glob(os.path.join(dir_path, '*.pkl'))
    fpaths.sort()
    if n_files is not None:
        fpaths = fpaths[:n_files]
    all_trajs = []
    for fpath in tqdm(fpaths):
        with open(fpath, 'rb') as f:
            traj = pickle.load(f).to_device(device)
            if max_t is not None:
                traj = traj[:,:max_t]
            all_trajs.append(traj)
            
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



def get_subICs(key, trajs, n_traj, traj_len):
    _, n_t_max, n_x = trajs.shape
    traj_key, start_key, shift_key = random.split(key, num=3)
    traj_idx = random.randint(traj_key, shape=(n_traj,), minval=0, maxval=len(trajs)-1)
    start_idx = random.randint(start_key, shape=(n_traj,), minval=0, maxval=traj_len-1)
    shift_idx = random.randint(shift_key, shape=(n_traj,), minval=0, maxval=n_x-1)
    
    def get_subtraj(args):
        t_idx, s_idx, shift_idx = args
        snap =  trajs[t_idx, s_idx]
        return jnp.roll(snap, shift_idx, axis=-1)

    subs = lax.map(get_subtraj, (traj_idx, start_idx, shift_idx), batch_size = 50)
    return (traj_idx, start_idx, shift_idx), subs


def load_ICs_from_path(key, fpath, device_type = 'cpu', max_t = None, n_traj=50, traj_len = 200):
    device = devices(device_type)[0]
    
    with open(fpath, 'rb') as f:
        trajs = pickle.load(f).to_device(device)
        if max_t is not None:
            trajs = trajs[:,:max_t]
    (traj_idx, start_idx, shift_idx), subs=get_subICs(key, trajs, n_traj = n_traj, traj_len = traj_len)
    
    return (traj_idx, start_idx, shift_idx), subs


def get_all_ICs(key, fpaths, n_traj_per_file = 50, n_sample_per_file = 50, traj_len = 200, device_type = 'cpu'):
    all_traj_idx = []
    all_start_idx = []
    all_shift_idx = []
    all_subICs = []
    for fpath_idx, fpath in enumerate(tqdm(fpaths)):
        key, subkey = random.split(key)
        (traj_idx, start_idx, shift_idx), subICs = load_ICs_from_path(subkey, 
                                                                      fpath, 
                                                                      device_type=device_type, 
                                                                      max_t=None, 
                                                                      n_traj=n_sample_per_file,
                                                                      traj_len=traj_len
                                                                      )
        all_traj_idx.append(traj_idx + fpath_idx*n_traj_per_file)
        all_start_idx.append(start_idx)
        all_shift_idx.append(shift_idx)
        all_subICs.append(subICs)
        
    all_subICs = jnp.concatenate(all_subICs)
    all_traj_idx = jnp.concatenate(all_traj_idx)
    all_start_idx = jnp.concatenate(all_start_idx)
    all_shift_idx = jnp.concatenate(all_shift_idx)
    
    return (all_traj_idx, all_start_idx, all_shift_idx), all_subICs

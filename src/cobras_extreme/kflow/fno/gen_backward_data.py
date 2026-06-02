import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
from pprint import pprint

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, vmap, jit
import numpy as np
from tqdm import tqdm

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

from cobras_extreme.kflow.data_utils import get_flow, backward_parser, get_backwards_fn, get_forward_fn
from cobras_extreme.kflow.fno.evaluate import load_params, build_apply_fn, predict_physical, load_norm_stats

from cobras_extreme import _kflow_data_dir, _kflow_dir
root_dir = _kflow_dir



# python gen_backward_data.py --Re 40 --k 4 --res 256 --save_dt_load 0.5 --save_dt_back 0.5 --end_time 5000 --forecast 8 --n_grads 320 --batch_size 32
if __name__ == '__main__': 
    from jax import devices
    device = devices()[0]
    continue_from_start = True
    
    dt_str = '0_5'
    n_preds = 8

    # backwards params
    batch_size = 2
    T_f = 4
    save_time_back = 0.5
    res = 256
    k = 4
    Re = 40
    end_time = 5000
    save_time_load = 0.5
    load_name = f'kolmogorov_n={int(res)}-Re={int(Re)}_k={k}_end={int(end_time)}_save={save_time_load}.npy'
    
    save_dir_name = f'surrogate_fno_grads_dt={dt_str}'

    load_path = os.path.join(_kflow_data_dir,
                             'forward', 
                             load_name
                             )
    print("Loading from: ", load_path)
    
    save_dir = os.path.join(_kflow_data_dir,
                             'fno',
                             save_dir_name)
    
    print("Saving to: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # ---------------
    # load FNO data
    # ---------------
    
    
    
    fno_dir = os.path.join(_kflow_data_dir, 'fno')
    params_path = os.path.join(fno_dir, 
                               f'output_dt={dt_str}',
                               'checkpoints',
                               'best_params.pkl'
                               )
    
    params = load_params(params_path)
    jax_params = jax.tree.map(jnp.asarray, params)
    
    modes = 16
    width = 32
    n_layers = 4
    apply_fn = build_apply_fn(modes, width, n_layers)
    norm_stats = load_norm_stats(os.path.join(fno_dir, 
                                              f'output_dt={dt_str}',
                                              'norm_stats.json'
                                              )
                                 )
    
        
    nu = 1 / Re
    def qoi_v(traj): 
        return nu * (traj ** 2).mean(axis=(-2,-1))

    

    def fwd(x0): 
        preds = jnp.zeros((n_preds+1, res,res))
        preds = preds.at[0].set(x0)
        for i in jnp.arange(1, n_preds+1):
            pred = predict_physical(apply_fn, jax_params, preds[i-1].reshape(1, res,res,1), norm_stats)[0,...,0]
            preds = preds.at[i].set(pred)
        return preds[1:]

    def fwd_qoi(x0): 
        traj = fwd(x0)
        return qoi_v(traj)

    # ---------------
    # start by loading onto 
    omega_hats = np.load(load_path)
    warm_start = 10 // save_time_load
    
    n_fwd_tot = len(omega_hats)
    n_fwd = n_fwd_tot - warm_start
    
    skip_ratio = 2
    
    idxes = np.arange(warm_start, n_fwd_tot, skip_ratio, dtype=int)
    n_bwd = len(idxes)
    
    key = random.PRNGKey(0)
    
    cotangent_vs = random.normal(key, 
                                 shape=(n_bwd, int(T_f//save_time_back))
                                 )
        

    back_fn = get_backwards_fn(fwd_qoi)
    
    print('starting grads')
    t0 = time.time()
    
    back_fn_v = vmap(back_fn)
    
    def bwd_batch(batch_idxes):
        omega_hats_0 = jnp.asarray(omega_hats[batch_idxes]).to_device(device)
        omega_0s = jnp.fft.irfft2(omega_hats_0)
        cotangents = cotangent_vs[batch_idxes]
        inputs = {'snap': omega_0s, 
                'cotangent': cotangents
                }
        grad_samples = back_fn_v(inputs)
        grad_dict = {'idxes': batch_idxes,
                     'grad_samples': grad_samples}
        return grad_dict
            
    n_batches = n_bwd//batch_size + (n_bwd % batch_size != 0)
    
    batch_start = 0
    if continue_from_start: 
        import glob
        batch_start = len(glob.glob(os.path.join(save_dir ,'*.npy')))
        print(f'Starting from file #{batch_start}!')
    
    for b_idx in tqdm(range(batch_start, n_batches)):
        batch_idxes = idxes[b_idx * batch_size: (b_idx+1)*batch_size]
        grad_dict = bwd_batch(batch_idxes)
        
        idx_0 = batch_idxes[0]
        save_path = os.path.join(save_dir, f'{idx_0:04}')
        jnp.save(save_path, grad_dict)

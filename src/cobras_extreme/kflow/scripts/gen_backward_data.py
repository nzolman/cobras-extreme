import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time
from pprint import pprint

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, lax, vmap
import numpy as np
from tqdm import tqdm

cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

from cobras_extreme.kflow.data_utils import get_flow, backward_parser, get_backwards_fn, get_forward_fn

from cobras_extreme import _kflow_dir
root_dir = _kflow_dir



# python gen_backward_data.py --Re 40 --k 4 --res 256 --save_dt_load 0.5 --save_dt_back 0.5 --end_time 5000 --forecast 8 --n_grads 320 --batch_size 32
if __name__ == '__main__': 
    # parse args and convert to dict
    parser = backward_parser()
    args = parser.parse_args()
    config = vars(args) # to dict
    pprint(config)
    
    flow = get_flow(config)
    
    dt = config['dt']
    
    
    # load params
    n = flow.grid_size[0]
    k = config['k']
    Re = flow.Re
    end_time = config['end_time']
    save_time_load = config['save_dt_load']
    
    
    # backwards params
    T_f = config['forecast']
    save_time_back = config['save_dt_back']
    qoi = config['qoi']
    n_grads = config['n_grads']
    batch_size = config['batch_size']

    load_name = f'kolmogorov_n={int(n)}-Re={int(Re)}_k={k}_end={int(end_time)}_save={save_time_load}.npy'
    
    save_dir_name = f'grads_n={int(n)}-Re={int(Re)}_k={k}_end={int(end_time)}_save-load={save_time_load}_save-back={save_time_back}_Tf={T_f}_n_grads={n_grads}'
    
    

    load_path = os.path.join(_kflow_dir,
                             'forward', 
                             load_name
                             )
    print("Loading from: ", load_path)
    
    save_dir = os.path.join(_kflow_dir,
                             'backward',
                             save_dir_name)
    print("Saving to: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # start by loading onto 
    omega_hats = np.load(load_path)
    warm_start = 10 // save_time_load
    
    n_fwd_tot = len(omega_hats)
    n_fwd = n_fwd_tot - warm_start
    
    skip_ratio = n_fwd // n_grads
    
    idxes = np.arange(warm_start, n_fwd_tot, skip_ratio, dtype=int)
    n_bwd = len(idxes)


    # n_bwd = len(omega_0s)
    
    
    key = random.PRNGKey(0)
    
    cotangent_vs = random.normal(key, 
                                 shape=(n_bwd, int(T_f//save_time_back))
                                 )
    

    
    fwd_fn = get_forward_fn(flow, qoi, T_f, dt, save_time_back)
    back_fn = get_backwards_fn(fwd_fn)
    
    print('starting grads')
    t0 = time.time()
    
    back_fn_v = vmap(back_fn)
    
    def bwd_batch(batch_idxes):
        omega_hats_0 = omega_hats[batch_idxes]
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
    if config['continue']: 
        import glob
        batch_start = len(glob.glob(os.path.join(save_dir ,'*.npy')))
        print(f'Starting from file #{batch_start}!')
    
    for b_idx in tqdm(range(batch_start, n_batches)):
        batch_idxes = idxes[b_idx * batch_size: (b_idx+1)*batch_size]
        grad_dict = bwd_batch(batch_idxes)
        
        idx_0 = batch_idxes[0]
        save_path = os.path.join(save_dir, f'{idx_0:04}')
        jnp.save(save_path, grad_dict)

    
    # grad_samples = lax.map(back_fn, inputs, 
    #                        batch_size=batch_size
    #                        )
    # print('time: ', time.time() - t0)
    
    # jnp.save(save_path, grad_samples)
    
    # print('done.')
    
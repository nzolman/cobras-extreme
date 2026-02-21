import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


from jax import numpy as jnp
from jax import lax

import glob
import numpy as np


import cobras_extreme
from cobras_extreme import _kflow_data_dir
fwd_data_dir = os.path.join(_kflow_data_dir, 'forward')
grad_data_dir = os.path.join(_kflow_data_dir, 'backward')
proj_data_dir = os.path.join(_kflow_data_dir, 'projections')
ker_data_dir = os.path.join(_kflow_data_dir, 'kernel')

from equations.flow import FlowConfig
from cobras_kflow.symm_utils import shift_sx_fft, get_fourier_idx
from cobras_kflow.kernel_utils import Y_star_Kx, Y_star_X, Y_star_X_v, K


res=256
Re=100
Tf=4

def get_ker(Re=40, Tf=4, res=256, N_train= 4000, n_modes = 20):
    res = int(res)
    Re = int(Re)
    Tf = int(Tf)
    flow = FlowConfig(grid_size = (res,res), Re = Re)
    kx, ky = flow.create_fft_mesh()

    a_10_idx, a_01_idx = get_fourier_idx(flow)

    def shift_sx_fft_wrapper(d):
        x_hat = d['snaps']
        grads_hat = d['grads']
        return shift_sx_fft(x_hat, grads_hat, kx, ky, a_10_idx)

    grad_dir = os.path.join(grad_data_dir, 
                            f'grads_n={res}-Re={Re}_k=4_end=5000_save-load=0.5_save-back=0.5_Tf={Tf}_n_grads=5000'
                            )
    fpaths = glob.glob(os.path.join(grad_dir, '*.npy'))
    fpaths.sort()
    print('num files', len(fpaths))


    # load gradient data
    idx_data = []
    grad_data = []
    for fpath in fpaths:
        data = np.load(fpath, allow_pickle=True).item()
        grad_data.append(data['grad_samples'])
        idx_data.append(data['idxes'])
        
    grad_data=np.concatenate(grad_data)
    idx_data=np.concatenate(idx_data)

    # load snap data
    snap_path = os.path.join(fwd_data_dir, f'kolmogorov_n={int(res)}-Re={int(Re)}_k=4_end=5000_save=0.5.npy')
    snap_hat_data = np.load(snap_path)


    # we want to remove the first several time steps to ensure we're properly on the attractor
    N_remove = 10
    grad_data = grad_data[N_remove:]
    snap_hat_data = snap_hat_data[2*N_remove:]

    print('num grads', len(grad_data))
    print('num snaps', len(snap_hat_data))

    N_x = len(snap_hat_data)
    N_g = len(grad_data)


    grad_hat_data = np.fft.rfft2(grad_data)
    snap_hat_red = snap_hat_data[::2][:N_g]
    data_hat = {'snaps': snap_hat_red, 'grads': grad_hat_data}

    snap_data_red = np.fft.irfft2(snap_hat_red)

    snap_flat_red = snap_data_red.reshape(-1, res**2)
    grad_flat = grad_data.reshape(-1, res**2)

    # grad_flat = 1/np.sqrt(N_g) * grad_flat
    # snap_flat_red =  1/np.sqrt(N_g) * snap_flat_red

    print(grad_flat.shape, snap_flat_red.shape)

    sigma = jnp.linalg.norm(snap_flat_red, axis=-1).mean()/2

    wrap_fn = lambda xj: Y_star_X_v(snap_flat_red, xj, grad_flat, N_train, N_train, sigma)

    Y_star_X_data = lax.map(wrap_fn, snap_flat_red, batch_size=1)

    U_ker, S_ker, Vh_ker = jnp.linalg.svd(Y_star_X_data.T, full_matrices=False)

    r = n_modes

    S_inv_sqrt = jnp.diag(jnp.sqrt(1/S_ker[:r]))

    from jax import vmap
    Y_star_KX_v = vmap(Y_star_Kx, in_axes=(None, 0, 0, None, None))


    def h(x):
        Y_star = Y_star_KX_v(x, snap_flat_red, grad_flat, N_g, sigma)
        return S_inv_sqrt[:r] @ U_ker[:,:r].T @ Y_star

    h_X = lax.map(h, snap_flat_red, batch_size=1)
    
        
    # Same for shifted data
    x_hat_shifts, grads_hat_shifts, s_xs = lax.map(shift_sx_fft_wrapper, data_hat, batch_size=1)
        
    x_shifts = np.fft.irfft2(x_hat_shifts)
    grad_shifts = np.fft.irfft2(grads_hat_shifts)
        
    x_shifts_flat = x_shifts.reshape(-1, res**2)
    grad_shifts_flat = grad_shifts.reshape(-1, res**2)
    
    wrap_fn_shift = lambda xj: Y_star_X_v(x_shifts_flat, xj, grad_shifts_flat, N_g, N_g, sigma)

    Y_star_X_data_shift = lax.map(wrap_fn_shift, x_shifts_flat, batch_size=1)
        
    U_ker_shift, S_ker_shift, Vh_ker_shift = jnp.linalg.svd(Y_star_X_data_shift.T, full_matrices=False)

    S_inv_sqrt_shift = jnp.diag(jnp.sqrt(1/S_ker_shift[:r]))
    
    from jax import vmap
    Y_star_KX_v = vmap(Y_star_Kx, in_axes=(None, 0, 0, None, None))


    def h_shift(x):
        Y_star = Y_star_KX_v(x, x_shifts_flat, grad_shifts_flat, N_g, sigma)
        return S_inv_sqrt_shift[:r] @ U_ker_shift[:,:r].T @ Y_star

    h_X_shift = lax.map(h_shift, x_shifts_flat, batch_size=1)

    data = dict(scale = sigma,
                S_ker = S_ker,
                z_ker = h_X,
                S_ker_symm = S_ker_shift,
                z_ker_symm = h_X_shift,
                )
    
    return data

if __name__ == '__main__': 
    
    from cobras_kflow import _data_dir
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser('get kernel cobras projections')
    parser.add_argument(
            '--Re',
            help='Reynolds number',
            default=40,
            type = float,
    )
    parser.add_argument('--Tf',
                        help='gradient length',
                        default=4,
                        type=int
                        )
    
    parser.add_argument('--res',
                        help='spatial resolution',
                        default=256,
                        type=int
                        )
    parser.add_argument('--N_train',
                        help='number of training samples used to build projections',
                        default=4000,
                        type=int
                        )
    
    parser.add_argument('--n_modes',
                        help='number of modes to build projections',
                        default=20,
                        type=int
                        )
    
    
    
    args = parser.parse_args()
    
    config = vars(args) # to dict
    pprint(config)
    res = int(config['res'])
    Re = int(config['Re'])
    Tf = int(config['Tf'])
    save_path = os.path.join(ker_data_dir, 
                             f'res={res}_Re={int(Re)}_Tf={Tf}'
                             )
    data = get_ker(**config)
    
    print('Saving to...')
    print(save_path)
    jnp.save(save_path, data)
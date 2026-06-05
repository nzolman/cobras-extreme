import os
from tqdm import tqdm
os.environ['JAX_PLATFORMS']='cpu'

import numpy as np
from jax import numpy as jnp
from jax import lax

import glob

from cobras_extreme import _kflow_data_dir
fwd_data_dir = os.path.join(_kflow_data_dir, 'forward')
grad_data_dir = os.path.join(_kflow_data_dir, 'backward')
proj_data_dir = os.path.join(_kflow_data_dir, 'projections')
ker_data_dir = os.path.join(_kflow_data_dir, 'kernel')

from equations.flow import FlowConfig
from cobras_extreme.kflow.symm_utils import shift_sx_fft, get_fourier_idx


def get_phi_psi(X, Y, U, S, Vh, r):
    S_inv_sqrt_r = np.diag(1.0 / np.sqrt(S[:r]))
    U_r = U[:, :r]
    Vh_r = Vh[:r, :]
    
    Phi = X @ Vh_r.T @ S_inv_sqrt_r
    Psi = Y @ U_r @ S_inv_sqrt_r
    
    return Phi, Psi

def get_projs(Re=40, Tf=4, res=256, N_train= 4000, n_modes = 20):
    
    print('Creating Mesh...')
    flow = FlowConfig(grid_size = (res,res), Re = Re)
    kx, ky = flow.create_fft_mesh()

    a_10_idx, a_01_idx = get_fourier_idx(flow)

    def shift_sx_fft_wrapper(d):
        x_hat = d['snaps']
        grads_hat = d['grads']
        return shift_sx_fft(x_hat, grads_hat, kx, ky, a_10_idx)
    
    
    grad_dir = os.path.join(grad_data_dir, 
                            f'grads_n={res}-Re={int(Re)}_k=4_end=5000_save-load=0.5_save-back=0.5_Tf={Tf}_n_grads=5000'
                            )
    print('Gathering gradients from : ')
    print(grad_dir)
    
    fpaths = glob.glob(os.path.join(grad_dir, '*.npy'))
    fpaths.sort()
    


    # load gradient data
    idx_data = []
    grad_data = []
    for fpath in tqdm(fpaths):
        data = np.load(fpath, allow_pickle=True).item()
        grad_data.append(data['grad_samples'])
        idx_data.append(data['idxes'])
    
    print('n grad files: ', len(grad_data))
    grad_data=jnp.concatenate(grad_data)
    idx_data=jnp.concatenate(idx_data)


    # load snap data
    print('Loading Snapshots...')
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


    # use same data as gradients
    snap_hat_red = snap_hat_data[::2][:N_g]
    
    print('Computing FFT...')
    print('\t Snaps')
    # convert to vorticity space
    snap_data_red = np.fft.irfft2(snap_hat_red)
    
    print('\t Grads')
    grad_hat_data = np.fft.rfft2(grad_data)
    
    data_hat = {'snaps': snap_hat_red, 'grads': grad_hat_data}


    snap_flat_red = snap_data_red.reshape(-1, res**2)
    grad_flat = grad_data.reshape(-1, res**2)

    # Compute CoBRAS
    print('Computing CoBRAS + SVD')
    Y_star_X_lin = 1/jnp.sqrt(N_train**2) * grad_flat[:N_train] @ snap_flat_red[:N_train].T
    cobras_U, cobras_S, cobras_Vh = jnp.linalg.svd(Y_star_X_lin, full_matrices=False)

    snap_train = snap_flat_red[:N_train]
    grad_train = grad_flat[:N_train]
    # CoBRAS
    Phi, Psi = get_phi_psi(
                            1/jnp.sqrt(N_train) * snap_train.T, 
                            1/jnp.sqrt(N_train) * grad_train.T, 
                            cobras_U, cobras_S, cobras_Vh, 
                            r = n_modes
                        )
    print('CoBRAS Projections: Psi shape:', Psi.shape)
    
    
    
    snap_mean = snap_train.mean(axis=0)
    pod_snap = snap_train - snap_mean
    
    print('pod_snap shape', pod_snap.shape)
    
    pod_U, pod_S, pod_Vh = jnp.linalg.svd((1.0/N_train)*(pod_snap) @ (pod_snap).T, 
                                          full_matrices=False
                                          )
    
    print('Computed POD')
    # n_modes x n_state
    # pod_Psi  = jnp.diag(1.0/jnp.sqrt(pod_S[:n_modes])) @ pod_Vh[:n_modes] @ pod_snap

    pod_Psi = (1/np.sqrt(N_train)) * pod_snap.T @ pod_U[:, :n_modes] @ np.diag(1/np.sqrt(pod_S[:n_modes]))
    
    print('Computed POD projections', pod_Psi.shape)
    print(snap_hat_data.shape)
    
    
    print('Shifting Data (Symmetry Reduction)...')
    x_hat_shifts, grads_hat_shifts, s_xs = lax.map(shift_sx_fft_wrapper, data_hat, batch_size=1)
    
    print('Inverse FFT shifted data...')
    print('\t snaps')
    x_shifts = np.fft.irfft2(x_hat_shifts)
    print('\t grads')
    grad_shifts = np.fft.irfft2(grads_hat_shifts)

    x_shifts_flat = x_shifts.reshape(-1, res**2)
    grad_shifts_flat = grad_shifts.reshape(-1, res**2)
    
    grads_shift_train = grad_shifts_flat[:N_train]
    x_shift_train = x_shifts_flat[:N_train]
        
    print('Computing Shifted CoBRAS SVD')
    Y_star_X_lin_symm = 1/jnp.sqrt(N_train**2) * grads_shift_train @ x_shift_train.T
    cobras_symm_U, cobras_symm_S, cobras_symm_Vh = np.linalg.svd(Y_star_X_lin_symm, 
                                                                 full_matrices=False
                                                                 )
    
    print('Computing Shifted CoBRAS projections')
    Phi_symm, Psi_symm = get_phi_psi(1/jnp.sqrt(N_train) * x_shift_train.T, 
                                    1/jnp.sqrt(N_train) * grads_shift_train.T, 
                                    cobras_symm_U, cobras_symm_S, cobras_symm_Vh, 
                                    r = n_modes
                                    )

    x_shift_mean = x_shift_train.mean(axis=0)
    pod_x_shift = x_shift_train - x_shift_mean
    
    pod_symm_U, pod_symm_S, pod_symm_Vh = jnp.linalg.svd((1/N_train)*pod_x_shift @ pod_x_shift.T,
                                                         full_matrices=False)


    pod_Psi_symm = (1/np.sqrt(N_train)) * pod_x_shift.T @ pod_symm_U[:, :n_modes] @ np.diag(1/np.sqrt(pod_symm_S[:n_modes]))
    # pod_Psi_symm = jnp.diag(1.0/jnp.sqrt(pod_symm_S[:n_modes])) @ pod_symm_Vh[:n_modes] @ pod_x_shift

    
    nu = 1.0 / Re
    
    e_disp = nu * jnp.mean(snap_flat_red**2, axis=-1)
    
    
    data = dict(Phi=Phi, 
                Psi=Psi,
                cobras_U=cobras_U, 
                cobras_S=cobras_S,
                cobras_Vh=cobras_Vh,
                pod_U = pod_U,
                pod_S = pod_S,
                pod_Vh = pod_Vh,
                pod_Psi=pod_Psi,
                cobras_symm_U = cobras_symm_U,
                cobras_symm_S = cobras_symm_S,
                cobras_symm_Vh = cobras_symm_Vh,
                Phi_symm = Phi_symm,
                Psi_symm = Psi_symm,
                pod_symm_U = pod_symm_U,
                pod_symm_S = pod_symm_S,
                pod_symm_Vh = pod_symm_Vh,
                pod_Psi_symm = pod_Psi_symm,
                z_cobras_all = snap_flat_red @ Psi,
                z_pod_all = (snap_flat_red - snap_mean) @ pod_Psi,
                z_cobras_symm_all = x_shifts_flat @ Psi_symm,
                z_pod_symm_all = (x_shifts_flat - x_shift_mean) @ pod_Psi_symm,
                e_disp = e_disp
                )
    
    return data

if __name__ == '__main__': 
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser('get linear cobras projections')
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
    save_path = os.path.join(proj_data_dir,
                             f'res={res}_Re={int(Re)}_Tf={Tf}'
                             )
    data = get_projs(**config)
    
    print('Saving to...')
    print(save_path)
    jnp.save(save_path, data)
'''
For the first 1000 trajectories, grab the subtrajs used to 
compute the gradients, and align them in the frame of reference 
of the wave moving at group velocity.
'''

import os
from tqdm import tqdm

from jax import vmap, lax
import jax.numpy as jnp

from cobras_extreme.mnls.waves import MNLS1D
from cobras_extreme.mnls.mnls_utils import load_data, load_bwd_data
from cobras_extreme import _mnls_data_dir

from cobras_extreme.mnls.shift_utils import get_snap_shift


# Setup solver with default params
N = 2**10
L_domain = 256 * jnp.pi
Tf = 200
dt = 0.025
frac = 2/3
solver = MNLS1D(N, L_domain, dt, M_contour=128, dealias_frac = frac)
x = solver.x

# grab shifter
shift_snap_v = get_snap_shift(solver)


# Grab forward data
fwd_dir = os.path.join(_mnls_data_dir,
                       'forward_Tf_400'
                       )

print('Loading fwd data from:', fwd_dir)
forward_data = load_data(fwd_dir, device_type='cpu')
print('forward_data.shape:', forward_data.shape)

# grab bwd data
bwd_dir = os.path.join(_mnls_data_dir,
                       'backward_Tf-load_400_grad_200-gauss_1.57'
                       )

print('Loading backward data from:', bwd_dir)
all_grads, all_traj_idxs, all_start_idxs, all_shift_idxs = load_bwd_data(bwd_dir)
print('all_grads.shape:', all_grads.shape)

# shift in frame of reference
def get_subtraj(t_idx, s_idx, shift_idx):
    traj = forward_data[t_idx]
    snap = lax.dynamic_slice_in_dim(traj, s_idx, 200, axis=0)
    return jnp.roll(snap, shift_idx, axis=-1)
get_subtraj_v = vmap(get_subtraj)


n_traj = 1000
subtrajs = get_subtraj_v(all_traj_idxs[:n_traj], 
                         all_start_idxs[:n_traj], 
                         all_shift_idxs[:n_traj]
                         )

print('subtrajs.shape:', subtrajs.shape)

aligned =jnp.array([shift_snap_v(subtraj, 
                                 jnp.arange(200), 
                                 0.5,   # group speed
                                 1,     # time step
                                 0      # offset idx (already shifted to center)
                                ) 
                        for subtraj in tqdm(subtrajs)
                    ]
                   )

aligned_save_path = os.path.join(_mnls_data_dir,
                                 'aligned_subtrajs_1k.npy'
                                 )
jnp.save(aligned_save_path, aligned)
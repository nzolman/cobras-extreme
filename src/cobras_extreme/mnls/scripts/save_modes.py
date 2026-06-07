'''
We save the CoBRAS modes, POD modes, and local POD modes to a file for later use
'''

import os
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import glob


from jax import random
cpu_device = jax.devices('cpu')[0]

from cobras_extreme import _mnls_data_dir
from cobras_extreme.plotting import set_defaults
colors = set_defaults(style = 'science', dpi=300)

from cobras_extreme.mnls.waves import MNLS1D
from cobras_extreme.mnls.shift_utils import get_snap_shift
from cobras_extreme.mnls.complex_utils import complex_to_real
from cobras_extreme.mnls.mnls_utils import load_bwd_data, get_all_ICs
from cobras_extreme.cobras import get_modes


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


# Grab Long data
long_data_path = os.path.join(_mnls_data_dir, 
                              'forward_Tf_4000', 
                              'mnls_fwd_long-4000.pkl'
                              )


key = random.PRNGKey(5001)
_, ic_key = random.split(key)

snap_path = os.path.join(_mnls_data_dir, 
                         'forward_Tf_400'
                         )
fpaths = glob.glob(os.path.join(snap_path, 
                                '*.pkl'))

_, x_snaps = get_all_ICs(ic_key, fpaths, n_traj_per_file = 50, n_sample_per_file = 50, traj_len = 200)


width = '1.57'
fpath = os.path.join(_mnls_data_dir, 
                     f'backward_Tf-load_400_grad_200-gauss_{width}'
                     )
grads = load_bwd_data(fpath)[0]

snaps_ri = complex_to_real(x_snaps, axis=-1)
grads_ri = complex_to_real(grads, axis=-1)



modes = get_modes(snaps_ri, grads_ri, r=20)

Phi_cobras = modes['Phi']
Psi_cobras = modes['Psi']
S_cobras = modes['S']


snaps_ri_pod = snaps_ri - snaps_ri.mean(axis=0, keepdims=True)

U_pod, S_pod, Vh_pod = jnp.linalg.svd(snaps_ri_pod.T @ snaps_ri_pod, hermitian=True, full_matrices=False)


# For a local POD, just compute the SVD on the middle of the domain
# (Note because the system is translation-equivariant, 
# the statistics are translation invariant, so the location of the window 
# doesn't really matter, but we choose the center for consistence)

middle_128_idx = jnp.arange(N//2 - 64, N//2 + 64)
snaps_ri_middle = complex_to_real(x_snaps[:,middle_128_idx], axis=-1)

snaps_ri_pod_local = snaps_ri_middle - snaps_ri_middle.mean(axis=0, keepdims=True)

U_pod_local, S_pod_local, Vh_pod_local = jnp.linalg.svd(snaps_ri_pod_local.T @ snaps_ri_pod_local, hermitian=True, full_matrices=False)


modes = get_modes(snaps_ri, grads_ri, r=20)

# Since we have more data than dimensions, we will use
# classical POD rather than the method of snapshots. 
# So we really only need U or Vh

Phi_cobras = modes['Phi']
Psi_cobras = modes['Psi']
S_cobras = modes['S']

modes['U_pod'] = U_pod[:,:20]
modes['S_pod'] = S_pod
modes['Vh_pod'] = Vh_pod[:20]

modes['U_pod_local'] = U_pod_local[:,:20]
modes['S_pod_local'] = S_pod_local
modes['Vh_pod_local'] = Vh_pod_local[:20]


modes_path = os.path.join(_mnls_data_dir,'modes.npy')
jnp.save(modes_path, modes, allow_pickle=True)
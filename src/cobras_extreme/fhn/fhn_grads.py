import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('..')

from jax import random, vmap
import jax.numpy as jnp
from jax.experimental import sparse

from cobras_extreme import _fhn_data_dir
from cobras_extreme.fhn.fhn import setup_small_world, get_grads
from cobras_extreme.fhn.fhn import gen_data_FCN, get_data_gen, setup_FCN

key = random.PRNGKey(0)
key_init, key_cotangents = random.split(key,2)
key_b = random.PRNGKey(1) # used for dynamics

# system size
n, m = 100,100
N = n*m

# load data
traj_path = os.path.join(_fhn_data_dir, '2025-12-31_fhn-world_0_real.npy')
trajs = jnp.load(traj_path, allow_pickle=True)

# load network
adj_path = os.path.join(_fhn_data_dir, 'small_world_adj_networkx.npy')
jax_dict_A = jnp.load(adj_path, allow_pickle=True).item()
jax_A = sparse.BCOO((jax_dict_A['data'], jax_dict_A['indices']), shape=(10000, 10000))

t_skip = 10                     # downsample factor used for producing original data
T_remove = 5000                 # amount of initial data to remove to ensure on attractor
n_remove = T_remove // t_skip
n_traj_train = 4                # number of trajectories to use for training data

X_train = jnp.concatenate(trajs[:n_traj_train, n_remove:], axis=0)


f_diffrax = setup_small_world(key_b, jax_A, n,m, k_nn=60, k=0.128)


key = random.PRNGKey(0)
key_init, key_cotangents = random.split(key,2)



tol = 1e-12
dt0 = 1e-3
dtmax = 5e-3
Tf_fwd = 50
save_dt = 1
skip_samples = 2
batch_size = 8
dump_freq = 10

save_name = os.path.join(_fhn_data_dir, 
                         f'fhn_small_world_grads-Tf={Tf_fwd}_square.npy'
                         )

all_grads = get_grads(f_diffrax, 
                      X_train, 
                      key_cotangents, 
                      Tf_fwd=Tf_fwd, 
                      save_dt = save_dt, 
                      dt0 = dt0,
                      dtmax = dtmax, 
                      tol = tol, 
                      skip_samples = skip_samples, 
                      batch_size=batch_size,
                      batch_save_freq=dump_freq,
                      save_name=save_name)
jnp.save(save_name, all_grads)
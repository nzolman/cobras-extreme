'''
This script trains an SVM classifier on the aligned trajectory
using features from the CoBRAS modes as input. We save SVM parameters,
decision values, and the resulting feedback control by using it as a
suppression strategy on the long trajectory.
'''


import os
from pprint import pprint
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from jax import jit
cpu_device = jax.devices('cpu')[0]

from cobras_extreme import _mnls_data_dir

from cobras_extreme.plotting import set_defaults
colors = set_defaults(style = 'science', dpi=300)

from cobras_extreme.mnls.waves import MNLS1D
from cobras_extreme.mnls.solver_step import iterative_func
from cobras_extreme.mnls.shift_utils import get_snap_shift
from cobras_extreme.mnls.unity_utils import periodic_channelwise_correlation_fft_batched
from cobras_extreme.mnls.complex_utils import split_ri_axis, ri_axis_to_complex, complex_to_ri_axis

from cobras_extreme.kflow.classification import train_svm_classifier

# Setup solver with default params
N = 2**10
L_domain = 256 * jnp.pi
Tf = 200
dt = 0.025
frac = 2/3
solver = MNLS1D(N, L_domain, dt, M_contour=128, dealias_frac = frac)
x = solver.x



def mnls_labels(trajs, thresh, t_pred=200):
    max_amp_0 = jnp.abs(trajs)[:,:t_pred].max(axis=(1,))
    labels = max_amp_0 > thresh
    return labels

def mnls_labels_rolling(traj, thresh, Tf=200):
    """
    traj: (T, N) single trajectory
    returns: (T-Tf, N) boolean array — at each t, whether max amplitude 
             in [t, t+Tf] exceeds thresh
    """
    T, N = traj.shape
    abs_traj = jnp.abs(traj)
    
    def window_label(t):
        window = jax.lax.dynamic_slice(abs_traj, (t, 0), (Tf, N))
        return window.max(axis=0) > thresh
    
    return jax.vmap(window_label)(jnp.arange(T - Tf))  # (T-Tf, N)


# grab shifter
shift_snap_v = get_snap_shift(solver)


# Grab Long data and shift
# this will be our completely untouched dataset used
# for evaluating.
long_data_path = os.path.join(_mnls_data_dir, 
                              'forward_Tf_4000', 
                              'mnls_fwd_long-4000.pkl'
                              )
import pickle
with open(long_data_path, 'rb') as f:
    long_data = pickle.load(f)
    
long_data.shape
aligned_long = shift_snap_v(long_data, jnp.arange(4000), 0.5, 1, 0)


# Grab modes
modes_path = os.path.join(_mnls_data_dir,'modes.npy')
modes = jnp.load(modes_path, allow_pickle=True).item()
Phi_cobras = modes['Phi']
Psi_cobras = modes['Psi']
S_cobras = modes['S']


# We will use aligned trajectory data to build our svm dataset
subs_path = os.path.join(_mnls_data_dir, 
                         'aligned_subtrajs_1k.npy')

aligned_subtrajs = jnp.load(subs_path, allow_pickle=True).to_device(cpu_device)
print('aligned_subtrajs.shape:', aligned_subtrajs.shape)



# break modes into 2d with real and imag components
psi_cobras_split = split_ri_axis(Psi_cobras, axis=0, axis_new = -1)
phi_cobras_split = split_ri_axis(Phi_cobras, axis=0, axis_new = -1)


# Convolve all Psis with the aligned long data
center_idx = 511
T_conv = 401 # change this if you want to use a longer dataset

u_real_i = complex_to_ri_axis(aligned_long)[:T_conv]

# this is for the long eval trajectory
psi_corr = periodic_channelwise_correlation_fft_batched(u_real_i, 
                                                        jnp.roll(psi_cobras_split, 
                                                                 shift=center_idx, axis=0
                                                                 )
                                                        )

# sum over the real/imag axis
psi_corr_sum = psi_corr.sum(axis=-1)


# Again for all the training data.
aligned_subtrajs_real = complex_to_ri_axis(aligned_subtrajs)
aligned_subtrajs_real_0 = aligned_subtrajs_real[:,0]

psi_corr_train = periodic_channelwise_correlation_fft_batched(aligned_subtrajs_real_0, 
                                                              jnp.roll(psi_cobras_split, 
                                                                       shift=center_idx, axis=0
                                                                       )
                                                              )

psi_corr_sum_train = psi_corr_train.sum(axis=-1)


t_pred = 200
q_thresh = jnp.abs(aligned_subtrajs).mean() + 4 * jnp.abs(aligned_subtrajs).std()
print('thresh', q_thresh)

n_traj_red = 500
n_x_skip = 4

# We use a reduced dataset, prioritizing variance over trajectories
# instead of variance over space. However, if we skip more than every 
# 4th point, we risk missing peaks. 
Y_all = mnls_labels(aligned_subtrajs, q_thresh, t_pred=t_pred)
Y_red = Y_all[:n_traj_red,::n_x_skip]
X_red = psi_corr_sum_train[:n_traj_red,::n_x_skip]

X_red_flat = jnp.concatenate(X_red, axis=0)
Y_red_flat = jnp.concatenate(Y_red, axis=0)

train_idx = jnp.arange(int(0.8 * len(Y_red_flat)))
test_idx = jnp.arange(int(0.8 * len(Y_red_flat)), 
                      len(Y_red_flat))




n_modes = 8

labels = Y_red_flat

svm_kwargs = dict(kernel='rbf',
                    C=1.0, 
                    gamma='scale', 
                    class_weight = 'balanced',
                    random_state = 0,
                max_iter=50000,
                probability=False # turn on to use CV with probas. 
                                    # NOTE: THIS WILL SLOW DOWN SIGNIFICANTLY
                )

print('training svm...')
metrics, svm, scaler = train_svm_classifier(
                                    X_red_flat[:,:n_modes], 
                                    labels,
                                    train_idx,
                                    test_idx,
                                    return_svm=True, 
                                    svm_kwargs = svm_kwargs
                                    )
print('metrics:')
pprint(metrics)




# Extract static parameters once
sv = jnp.array(svm.support_vectors_)    # (n_sv, n_features)
dual_coefs = jnp.array(svm.dual_coef_.ravel())  # (n_sv,)
gamma = svm._gamma
b = jnp.array(svm.intercept_)
s_mean = jnp.array(scaler.mean_)
s_scale = jnp.array(scaler.scale_)



@jax.jit
def decision_function(x_unscaled):
    """x: (n_points, n_features)"""
    x = (x_unscaled - s_mean) / s_scale
    sv_sqnorms = jnp.sum(sv ** 2, axis=1)
    x_sqnorms  = jnp.sum(x ** 2, axis=1)
    sq_dists = x_sqnorms[:, None] + sv_sqnorms[None, :] - 2 * (x @ sv.T)
    K = jnp.exp(-gamma * sq_dists)
    return (K * dual_coefs[None, :]).sum(axis=1) + b


@jax.jit
def svm_gradient(x_unscaled):
    """x: (n_points, n_features)"""
    x = (x_unscaled - s_mean) / s_scale
    sv_sqnorms = jnp.sum(sv ** 2, axis=1)
    x_sqnorms  = jnp.sum(x ** 2, axis=1)
    sq_dists = x_sqnorms[:, None] + sv_sqnorms[None, :] - 2 * (x @ sv.T)
    K = jnp.exp(-gamma * sq_dists)
    W = K * dual_coefs[None, :]
    scaled_grad = 2 * gamma * (W @ sv - W.sum(axis=1, keepdims=True) * x)
    return scaled_grad / s_scale




# we compute the distribution of decision boundaries
all_decs =  jnp.array([decision_function(snap) 
                       for snap in tqdm(psi_corr_sum[:600,:,:n_modes])
                       ])

all_train_decs = jnp.array([decision_function(snap) 
                       for snap in tqdm(X_red[:,:,:n_modes])
                       ])


# Precompute quantile boundaries from the distribution
# we use the test_idx as a validaiton set (note this is separate) 
# from our final evaluation, which is on the long trajectory.
n_quantiles = 100
quantile_boundaries = jnp.array(jnp.quantile(all_train_decs.flatten()[test_idx][all_train_decs.flatten()[test_idx] >= 0], 
                                             jnp.linspace(0, 1, n_quantiles + 1)
                                             )
                                )

@jax.jit
def assign_quantile(x):
    # Returns indices 0..n_quantiles-1
    return jnp.clip(jnp.searchsorted(quantile_boundaries, x, side='right') - 1, 0, n_quantiles - 1)

all_pos_quants = jnp.linspace(0, 1, n_quantiles + 1)[assign_quantile(all_decs.flatten())].reshape(all_decs.shape)



decs_dict = {'all_train_decs': all_train_decs,
            'aligned_long_decs': all_decs,
            'all_pos_quants': all_pos_quants
            }

svm_dir = os.path.join(_mnls_data_dir, 'svm')
os.makedirs(svm_dir, exist_ok=True)

jnp.save(f'{svm_dir}/svm_r-{n_modes}-decs.npy', 
         decs_dict)


svm_dict = {
    'sv': sv,
    'dual_coefs': dual_coefs,
    'gamma': gamma,
    'b': b,
    's_mean': s_mean,
    's_scale': s_scale,
    'quantile_boundaries': quantile_boundaries,
 }

jnp.save(f'{svm_dir}/svm_r-{n_modes}-svm_params.npy', 
         svm_dict)




@jit
def get_ctrl_suppresion(state, centered_psi, Phi_0_complex): 

    # transform to real axis
    state_ri = complex_to_ri_axis(state.reshape(1,-1))
    
    # local projection onto Psi modes
    z_eta =  periodic_channelwise_correlation_fft_batched(state_ri, centered_psi)[0].sum(axis=-1)

    # compute classifier gradient and normalize
    svm_grad = svm_gradient(z_eta)
    unit_svm_grad = svm_grad / jnp.linalg.norm(svm_grad, axis=-1, keepdims=True)


    # weight the control by the positive quantile of the decision function
    # normal events receive 0 control
    dec = decision_function(z_eta) >= 0
    u_z = dec.reshape(-1, 1) * unit_svm_grad

    u_A_complex = Phi_0_complex @ u_z.T

    return u_A_complex


shift_idx = center_idx
Phi_0_complex = ri_axis_to_complex(phi_cobras_split[shift_idx,:n_modes], 
                                   axis=-1)
centered_psi = jnp.roll(psi_cobras_split[:,:n_modes], 
                        shift=shift_idx, 
                        axis=0)

u0_wave = long_data[0]

n_modes = 8
k_gain = 0.01
dTf = 1

num_steps = int(dTf/dt)
save_freq  = int(1/dt)

u_wave = u0_wave
all_snaps = [u_wave]
all_ctrl = []


print('Starting control...')
for i in tqdm(range(400)):
    ctrl = get_ctrl_suppresion(u_wave, centered_psi, Phi_0_complex)
    solver.set_control(-k_gain * ctrl)
    all_ctrl.append(ctrl)
    u_wave_hat = jnp.fft.fft(u_wave)

    _, traj_hat = iterative_func(solver.step, u_wave_hat, steps=num_steps, save_n = save_freq)
    traj = jnp.fft.ifft(traj_hat, axis=-1)
    u_wave = traj[-1]
    all_snaps.append(u_wave)
    
all_snaps = jnp.array(all_snaps)
all_ctrl = jnp.array(all_ctrl)


ctrl_results = {'snaps': all_snaps,
                'ctrls': all_ctrl
                }

jnp.save(f'{svm_dir}/svm_r-{n_modes}-ctrl.npy', 
         ctrl_results)
print('Control results saved.')
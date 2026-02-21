from jax import numpy as jnp
from jax import scipy as jscp
import scipy as scp

import cobras_kflow
from equations.flow import FlowConfig


def get_fourier_idx(flow):
    x_mesh, y_mesh = flow.create_mesh()
    kx, ky = flow.create_fft_mesh()
    wave_numbers = jnp.array([(kx * 2*jnp.pi).flatten(),(ky*2*jnp.pi).flatten()]).T
    a_10_idx = jnp.argmin(jnp.linalg.norm(wave_numbers - jnp.array([1,0]), axis=1))
    a_01_idx = jnp.argmin(jnp.linalg.norm(wave_numbers - jnp.array([0,1]), axis=1))
    return a_10_idx, a_01_idx



def apply_shift_fft(x_hat, off_x, off_y, kx, ky):
    shifted_hat = jnp.exp(-2*jnp.pi* 1.0j*(kx * off_x + ky * off_y)) * x_hat
    return shifted_hat

# def apply_shift_real(x, off_x, off_y, kx, ky):
#     x_hat = jnp.fft.rfft2(x)
#     shifted_hat = apply_shift_fft(x_hat, off_x, off_y, kx, ky)
#     shifted = jnp.fft.irfftn(shifted_hat)
#     return shifted

# def compute_shift_fft(x_hat, x_ref_hat, x_mesh, y_mesh):
#     convolved = jnp.fft.irfftn(x_ref_hat * jnp.conjugate(x_hat))
#     ker = (1/9.)*jnp.ones((3,3))
#     # convolved = jscp.signal.convolve2d(convolved, ker)
#     convolved = scp.signal.convolve2d(convolved, ker, boundary='wrap')

#     max_idx = jnp.array(jnp.where((convolved.max() == convolved))).T[0]
#     x_idx, y_idx = max_idx
    
#     # take average w.r.t. neighbor values (accomodates for resolution)
#     x_loc = x_mesh[x_idx, y_idx] 
#     y_loc = y_mesh[x_idx, y_idx] 
    
#     return (x_loc, y_loc)

# def compute_shift_real(x, x_ref, x_mesh, y_mesh):
#     x_hat = jnp.fft.rfftn(x)
#     x_ref_hat = jnp.fft.rfftn(x_ref)
    
#     (x_loc, y_loc) = compute_shift_fft(x_hat, x_ref_hat, x_mesh, y_mesh)
    
#     return (x_loc, y_loc)

# # @jit
# def snapshot_to_template(x, x_ref, x_mesh, y_mesh, kx, ky):
#     (x_loc, y_loc) = compute_shift_real(x, x_ref, x_mesh, y_mesh)
    
#     shifted = apply_shift_real(x, x_loc, y_loc, kx, ky)
#     return shifted
def shift_sx_fft_snap(x_hat, kx, ky, a_10_idx):
    n = x_hat.shape[-2]*x_hat.shape[-1]
    s_x =  jnp.angle(x_hat.reshape(n)[a_10_idx])
    s_y = 0
    x_hat_shift = apply_shift_fft(x_hat, s_x, s_y, kx, ky)
    return x_hat_shift, s_x

def shift_sx_fft(x_hat, grads_hat, kx, ky, a_10_idx):
    n = x_hat.shape[-2]*x_hat.shape[-1]
    s_x =  jnp.angle(x_hat.reshape(n)[a_10_idx])
    s_y = 0
    x_hat_shift = apply_shift_fft(x_hat, s_x, s_y, kx, ky)
    grads_hat_shift = apply_shift_fft(grads_hat, s_x, s_y, kx, ky)
    return x_hat_shift, grads_hat_shift, s_x

def shift_sy_fft(x_hat, grads_hat,  kx, ky, a_01_idx):
    n = x_hat.shape[-2]*x_hat.shape[-1]
    s_y =  jnp.angle(x_hat.reshape(n)[a_01_idx])
    # s_y = y_period * (s_y // y_period) # discrete
    # s_y = jnp.round(s_y/y_period)*y_period
    s_x = 0
    x_hat_shift = apply_shift_fft(x_hat, s_x, s_y, kx, ky)
    grads_hat_shift = apply_shift_fft(grads_hat, s_x, s_y, kx, ky)
    return x_hat_shift, grads_hat_shift, s_y

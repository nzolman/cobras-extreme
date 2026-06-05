from jax import numpy as jnp
from jax import scipy as jscp
import scipy as scp

import cobras_extreme
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
    s_x = 0
    x_hat_shift = apply_shift_fft(x_hat, s_x, s_y, kx, ky)
    grads_hat_shift = apply_shift_fft(grads_hat, s_x, s_y, kx, ky)
    return x_hat_shift, grads_hat_shift, s_y

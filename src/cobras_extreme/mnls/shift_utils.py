from jax import vmap
import jax.numpy as jnp

from cobras_extreme.mnls.complex_utils import complex_to_real, split_ri_axis, merge_ri_axis

def get_snap_shift(solver):
    def shift_snap(snap, snap_t_idx,
                g_speed = 0.5,
                dt=1,
                offset_x = 0
                ):
        k = solver.k
        t_shift = (snap_t_idx)*dt
        
        x_shift = -g_speed * t_shift + offset_x
        
        snap_hat = jnp.fft.fft(snap)
        snap_shift_hat = snap_hat * jnp.exp(-(2.0j) * jnp.pi * x_shift * k)
        snap_shift = jnp.fft.ifft(snap_shift_hat)
        
        return snap_shift

    shift_snap_v = vmap(shift_snap, in_axes=(0,0,None,None,None))
    return shift_snap_v


def shift_modes(modes, center, dx):
    '''
    center must be an integer multiple of dx! (note, reference center is defined
    at the center of the domain)
    '''
    x_dim = modes.shape[0] // 2
    ref_center_idx = x_dim // 2
    
    shift_idx = ref_center_idx - (center / dx).astype(int)
    modes_ri = split_ri_axis(modes, axis=0, axis_new=-1)  # (x_dim, n_modes, 2)

    modes_ri_shift = jnp.roll(modes_ri, shift=-shift_idx, axis=0)
    modes_shift = merge_ri_axis(modes_ri_shift, axis_ri=-1, axis_out=0)
    return modes_shift    

def local_projection(Psi_modes_2N, u, center, dx): 
    modes_shift = shift_modes(Psi_modes_2N, center, dx)
    u_2n = complex_to_real(u, axis=-1)
    return u_2n @ modes_shift

shift_modes_v = vmap(shift_modes, in_axes=(None, 0, None))
local_proj_v = vmap(local_projection, in_axes=(None, None, 0, None))

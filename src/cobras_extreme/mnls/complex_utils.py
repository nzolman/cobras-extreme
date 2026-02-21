from jax import numpy as jnp

def complex_to_real(z, axis):
    """
    Take a complex array z and concatenate its real and imaginary parts
    along the given axis. If z.shape[axis] = N, output.shape[axis] = 2N.
    """
    axis = axis % z.ndim

    real = jnp.real(z)
    imag = jnp.imag(z)

    return jnp.concatenate([real, imag], axis=axis)


def real_to_complex(x, axis):
    """
    Convert a real array whose given axis stores real parts in the first half
    and imaginary parts in the second half into a complex array along that axis.
    
    x:    real-valued JAX array
    axis: integer axis index where real/imag halves live
    """
    # Normalize axis index
    axis = axis % x.ndim
    
    # Size of the 2N axis
    L = x.shape[axis]
    
    N = L // 2
    
    # Slice real and imaginary parts
    real = jnp.take(x, indices=jnp.arange(N), axis=axis)
    imag = jnp.take(x, indices=jnp.arange(N, 2*N), axis=axis)
    
    # Combine into complex
    return real + 1j * imag

def complex_to_ri_axis(z, axis_new=-1):
    """
    Convert a complex array z into a real array with a new axis of size 2
    storing [real, imag] along axis_new.
    """
    real = jnp.real(z)
    imag = jnp.imag(z)
    stacked = jnp.stack([real, imag], axis=axis_new)
    return stacked

def ri_axis_to_complex(x, axis=-1):
    """
    Convert a real array with a 2-sized axis (real, imag) into a complex array.
    """
    axis = axis % x.ndim
    real = jnp.take(x, 0, axis=axis)
    imag = jnp.take(x, 1, axis=axis)
    return real + 1j * imag

def split_ri_axis(x, axis, axis_new=-1):
    """
    Take a real array whose given axis has length 2N (real then imag),
    and split it into a new axis of size 2: (..., N, 2, ...).
    """
    axis = axis % x.ndim
    L = x.shape[axis]
    N = L // 2

    real = jnp.take(x, jnp.arange(N), axis=axis)
    imag = jnp.take(x, jnp.arange(N, 2*N), axis=axis)

    stacked = jnp.stack([real, imag], axis=axis_new)
    return stacked

def merge_ri_axis(x, axis_ri, axis_out):
    """
    Inverse of split_ri_axis: take an array with a 2-sized axis (real, imag)
    and merge it into a single axis of length 2N.
    """
    axis_ri = axis_ri % x.ndim
    real = jnp.take(x, 0, axis=axis_ri)
    imag = jnp.take(x, 1, axis=axis_ri)
    merged = jnp.concatenate([real, imag], axis=axis_out)
    return merged




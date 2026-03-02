import jax.numpy as jnp
import os

from cobras_extreme import _kflow_data_dir

def load_snaps(Re=40, res=256, return_real = False, snap_idx = jnp.array([0])):
    fwd_data_dir = os.path.join(_kflow_data_dir, 'forward')
    fname = f'kolmogorov_n={res}-Re={Re}_k=4_end=5000_save=0.5.npy'
    snap_hat_data = jnp.load(os.path.join(fwd_data_dir, fname), 
                             allow_pickle=True)[snap_idx]
    
    if return_real:
        return jnp.fft.irfft2(snap_hat_data, axes=(-2,-1))
    else:
        return snap_hat_data

def load_projs(Re=40, Tf=4, res=256, return_snaps = False, **snap_kwargs):
    '''To-do: specify device'''
    proj_data_dir = os.path.join(_kflow_data_dir, 'projections')
    ker_data_dir = os.path.join(_kflow_data_dir, 'kernel')
    
    fname = f'res={res}_Re={Re}_Tf={Tf}.npy'
    proj_fpath = os.path.join(proj_data_dir, fname)
    data = jnp.load(proj_fpath, allow_pickle=True).item()
    
    ker_fpath = os.path.join(ker_data_dir, fname)
    ker_data = jnp.load(ker_fpath, allow_pickle=True).item()
    data.update(ker_data)
    
    if return_snaps: 
        data['snaps'] = load_snaps(Re=Re, res=res, **snap_kwargs)
    return data
from jax import numpy as jnp
from jax import random


def get_factorization(key, grads, snaps, res):
    '''
    grads: ndarray (N_samples, N_subsamples, res, res)
    '''
    
    # state dimension
    n = res * res 
    N_grad = len(grads)
    N_snap = len(snaps)
    
    grads_flatish = grads.reshape(*grads.shape[:2], n) 
    xis = random.normal(key, shape=grads_flatish.shape[:2])
    key, subkey = random.split(key)
    
    Y = (1/jnp.sqrt(N_grad)) * jnp.einsum('Nn,Nnj->Nj', xis, grads_flatish)
    
    snaps_flat = snaps.reshape(N_snap, n)
    X = (1/jnp.sqrt(N_snap)) * snaps_flat
    
    # return transpose to align with CoBRAS paper
    return subkey, (X.T, Y.T)

def get_svd(X,Y):
    return jnp.linalg.svd(Y.T @ X)

def get_phi_psi(X, Y, U, S, Vh, r):
    S_inv_sqrt_r = jnp.diag(1.0 / jnp.sqrt(S[:r]))
    U_r = U[:, :r]
    Vh_r = Vh[:r, :]
    
    Phi = X @ Vh_r.T @ S_inv_sqrt_r
    Psi = Y @ U_r @ S_inv_sqrt_r
    
    return Phi, Psi

from jax import jit
import jax.numpy as jnp

from cobras_extreme.mnls.complex_utils import complex_to_ri_axis
from cobras_extreme.mnls.unity_utils import periodic_channelwise_correlation_fft_batched


def get_svm_fns(sv, s_mean, s_scale, gamma, dual_coefs, b):
    @jit
    def decision_function(x_unscaled):
        """x: (n_points, n_features)"""
        x = (x_unscaled - s_mean) / s_scale
        sv_sqnorms = jnp.sum(sv ** 2, axis=1)
        x_sqnorms  = jnp.sum(x ** 2, axis=1)
        sq_dists = x_sqnorms[:, None] + sv_sqnorms[None, :] - 2 * (x @ sv.T)
        K = jnp.exp(-gamma * sq_dists)
        return (K * dual_coefs[None, :]).sum(axis=1) + b

    @jit
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
    
    return decision_function, svm_gradient

def get_quant_fn(quantile_boundaries,n_quantiles):
    @jit
    def assign_quantile(x):
        # Returns indices 0..n_quantiles-1
        return jnp.clip(jnp.searchsorted(quantile_boundaries, x, side='right') - 1, 0, n_quantiles - 1)
    return assign_quantile

def get_ctrl_suppresion_fn(decision_function, svm_gradient):

    @jit
    def get_ctrl_suppresion(state, centered_psi, Phi_0_complex): 

        # transform to real axis
        state_ri = complex_to_ri_axis(state.reshape(1,-1))
        
        # local projection onto Psi modes
        z_eta =  periodic_channelwise_correlation_fft_batched(state_ri, centered_psi)[0].sum(axis=-1)

        # compute classifier gradient and normalize
        svm_grad = svm_gradient(z_eta)
        unit_svm_grad = svm_grad / jnp.linalg.norm(svm_grad, axis=-1, keepdims=True)

        # weight the control whether or not there's a predicted extreme event
        dec = decision_function(z_eta) >= 0
        u_z = dec.reshape(-1, 1) * unit_svm_grad

        # transform back to physical space
        u_A_complex = Phi_0_complex @ u_z.T

        return u_A_complex

    return get_ctrl_suppresion
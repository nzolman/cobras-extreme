import os

import jax.numpy as jnp
import numpy as np

r'''This is purely for RBF kernel centered at zero.

(Y^*K_x)_i = (1/sqrt(s_g)) * g_i^T G(x_i)^{-1} \nabla K_x(x_i)
G(x_i)^-1 = beta(x_i) [I - gamma(x_i) x_i x_i^T ]
\nabla K_x(x_i) = p(\alpha + x_i^T x)^{p-1}x


(Y^*K_x)_i  = (1/sqrt(s_g)) * g_i^T (beta(x_i) [I - gamma(x_i) x_i x_i^T ])(p(\alpha + x_i^T x)^{p-1}x)
            = \eta(x, x_i)[g_i^T x - \gamma(x_i) g_i^T x_i * x_i^T x]

Where
    \eta(x, x_i) = p * \beta(x_i) * (\alpha + x_i^T x)^{p-1} / sqrt(s_g)
'''


def K(x,y, sigma): 
    diff_sq = jnp.dot(x-y, x-y)
    
    return jnp.exp(-diff_sq/(2*sigma**2))

def DK(x,y, sigma): 
    return (-1/sigma**2) * K(x,y,sigma) * (x - y)


def Y_star_Kx(x, xi, gi, s_g, sigma):
    coef = (sigma**2/jnp.sqrt(s_g))
    return coef * jnp.dot(gi, DK(xi,x,sigma))


def Y_star_X(xi, xj, gi, s_g, s_x, sigma): 
    zero = jnp.zeros_like(xj)
    dK_ij = DK(xi, xj, sigma) - DK(xi, zero, sigma)
    
    coef = sigma**2/jnp.sqrt(s_g*s_x)

    return coef * jnp.dot(gi, dK_ij)
    

from jax import vmap
Y_star_X_v = vmap(Y_star_X, in_axes=(0, None, 0, None, None, None))
Y_star_X_vv = vmap(Y_star_X_v, in_axes=(None,0, None, None, None, None))

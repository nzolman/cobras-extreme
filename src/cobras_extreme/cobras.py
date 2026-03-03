import jax.numpy as jnp

def balance_factorization(X,Y):
    cobras_balance = Y.T @ X

    U_cobras, S_cobras, Vh_cobras = jnp.linalg.svd(cobras_balance, 
                                                   full_matrices = False
                                                   )
    return U_cobras, S_cobras, Vh_cobras

def get_phi_psi(X, Y, U, S, Vh, r):
    S_inv_sqrt_r = jnp.diag(1.0 / jnp.sqrt(S[:r]))
    U_r = U[:, :r]
    Vh_r = Vh[:r, :]
    
    Phi = X @ Vh_r.T @ S_inv_sqrt_r
    Psi = Y @ U_r @ S_inv_sqrt_r
    
    return Phi, Psi


def get_modes(x_real, g_real, r = 20): 
    n_grad = x_real.shape[0]
    n_snap = g_real.shape[0]
    Y_cobras = 1/jnp.sqrt(n_grad) * g_real.T
    X_cobras = 1/jnp.sqrt(n_snap) * x_real.T

    cobras_balance = Y_cobras.T @ X_cobras

    U_cobras, S_cobras, Vh_cobras = jnp.linalg.svd(cobras_balance, full_matrices = False)
    Phi_cobras, Psi_cobras = get_phi_psi(X_cobras, Y_cobras, U_cobras, S_cobras, Vh_cobras, r=r)
    
    return {'Phi': Phi_cobras, 'Psi': Psi_cobras, 'S': S_cobras}

if __name__ == "__main__":
    # Example usage
    n_grad = 100
    n_snap = 1000
    
    x0s = jnp.arange(10000).reshape((n_snap, 10))
    grads = jnp.arange(1000).reshape((n_grad, 10))
        

    Y_cobras = 1/jnp.sqrt(n_grad) * grads.T
    X_cobras = 1/jnp.sqrt(n_snap) * x0s.T


    U, S, Vh = balance_factorization(X_cobras, Y_cobras)
    
    r = 5  # Desired rank
    Phi, Psi = get_phi_psi(X_cobras, Y_cobras, U, S, Vh, r)
    
    print("Phi:", Phi)
    print("Psi:", Psi)
import jax.numpy as jnp

def periodic_tent(x, center, width, a, b):
    """
    Periodic tent function on [a,b] with peak at `center` and full width `width`.
    The domain is treated as periodic with period L = b - a.
    """
    L = b - a
    # Periodic distance from x to center
    dx = jnp.abs(x - center)
    dx = jnp.minimum(dx, L - dx)

    # Tent shape: height 1 at center, 0 at distance width/2
    half = width / 2.0
    return jnp.maximum(1.0 - dx / half, 0.0)


def periodic_tent_partition_of_unity(x, a, b, K):
    """
    Fully periodic partition of unity on [a,b] using K periodic tent functions.

    Parameters
    ----------
    x : array
        Grid points where the partition is evaluated.
    a, b : float
        Interval endpoints (periodic domain).
    K : int
        Number of tents.

    Returns
    -------
    w : array of shape (K, len(x))
        w[k, :] is the k-th periodic tent evaluated at x.
        Sum over k is identically 1.
    """
    L = b - a
    centers = jnp.linspace(a, b, K, endpoint=False)
    width = L / (K - 1) * 2.0  # ensures smooth overlap

    # Evaluate each periodic tent
    w = jnp.stack(
        [periodic_tent(x, centers[k], width, a, b) for k in range(K)],
        axis=0
    )

    # Normalize to enforce partition of unity
    w = w / (jnp.sum(w, axis=0, keepdims=True) + 1e-12)

    return w, centers



def periodic_channelwise_correlation_fft_batched(u, psi):
    """
    Computes channel-wise periodic correlation:
        (psi_i ⋆ u_k)_j(x_n) = ∫ u_k,j(y) psi_i,j(y - x_n) dy

    Parameters
    ----------
    u : array, shape (T, N, 2)
        Batch of T signals.
    psi : array, shape (N, r, 2)
        Bank of r filters.
    dx : float
        Grid spacing.

    Returns
    -------
    corr : array, shape (T, N, r, 2)
        corr[k, n, i, j] = (ψ_i,j ⋆ u_k,j)(x_n)
    """

    T, N, C = u.shape
    # assert C == 2

    # --- IMPORTANT ---
    # Flip psi along the spatial axis to match ψ(y - x)
    psi_flipped = jnp.flip(psi, axis=0)   # (N, r, 2)

    # FFT along spatial axes
    U = jnp.fft.fft(u, axis=1)                 # (T, N, 2)
    Psi = jnp.fft.fft(psi_flipped, axis=0)     # (N, r, 2)

    # Rearrange psi to (r, N, 2)
    Psi = Psi.transpose(1, 0, 2)

    # Broadcast:
    #   U_b:   (1, T, N, 2)
    #   Psi_b: (r, 1, N, 2)
    U_b   = U[None, ...]
    Psi_b = Psi[:, None, ...]

    # CORRELATION: multiply by conjugate of Psi
    F_corr = jnp.conj(Psi_b) * U_b            # (r, T, N, 2)

    # Inverse FFT along spatial axis (-2)
    corr = jnp.fft.ifft(F_corr, axis=-2).real # (r, T, N, 2)

    # reorder to (T, N, r, 2)
    corr = corr.transpose(1, 2, 0, 3)

    return corr

def recon_from_z(tens_fns, z_local, shifted_phi_complex, n_modes):
    recon = jnp.einsum('cx,tcr,cxr->tx', tens_fns, z_local[...,:n_modes], shifted_phi_complex[..., :n_modes])
    return recon

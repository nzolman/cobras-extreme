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


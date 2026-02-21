import jax
import jax.numpy as jnp
from jax import lax
from jax import jit
from functools import partial


def iterative_func(func, initialization, steps, save_n, ignore_intermediate_steps=True):
  """
  Credit: Sajeda Mokbel [1]
  Lax.scan to iteratively apply a function given an initial value 

  Args:
      func (method): the time stepping function
      initialization(grid array): the initial state
      steps (int):  number of timesteps
      save_n (int): save every n steps
      ignore_intermediate_steps (bool): if saving every n steps, ignore intermediate steps.
                                        this drastically reduces the memory requirements.

    [1] https://github.com/smokbel/Controlling-Kolmogorov-Flow/blob/45563a836b2eb3ad8ba319ec873b7b2df4802585/solvers/transient.py#L40
  """
  if ignore_intermediate_steps:
    
    def inner_scan(initialization):
      @partial(jax.checkpoint,
         policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
      def f(init, inputs):
        return (func(init), init)
      # f = lambda init, inputs: (func(init), init)
      final_state, outputs = lax.scan(f, initialization, xs=None, length=save_n)
      return final_state
    
    @partial(jax.checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def outer_scan(init, inputs):
        results = inner_scan(init)
        return (results, results)
    
    # outer_scan = lambda init, inputs: (inner_scan(init), inner_scan(init))
    outer_steps = int(steps / save_n)
    final_state, outputs = lax.scan(outer_scan, initialization, xs=None, length=outer_steps)
    return final_state, outputs
  
  else:
    @partial(jax.checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def f(init, inputs):
      return (func(init), init)
    # f = lambda init, inputs: (func(init), init)
    # Scan used to iteratively apply timestepping
    final_state, outputs = lax.scan(f, initialization, xs=None,length=steps)
    return final_state, outputs

def compute_etd_coefficients_contour(L, h, M=32):
    """
    Compute ETDRK4 coefficients using FULL circle contour integration.
    
    This works for both real and complex eigenvalues.
    
    Mathematical background:
    For diagonal operators, we evaluate:
        f(L_j) = (1/2πi) ∮ f(ζ)/(ζ - L_j) dζ
    
    For a circle centered at L_j with radius r:
        ζ = L_j + r·e^(iθ)
        dζ = i·r·e^(iθ)·dθ
    
    The integral becomes:
        f(L_j) = (1/2π) ∫₀^(2π) f(L_j + r·e^(iθ)) dθ
               = mean over full circle of f(L_j + r·e^(iθ))
    
    Args:
        L: Linear operator eigenvalues (1D array in Fourier space)
        h: Time step
        M: Number of contour points (default: 32)
    
    Returns:
        Dictionary of ETD4RK coefficients
    """
    N = len(L)
    r = 1.0  # Contour radius
    
    # FULL circle: θ from 0 to 2π
    theta = 2 * jnp.pi * (jnp.arange(M) + 0.5) / M  # Equally spaced on [0, 2π]
    roots = r * jnp.exp(1j * theta)  # Points on circle
    
    # For each eigenvalue L[j], evaluate on circle centered at h*L[j]
    LR = (h * L)[:, jnp.newaxis] + roots[jnp.newaxis, :]  # Shape: (N, M)
    
    # Compute exponentials
    exp_LR = jnp.exp(LR)
    exp_LR_half = jnp.exp(LR / 2)
    
    # Compute coefficients via contour integration
    # The mean over the circle approximates the contour integral
    
    # Q: (e^(z/2) - 1) / z
    Q = h * jnp.mean((exp_LR_half - 1) / LR, axis=1)
    
    # f1: [−4 − z + e^z(4 − 3z + z²)] / z³
    f1 = h * jnp.mean((-4 - LR + exp_LR * (4 - 3*LR + LR**2)) / LR**3, axis=1)
    
    # f2: [2 + z + e^z(−2 + z)] / z³
    f2 = h * jnp.mean((2 + LR + exp_LR * (-2 + LR)) / LR**3, axis=1)
    
    # f3: [−4 − 3z − z² + e^z(4 − z)] / z³
    f3 = h * jnp.mean((-4 - 3*LR - LR**2 + exp_LR * (4 - LR)) / LR**3, axis=1)
    
    # Exponential propagators
    E = jnp.exp(h * L)
    E_2 = jnp.exp(h * L / 2)
    
    # For real PDEs, the coefficients should be real (or have negligible imaginary parts)
    # But we keep them complex for generality
    
    return {
        'E': E,
        'E_2': E_2,
        'Q': Q,
        'f1': f1,
        'f2': f2,
        'f3': f3
    }


def step_ETDRK_control(state,control, coeffs, nonlinear_term):
    """
    Advance solution by one time step using ETD4RK. Assuming zero-hold control
    for the duration of the time step.
    
    Implements Kassam & Trefethen (2005) / Cox & Matthews (2002):
    u_n: state
    
    a_n = e^(Lh/2) u_n + L^(-1)(e^(Lh/2) - I)N(u_n)
    b_n = e^(Lh/2) u_n + L^(-1)(e^(Lh/2) - I)N(a_n)
    c_n = e^(Lh/2) a_n + L^(-1)(e^(Lh/2) - I)(2N(b_n) - N(u_n))
    u_{n+1} = e^(Lh) u_n + [f1*N(u_n) + 2*f2*(N(a_n) + N(b_n)) + f3*N(c_n)]
    
    where f1, f2, f3 are computed via contour integrals to avoid cancellation.
    
    Args:
        state: Current solution in physical space
        control: Control input to be applied during the time step
        coeffs: Precomputed ETD4RK coefficients
        nonlinear_term: Function to compute the nonlinear term N(u, control)
    Returns:
        Solution at next time step
    """
    E = coeffs['E']
    E_2 = coeffs['E_2']
    Q = coeffs['Q']
    f1 = coeffs['f1']
    f2 = coeffs['f2']
    f3 = coeffs['f3']
    
    # Transform to Fourier space
    state_hat = jnp.fft.fft(state)
    # DEALIAS!
    
    # To-do: de-alias after each FFT (of the nonlinear terms)!
    # NOTE: We might be doing an extra unncessary FFT/IFFT pair.
    # if this function just always took in a state_hat, we wouldn't
    # have to do the first and second FFT.
    
    # Stage 1: a_n
    N_state = nonlinear_term(state, control)
    N_state_hat = jnp.fft.fft(N_state)
    # DEALIAS!

    a_n_hat = E_2 * state_hat + Q * N_state_hat
    a_n = jnp.fft.ifft(a_n_hat)
    
    # Stage 2: b_n
    N_a = nonlinear_term(a_n, control)
    N_a_hat = jnp.fft.fft(N_a)
    # DEALIAS!
    
    b_n_hat = E_2 * state_hat + Q * N_a_hat
    b_n = jnp.fft.ifft(b_n_hat)
    
    # Stage 3: c_n
    N_b = nonlinear_term(b_n, control)
    N_b_hat = jnp.fft.fft(N_b)
    # DEALIAS!
    
    c_n_hat = E_2 * a_n_hat + Q * (2*N_b_hat - N_state_hat)
    c_n = jnp.fft.ifft(c_n_hat)
    
    # Stage 4: state_{n+1}
    N_c = nonlinear_term(c_n, control)
    N_c_hat = jnp.fft.fft(N_c)
    # DEALIAS!
    
    # ETD4RK update formula
    state_next_hat = E * state_hat + f1 * N_state_hat + 2 * f2 * (N_a_hat + N_b_hat) + f3 * N_c_hat

    state_next = jnp.fft.ifft(state_next_hat)

    return state_next




def step_ETDRK(state, coeffs, nonlinear_term):
    """
    Advance solution by one time step using ETD4RK. Assuming any control is
    handled within the nonlinear_term function.
    
    Implements Kassam & Trefethen (2005) / Cox & Matthews (2002):
    u_n: state
    
    a_n = e^(Lh/2) u_n + L^(-1)(e^(Lh/2) - I)N(u_n)
    b_n = e^(Lh/2) u_n + L^(-1)(e^(Lh/2) - I)N(a_n)
    c_n = e^(Lh/2) a_n + L^(-1)(e^(Lh/2) - I)(2N(b_n) - N(u_n))
    u_{n+1} = e^(Lh) u_n + [f1*N(u_n) + 2*f2*(N(a_n) + N(b_n)) + f3*N(c_n)]
    
    where f1, f2, f3 are computed via contour integrals to avoid cancellation.
    
    Args:
        state: Current solution in FFT space
        coeffs: Precomputed ETD4RK coefficients
        nonlinear_term: Function to compute the nonlinear term N(u)
        dealias_mask: dealias mask for removing high-freq wave numbers
    Returns:
        Solution at next time step
    """
    # TO-DO: could implement rfft for all real-valued PDEs + custom contour integration
    
    E = coeffs['E']
    E_2 = coeffs['E_2']
    Q = coeffs['Q']
    f1 = coeffs['f1']
    f2 = coeffs['f2']
    f3 = coeffs['f3']
    
    # Stage 1: a_n
    N_state = nonlinear_term(state)
    a_n = E_2 * state + Q * N_state
    
    # Stage 2: b_n
    N_a = nonlinear_term(a_n)
    b_n = E_2 * state + Q * N_a
    
    # Stage 3: c_n
    N_b = nonlinear_term(b_n)
    
    c_n = E_2 * a_n + Q * (2*N_b - N_state)
    
    # Stage 4: state_{n+1}
    N_c = nonlinear_term(c_n)
    
    # ETD4RK update formula
    state_next = E * state + f1 * N_state + 2 * f2 * (N_a + N_b) + f3 * N_c
    
    return state_next

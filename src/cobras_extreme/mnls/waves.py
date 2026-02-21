import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from cobras_extreme.mnls.solver_step import compute_etd_coefficients_contour, step_ETDRK




class PseudoSpectralSolver1DBase:
    def __init__(self, N_grid, L_domain, dt, M_contour=32, dealias_frac=0.0, real = False):
        self.N_grid = N_grid
        self.L_domain = L_domain
        self.dt = dt
        self.M_contour = M_contour
        self.dealias_frac = dealias_frac    # fraction (in [0,1]) of wavenumbers to mask
                                            # 0: use all wavenumbers, 1: remove all wavenumbers
        self.real = real
        # Spatial grid
        self.x = jnp.linspace(0, L_domain, N_grid, endpoint=False)
        self.dx = self.L_domain / self.N_grid
        
        # Wavenumbers for FFT (ordinary frequency convention)
        # k = n/L for n = 0, 1, ..., N/2-1, -N/2, ..., -1
        # This removes the factor of 2π from the definition

        self.k = jnp.fft.fftfreq(self.N_grid, d=self.dx)
        self.k2pi = 2 * jnp.pi * self.k
        self.Dx_hat = 1j * self.k2pi  # First derivative operator in Fourier space

        self.control = 0.0 * self.x  # Default control is zero
        self._init_linear_operator()
        
        self.dealias_frac = dealias_frac
        max_k = jnp.abs(self.k).max()
        
        self.dealias_mask = jnp.abs(self.k) <= (1 - dealias_frac) * max_k
        
        # Precompute ETD4RK coefficients using contour integration
        self.coeffs = compute_etd_coefficients_contour(self.L_hat, dt, M=M_contour)
        
        self._init_step()

    def _dealias(self, term_hat):
        return term_hat * self.dealias_mask
            
    def _init_linear_operator(self):
        '''Placeholder for linear operator initialization; override in subclass if needed'''
        pass

    def nonlinear_term(self, u):
        '''Placeholder for nonlinear term; override in subclass'''
        pass
    
    def _init_step(self):
        if self.real:
            # step function that returns real output
            # TO-DO: could implement rfft for all real-valued PDEs + custom contour integration
            step_fn = lambda state_hat: step_ETDRK(state_hat, self.coeffs, self.nonlinear_term)
        else:
            step_fn = lambda state_hat: step_ETDRK(state_hat, self.coeffs, self.nonlinear_term)
        self.step = jit(step_fn)
        
    def set_control(self, control):
        self.control = control
        self._init_step()



class KS1D(PseudoSpectralSolver1DBase):
    """
    Kuramoto-Sivashinsky equation solver using ETD4RK method.
    
    Solves equation:
    ∂u/∂t + u∂u/∂x + ∂²u/∂x² + ∂⁴u/∂x⁴ = 0
    """
    def __init__(self, N_grid, L_domain, dt, M_contour=32, dealias_frac = 0.0):
        super().__init__(N_grid, L_domain, dt, M_contour,dealias_frac=dealias_frac, real=True)
    
    def _init_linear_operator(self):
        # Linear operator in Fourier space
        # From equation: ∂u/∂t + u∂u/∂x + ∂²u/∂x² + ∂⁴u/∂x⁴ = 0
        # Rearranging: ∂u/∂t = -u∂u/∂x - ∂²u/∂x² - ∂⁴u/∂x⁴ + N(u)
        # In Fourier space with ordinary frequency: ∂û/∂t = L̂û + N̂(u)
        # With ∂/∂x → i(2πk) in Fourier space:
        # L̂ = -(2πki)² + (2πki)⁴
        self.L_hat = -(self.Dx_hat)**2 - (self.Dx_hat)**4

    def nonlinear_term(self, u_hat):
        # Nonlinear term: N(u) = -u∂u/∂x
        u = jnp.fft.ifft(u_hat).real
        u_x = jnp.fft.ifft(self.Dx_hat * u_hat).real
        
        physical_term = -u * u_x + self.control
        
        # TO-DO: dealias
        return jnp.fft.fft(physical_term)
    
    
   
class KdV1D(PseudoSpectralSolver1DBase):
    """
    Korteweg-de Vries equation solver using ETD4RK method.
    
    Solves equation:
    ∂u/∂t + u∂u/∂x + ∂³u/∂x³ = 0
    """
    def __init__(self, N_grid, L_domain, dt, M_contour=32, dealias_frac = 0.0, interaction_coef = 6.0):
        self.interaction_coef = interaction_coef
        super().__init__(N_grid, L_domain, dt, M_contour,dealias_frac=dealias_frac, real=True)
        
    def _init_linear_operator(self):
        # Linear operator in Fourier space
        # From equation: ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0
        # Rearranging: ∂u/∂t = -6u∂u/∂x - ∂³u/∂x³ + N(u)
        # In Fourier space with ordinary frequency: ∂û/∂t = L̂û + N̂(u)
        # With ∂/∂x → i(2πk) in Fourier space:
        # L̂ = -(2πki)³
        self.L_hat = - (self.Dx_hat)**3

    def nonlinear_term(self, u_hat):
        # Nonlinear term: N(u) = -u∂u/∂x
        u = jnp.fft.ifft(u_hat).real
        u_x = jnp.fft.ifft(self.Dx_hat * u_hat).real
        physical_term = - self.interaction_coef * u * u_x + self.control
        
        #TO-DO: Dealias
        return jnp.fft.fft(physical_term)
    
class Burgers1D(PseudoSpectralSolver1DBase):
    """
    Burgers' equation solver using ETD4RK method.

    Solves equation:
    ∂u/∂t + u∂u/∂x - eps ∂²u/∂x² = 0
    """
    def __init__(self, N_grid, L_domain, dt, M_contour=32, dealias_frac = 0.0, eps = 0.03):
        self.eps = eps
        super().__init__(N_grid, L_domain, dt, M_contour,dealias_frac=dealias_frac, real=True)

    def _init_linear_operator(self):
        # Linear operator in Fourier space
        # From equation: ∂u/∂t = -u∂u/∂x + eps ∂²u/∂x²
        # In Fourier space with ordinary frequency: ∂û/∂t = L̂û + N̂(u)
        # With ∂/∂x → i(2πk) in Fourier space:
        # L̂ = eps (2πki)²
        self.L_hat = self.eps * (self.Dx_hat)**2

    def nonlinear_term(self, u_hat):
        # Nonlinear term: N(u) = -u∂u/∂x
        u = jnp.fft.ifft(u_hat).real
        u_x = jnp.fft.ifft(self.Dx_hat * u_hat).real
        
        physical_term = - u * u_x + self.control
        
        return jnp.fft.fft(physical_term)

class NLS1D(PseudoSpectralSolver1DBase):
    '''MNLS from Cousins & Sapsis (2016)'''
    def _init_linear_operator(self):
        # Linear operator in Fourier space
        # From equation (2.1): ∂u/∂t + (1/2)∂u/∂x + (i/8)∂²u/∂x² - (1/16)∂³u/∂x³ + N(u) = 0
        # Rearranging: ∂u/∂t = -(1/2)∂u/∂x - (i/8)∂²u/∂x² + (1/16)∂³u/∂x³ + N(u)
        self.L_hat = -(1/2.0)*self.Dx_hat -(1j/8.0)*(self.Dx_hat)**2

    def nonlinear_term(self, u_hat):
        """
        Compute nonlinear terms N(u) in physical space.
        
        N(u) = -(i/2)|u|²u
        
        Args:
            u: Complex wave envelope in physical space
        
        Returns:
            Nonlinear term in physical space
        """        
        # Nonlinear terms
        u = jnp.fft.ifft(u_hat)
        u_abs_sq = jnp.abs(u)**2
        
        N_u = -(1j/2) * u_abs_sq * u
        physical_term = N_u + self.control
        # TO-DO: dealias
        
        return jnp.fft.fft(physical_term)


class MNLS1D(PseudoSpectralSolver1DBase):
    '''MNLS from Cousins & Sapsis (2016)'''
    def _init_linear_operator(self):
        # Linear operator in Fourier space
        # From equation (2.2): ∂u/∂t + (1/2)∂u/∂x + (i/8)∂²u/∂x² - (1/16)∂³u/∂x³ + N(u) = 0
        # Rearranging: ∂u/∂t = -(1/2)∂u/∂x - (i/8)∂²u/∂x² + (1/16)∂³u/∂x³ - N(u)
        self.L_hat = -(1/2.0)*self.Dx_hat -(1j/8.0)*(self.Dx_hat)**2 + (1/16) * (self.Dx_hat) **3
        
    def velocity_potential_term(self, u):
        """
        Compute ∂φ/∂x|_{z=0} = -ℱ⁻¹[|k|ℱ[|u|²]]/2
        
        Note: With ordinary frequency convention, |k| in Fourier space
        corresponds to |k|×2π in angular frequency.
        
        Args:
            u: Complex wave envelope in physical space
        
        Returns:
            Real-valued velocity potential derivative
        """
        u_abs_sq = jnp.abs(u)**2
        u_abs_sq_hat = jnp.fft.fft(u_abs_sq)
        u_abs_sq_hat = self._dealias(u_abs_sq_hat)
        
        # Multiply by |k| (with 2π factor for ordinary frequency)
        dphi_dx_hat = -jnp.pi * jnp.abs(self.k) * u_abs_sq_hat
        
        # Transform back to physical space
        dphi_dx = jnp.fft.ifft(dphi_dx_hat)
        
        return dphi_dx

    def nonlinear_term(self, u_hat):
        """
        Compute nonlinear terms N(u) in physical space.
        
        N(u) = -(i/2)|u|²u - (3/2)|u|²∂u/∂x - (1/4)u²∂u*/∂x - iu∂φ/∂x|_{z=0}
        
        Args:
            u: Complex wave envelope in physical space
        
        Returns:
            Nonlinear term in physical space
        """
        # Compute derivatives in Fourier space (ordinary frequency)
        
        # ∂/∂x corresponds to multiplication by i(2πk)
        du_dx_hat = self.Dx_hat * u_hat
        du_dx = jnp.fft.ifft(du_dx_hat)
        
        du_star_dx = jnp.conj(du_dx)
        
        # Nonlinear terms
        u = jnp.fft.ifft(u_hat)
        u_abs_sq = jnp.abs(u)**2
        
        term1 = -(1j/2) * u_abs_sq * u
        term2 = -(3/2) * u_abs_sq * du_dx
        term3 = -(1/4) * u**2 * du_star_dx
        
        # Velocity potential term
        dphi_dx = self.velocity_potential_term(u)
        term4 = -1j * u * dphi_dx
        
        
        # individually dealias each of the nonlinear terms
        term1_hat = self._dealias(jnp.fft.fft(term1))
        term2_hat = self._dealias(jnp.fft.fft(term2))
        term3_hat = self._dealias(jnp.fft.fft(term3))
        term4_hat = self._dealias(jnp.fft.fft(term4))
        
        N_u_hat = term1_hat + term2_hat + term3_hat + term4_hat
        
        control_hat = jnp.fft.fft(self.control)
        return N_u_hat + control_hat


class MMT(PseudoSpectralSolver1DBase):
    '''MMT from Cousins & Sapsis (2014) [1]
    
        i∂u/∂t = |∂_x|^α u + λ |∂_x|^{−β/4} (||∂_x|^{−β/4} u|^2 |∂_x|^{−β/4} u) + iDu
        
        where 
        \hat{|∂_x|^α u}(k) = |k|^{\alpha} \hat{u}(k)
        
        and
            \hat (Du)(k) = -(|k| - k_{crit})^2 \hat u(k), |k| > k_crit
            and 0 otherwise
        is a selective Laplacian Operator. 
    
        We specifically look at the case where β=0,
        
        i∂u/∂t = |∂_x|^α u + λ  (|u|^2 u) + iDu
    
    [1] Cousins, Will, and Themistoklis P. Sapsis. 
        "Quantification and prediction of extreme events in a 
        one-dimensional nonlinear dispersive wave model." 
        Physica D: Nonlinear Phenomena 280 (2014): 48-58.
        
    '''
    
    def __init__(self, N_grid, L_domain, dt, M_contour=32, dealias_frac=0.0, 
                 lambda_coef = -4, alpha = 0.5, k_crit = 500):
        self.lambda_coef = lambda_coef
        self.alpha = alpha
        self.beta = 0.0 # just for clarity
        self.k_crit = k_crit
        super().__init__(N_grid, L_domain, dt, M_contour,dealias_frac=dealias_frac, real=False)

    def _init_linear_operator(self):
        '''Linear operator in Fourier space. 
        
        Rearranging: ∂u/∂t = -i|∂_x|^α u  - iλ (|u|^2 u) + Du
        
        The only linear piece is the selective laplace term: 
            \hat (Du)(k) = -(|k| - k_{crit})^2 \hat u(k),   |k| > k_crit
                                                            and 0 otherwise
        '''

        k_mask = jnp.abs(self.k2pi) > self.k_crit
        self.L_hat = -(1/2.0)*self.Dx_hat -(1j/8.0)*(self.Dx_hat)**2 + (1/16) * (self.Dx_hat) **3
        

    def nonlinear_term(self, u_hat):
        """
        Compute nonlinear terms N(u) in physical space.
        
        N(u) = -i|∂_x|^α u  - iλ (|u|^2 u)
        \hat N(u) = -i |k|^{α} - i \lambda F(|u|^2 u) 
        
        Args:
            u: Complex wave envelope in physical space
        
        Returns:
            Nonlinear term in physical space
        """
        # Compute derivatives in Fourier space (ordinary frequency)
        u = jnp.fft.ifft(u_hat)
        dx_alpha_hat = jnp.abs(self.Dx_hat)**self.alpha
        term1_hat =  (-1j) * dx_alpha_hat *  u_hat
        
        # TO-DO: dealias
        term1 = jnp.fft.ifft(term1_hat)
        
        # Nonlinear terms
        u_abs_sq = jnp.abs(u)**2
        
        term2 = -(1j) * self.lambda_coef * u_abs_sq * u
        
        N_u = term1 + term2
        
        
        # TO-DO: dealias. Can probably just be done
        # in the step_fn
        
        return N_u + self.control
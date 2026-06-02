import jax
import jax.numpy as jnp
from jax import jit, vjp
from jax import grad, value_and_grad, vmap
import argparse

import cobras_extreme
from solvers import transient
from equations.flow import FlowConfig 
import equations.base as base
import equations.utils as utils 


def get_energy_wrapper(kx, ky, nu, res):
    def energy_wrapper_fft(omega_hat):
        return utils.compute_energy_dissipation(omega_hat, nu,)
    
    energy_wrapper_fft = vmap(energy_wrapper_fft, in_axes=(0,))
    return energy_wrapper_fft


def sim(vorticity_hat0, flow, end_time, save_time=0.25, dt=1e-3):
    # Underlying equation
    equation = base.PseudoSpectralNavierStokes2D(flow)
    
    # Timing
    # Save every x seconds 
    total_steps = int(end_time // dt)
    step_to_save = int(save_time // dt) 

    step_fn = transient.RK4_CN(equation, dt)
    _, trajectory = transient.iterative_func(step_fn, vorticity_hat0, total_steps, step_to_save)
    return trajectory

def get_e_traj_wrapper(flow, end_time, save_time=0.25, dt=1e-3):
    kx, ky = flow.create_fft_mesh()
    nu = flow.nu
    n = flow.grid_size[0]
    
    energy_wrapper_fft = get_energy_wrapper(kx, ky, nu, n)

    @jit
    def wrapper(snapshot):
        vorticity_hat0 = jnp.fft.rfftn(snapshot)

        # # Underlying equation
        # equation = base.PseudoSpectralNavierStokes2D(flow)
        # # Timing
        # # Save every x seconds 
        
        # total_steps = int(end_time // dt)
        # step_to_save = int(save_time // dt) 

        # step_fn = transient.RK4_CN(equation, dt)
        # _, trajectory = transient.iterative_func(step_fn, vorticity_hat0, total_steps, step_to_save)

        trajectory = sim(vorticity_hat0, flow, end_time, save_time=save_time, dt=dt)
        e_traj = energy_wrapper_fft(trajectory)
        return e_traj
    
    return wrapper

def forward_parser(*args):
    parser = argparse.ArgumentParser('Simulate Kolmogorov Flow and save data.')
    
    parser.add_argument(
            '--Re',
            help='Reynolds number',
            default=40,
            type=float
    )
    
    parser.add_argument(
            '--k',
            help='Forcing wavenumber',
            default=4,
            type=int
    )
    
    parser.add_argument(
            '--res',
            help='Grid resolution',
            default=256,
            type=int
    )

    parser.add_argument(
            '--save_dt',
            help='dt to save intermediate snapshots',
            default=0.25,
            type=float
    )
    
    parser.add_argument(
            '--dt',
            help='simulation dt',
            default=1.0e-3,
            type=float
    )
    
    parser.add_argument(
            '--end_time',
            help='End Time',
            default=5000,
            type=float
    )
    
    parser.add_argument(
            '--flow',
            help='Flow Name',
            default='classic',
            type=str
    )
    

    return parser


def backward_parser(*args):
    parser = argparse.ArgumentParser('Simulate Kolmogorov Flow and save data.')
    
    parser.add_argument(
            '--Re',
            help='Reynolds number',
            default=40,
            type=float
    )
    
    parser.add_argument(
            '--k',
            help='Forcing wavenumber',
            default=4,
            type=int
    )
    
    parser.add_argument(
            '--res',
            help='Grid resolution',
            default=256,
            type=int
    )

    parser.add_argument(
            '--save_dt_load',
            help='intermediate dt from saved snapshots.',
            default=0.5,
            type=float
    )
    
    parser.add_argument(
            '--save_dt_back',
            help='dt to save from flow map',
            default=0.5,
            type=float
    )
    
    parser.add_argument(
            '--end_time',
            help='End Time',
            default=5000,
            type=float
    )
    
    parser.add_argument(
            '--forecast',
            help='amount of time to forecast into the future',
            default=8,
            type=int
    )
    
    parser.add_argument(
            '--dt',
            help='simulation dt',
            default=1.0e-3,
            type=float
    )
    
    parser.add_argument(
            '--n_grads',
            help='number of gradients',
            default=5000,
            type=int
    )
    
    parser.add_argument(
            '--batch_size',
            help='number of gradients to take per batch',
            default=8,
            type=int
    )
    
    parser.add_argument(
            '--flow',
            help='Flow Name',
            default='classic',
            type=str
    )
    
    parser.add_argument(
            '--qoi',
            help='Quantity of interest',
            default='dissipation',
            type=str
    )
    
    parser.add_argument('--continue',
                        help='continue where left off',
                        default = False,
                        type = bool)
    

    return parser
    
def get_flow(config):
    n = config['res']

    # Physical parameters 
    flow_config = {'k': config['k'], 
                   'Re': config['Re'], 
                   'grid_size': (n,n)
                   }

    # select flow
    if config['flow'] == 'classic': 
        flow = FlowConfig(**flow_config)
    else:
        raise NotImplementedError
    return flow


def get_forward_fn(flow, qoi, T_f, dt, save_dt):
    if qoi == 'dissipation': 
        wrapper = get_e_traj_wrapper(flow, 
                                     T_f, save_time=save_dt, 
                                     dt=dt
                                     )
    else:
        return NotImplementedError
    
    return wrapper

def get_backwards_fn(forward_fn):

    def vjp_eval(d):
        y, vjp_fn = vjp(forward_fn, d['snap'])
        grad_sample = vjp_fn(d['cotangent'])[0]
        return grad_sample
    
    return vjp_eval
        
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cobras_extreme.kflow.data_utils import forward_parser, get_flow

import equations.base as base
from solvers import transient


from cobras_extreme import _kflow_dir, _data_dir
root_dir = _kflow_dir

if __name__ == '__main__': 
    # parse args and convert to dict
    parser = forward_parser()
    args = parser.parse_args()
    config = vars(args) # to dict
    
    flow = get_flow(config)
    n = flow.grid_size[0]
    k = config['k']
    
    end_time = config['end_time']
    dt = config['dt']
    save_time = config['save_dt']

    save_name = f'kolmogorov_n={int(n)}-Re={int(flow.Re)}_k={k}_end={int(end_time)}_save={save_time}'

    save_path = os.path.join(_data_dir, 'forward', save_name)
    print("Saving to: ", save_path)

    vorticity_hat0 = flow.initialize_state()
    
    # Underlying equation
    equation = base.PseudoSpectralNavierStokes2D(flow)

    total_steps = int(end_time // dt)
    step_to_save = int(save_time // dt) 

    step_fn = transient.RK4_CN(equation, dt)
    
    t0 = time.time()
    _, trajectory = transient.iterative_func(step_fn, vorticity_hat0, total_steps, step_to_save)
    
    print('Time:', time.time() - t0)
    jnp.save(save_path, trajectory)

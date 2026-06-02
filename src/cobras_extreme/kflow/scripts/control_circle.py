import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tqdm import tqdm

from jax import numpy as jnp
from jax import lax

import glob
import numpy as np


import cobras_extreme
from cobras_extreme import _data_dir, _kflow_data_dir
from cobras_extreme.kflow.load_utils import load_projs


from equations.utils import compute_velocity_fft
from equations.flow import FlowConfig
from equations import base
from solvers import transient

res=256
Re=40
Tf=4

# cobras or pod
MODES = 'pod' 

flow = FlowConfig(grid_size = (res,res), Re = Re)
kx, ky = flow.create_fft_mesh()

print('loading data...')
# load snap data
snap_hat_data = np.load(f'/home/nzolman/data/projects/cobras-kflow/data/forward/kolmogorov_n={res}-Re={Re}_k=4_end=5000_save=0.5.npy')
print('data loaded')

save_dir = os.path.join(_kflow_data_dir, 'ctrl', 'test')
os.makedirs(save_dir, exist_ok=True)

init_idxes = jnp.array([8200])

data = load_projs(Re=Re, Tf=Tf, res=res, return_snaps=True, 
                  snap_idx=init_idxes, return_real=True)

if MODES == 'cobras':
    Phi = data['Phi']
    Psi = data['Psi']
    z_data = data['z_cobras_all'][:4000, :2]
elif MODES == 'pod':
    Phi = data['pod_Psi']
    Psi = data['pod_Psi']
    z_data = data['z_pod_all'][:4000, :2]
    
    
print('Modes:', MODES)
print('z_data shape:', z_data.shape)

amps = np.linalg.norm(z_data, axis=-1)

R_REF = np.median(amps)
print('Reference radius:', R_REF)

K_GAIN = 1.0

x_mesh, y_mesh = flow.create_mesh()

kx, ky = flow.create_fft_mesh()
nu = flow.nu

def control(state, r_ref = 1.0, k_gain = 1):
    z = Psi.T[:2] @ state.flatten()
    
    r = jnp.linalg.norm(z)
    theta = jnp.atan2(z[1], z[0])
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    dr = k_gain * (r_ref - r) * jnp.array([cos_theta, sin_theta])
    omega_control = (Phi[:,:2] @ dr).reshape(state.shape)
    
    # return omega_control
    omega_hat_control = jnp.fft.rfft2(omega_control)
    uhat_control, vhat_control = compute_velocity_fft(omega_hat_control, kx, ky)
    
    u_control = jnp.fft.irfft2(uhat_control)
    v_control = jnp.fft.irfft2(vhat_control)
    
    return  u_control, v_control



end_time = 1        # amount of time to control the flow
save_time = 0.5     # inbetween steps to save
dt = 1e-3           # simulation dt

total_steps = int(end_time // dt)
step_to_save = int(save_time // dt) 

control_len = 1000

ckpt_freq = 50

for i, snap_idx in enumerate(init_idxes):
    print(f'starting {snap_idx}')
    all_trajs = []
    save_path = os.path.join(save_dir,
                            f't0={snap_idx}_{control_len}-gain={K_GAIN:.2f}_{MODES}.npy'
                            )

    state = data['snaps'][i]
    
    for i in tqdm(range(control_len)):
        
        # compute control
        u_ctrl =  control(state, r_ref = R_REF, k_gain= K_GAIN)
        
        # pass control to solver
        flow.control_function = u_ctrl
        
        # setup sim
        equation = base.PseudoSpectralNavierStokes2D(flow)
        step_fn = transient.RK4_CN(equation, dt)
        vorticity_hat0 = jnp.fft.rfftn(state)
        
        # output control response
        _, trajectory = transient.iterative_func(step_fn, vorticity_hat0, total_steps, step_to_save)

        all_trajs.append(trajectory)
        
        state = jnp.fft.irfft2(trajectory[-1])
        if (i % ckpt_freq) == ckpt_freq - 1:
            jnp.save(save_path, all_trajs)
            print(f'saving {i}.\n', save_path)

    jnp.save(save_path, all_trajs)
    print('done.\n', save_path)
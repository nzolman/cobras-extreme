import os
import jax
jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import random

from jax import vjp
from tqdm import tqdm

key = random.PRNGKey(0)
key_init, key_cotangents = random.split(key,2)

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
from diffrax import TqdmProgressMeter

from cobras_extreme import _fhn_data_dir
from cobras_extreme.fhn.fhn import setup_FCN

N = 10001

# LOAD_PATH = None
LOAD_PATH = os.path.join(_fhn_data_dir, 
                      f'fhn_{N}_forward_long.npy')


progress_meter = TqdmProgressMeter(refresh_steps = 100)


f_diffrax = setup_FCN(N=N)

x0_strength = 0.5
x0 = random.uniform(key_init, shape=(2*N,), minval=-1, maxval=1) * x0_strength
# y0 = random.uniform(key_y, shape=(N,)) * y0_strength
solver = Dopri5()

term = ODETerm(f_diffrax)
tol = 1e-9

# stepsize_controller = ConstantStepSize()

Tf = int(1e5)
maxdt0 = 5e-1
dt0 = 1e-2

stepsize_controller = PIDController(rtol=tol, atol=tol, dtmax=maxdt0, dtmin=1e-5)
base_max_steps = int(Tf/dt0)
max_steps = 10*base_max_steps

if (LOAD_PATH is None) or (not os.path.exists(LOAD_PATH)):
    # Gen forward data
    print('Starting forward solve...')
    saveat = SaveAt(ts=jnp.arange(0, Tf))
    sol = diffeqsolve(term, solver, t0=0, t1=Tf, 
                    dt0=dt0, y0=x0, saveat=saveat, 
                    stepsize_controller=stepsize_controller, max_steps=max_steps, 
                    progress_meter=progress_meter)

    xs = sol.ys
    ts = sol.ts

    print('Saving forward solve...')
    jnp.save(os.path.join(_fhn_data_dir, 
                        f'fhn_{N}_forward_long.npy'), 
            xs)

else:
    print('Loading forward data...')
    xs = jnp.load(LOAD_PATH)

# setup forward + backward data
Tf_fwd = 100
Tf_dataset = int(1e5)

save_dt = 1
f_diffrax = setup_FCN(N, k=0.128)

saveat_fwd = SaveAt(ts=jnp.arange(0, Tf_fwd, save_dt))

from jax import vmap

# qoi at time t
def qoi_t(x):
    x_std = x[:N].std()
    y_std = x[N:].std()
    
    return jnp.mean(x**2)
    # return x_std**2 + y_std**2

qoi_v = vmap(qoi_t, in_axes=(0,))

solver = Dopri5()
term = ODETerm(f_diffrax)
tol = 1e-9

Tf = int(1e5)
maxdt0 = 5e-1
dt0 = 1e-2

stepsize_controller = PIDController(rtol=tol, atol=tol, dtmax=maxdt0, dtmin=1e-5)
base_max_steps = int(Tf/dt0)
max_steps = 10*base_max_steps

def fwd_diffrax(x0):
    sol = diffeqsolve(term, solver, t0=0, t1=Tf_fwd, 
                  dt0=dt0, y0=x0, saveat=saveat_fwd, 
                  stepsize_controller=stepsize_controller, max_steps=max_steps)
    return sol.ys

def qoi_diffrax(x0):
    return qoi_v(fwd_diffrax(x0))


def generate_data_diffrax(key, Tf, save_dt=1, x0_strength=0.5):
    
    saveat_data = SaveAt(ts=jnp.arange(0, Tf, save_dt))
    x0 = random.uniform(key, shape=(2*N,), 
                        minval=-x0_strength, 
                        maxval=x0_strength
                        )
    
    sol = diffeqsolve(term, solver, t0=0, t1=Tf, 
                  dt0=dt0, y0=x0, saveat=saveat_data, 
                  stepsize_controller=stepsize_controller, 
                  max_steps=max_steps)
    return sol.ys




def get_jvp(x0, cotangent):
    y, vjp_fn = vjp(qoi_diffrax, x0)
    grad_sample = vjp_fn(cotangent)[0]
    return grad_sample

def get_jvp_lax(d):
    x0, cotangent = d
    return get_jvp(x0, cotangent)

get_jvp_v = vmap(get_jvp, in_axes=(0, 0))


num_samples = len(xs)//10
output_dim = Tf_fwd
cotangets = random.normal(key_cotangents, shape=(num_samples, output_dim))
key = random.split(key)[0]

all_x0s = xs[::len(xs)//num_samples]

batch_size = 5
n_batches = num_samples // batch_size


print('Starting backwards solves...')
all_grad_data = []
for i in tqdm(range(n_batches)):
    cots = cotangets[i*batch_size:(i+1)*batch_size]
    x0s = all_x0s[i*batch_size:(i+1)*batch_size]
    x0s = x0s.to_device(jax.devices()[0]) # put on the correct device
    grad_data = get_jvp_v(x0s, cots)
    # grad_data = lax.map(get_jvp_lax, (x0s, cots))
    all_grad_data.append(grad_data)
    
all_grad_data = jnp.array(all_grad_data)
all_grad_data = jnp.concatenate(all_grad_data)


print('Saving backwards solves...')
jnp.save(os.path.join(_fhn_data_dir, 
                    #   f'fhn_{N}_grad_std_Tf={Tf_fwd}.npy'), 
                        f'fhn_{N}_grad_square_Tf={Tf_fwd}.npy'), 
         all_grad_data)
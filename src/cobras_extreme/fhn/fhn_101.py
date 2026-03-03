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

def setup_fwd(N, k=0.128):

    a = -0.02651
    c= 0.02 

    ns = jnp.arange(N)

    bs = 0.006 + ns/(N-1)*0.008 
    K =  k / (N-1)


    def f(z): 
        x = z[:N]
        y = z[N:]
        
        dx = x*(a-x)*(x-1) - y + K*(x.sum() - N*x)
        dy = bs*x - c*y
        return jnp.concatenate([dx, dy])

    def f_diffrax(t, y, args):
        return f(y)
    
    return f_diffrax



progress_meter = TqdmProgressMeter(refresh_steps = 100)

N = 101
f_diffrax = setup_fwd(N=N)

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


# Gen forward data
saveat = SaveAt(ts=jnp.arange(0, Tf))
sol = diffeqsolve(term, solver, t0=0, t1=Tf, 
                  dt0=dt0, y0=x0, saveat=saveat, 
                  stepsize_controller=stepsize_controller, max_steps=max_steps, 
                  progress_meter=progress_meter)

xs = sol.ys
ts = sol.ts



# setup forward + backward data
Tf_fwd = 200
Tf_dataset = int(1e5)

save_dt = 1
N = 101
f_diffrax = setup_fwd(N, k=0.128)


saveat_fwd = SaveAt(ts=jnp.arange(0, Tf_fwd, save_dt))


from jax import vmap


# qoi at time t
def qoi_t(x):
    return jnp.mean(x**2)

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

batch_size = 10
n_batches = num_samples // batch_size


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
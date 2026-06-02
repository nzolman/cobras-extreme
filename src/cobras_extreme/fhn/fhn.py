from jax import random, vmap, jit, vjp
from diffrax import ODETerm, SaveAt, diffeqsolve, Dopri5, PIDController
from diffrax import TqdmProgressMeter, NoProgressMeter
import jax.numpy as jnp
import jax
from tqdm import tqdm
from jax.experimental.sparse import BCOO # sparse matrix support

from jax import random
import networkx as nx
from jax import numpy as jnp
from sklearn.neighbors import kneighbors_graph

def qoi_t(x):
    # return jnp.std(x[:N])**2
    N = x.shape[-1]//2
    # return jnp.mean(x[:N])
    return jnp.mean(x**2, axis=-1)

qoi_v = vmap(qoi_t, in_axes=(0,))


def setup_FCN(N, k=0.128):
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

def setup_small_world(key_b, jax_A, n,m, k_nn=60, k=0.128):
        
    N = n*m

    a = -0.0276 
    c= 0.02 
    
    K =  k/k_nn
    
    bs = random.uniform(key_b, minval=6e-3, maxval=1.4e-2, shape = (N,))
    

    jax_a = jax_A.sum(axis=0).todense()
    def f(z): 
        x = z[:N]
        y = z[N:]
        
        dx = x*(a-x)*(x-1) - y + K*(jax_A @ x - jax_a * x)
        dy = bs*x - c*y
        return jnp.concatenate([dx, dy])

    def f_diffrax(t, y, args):
        return f(y)

    return f_diffrax
    
    
def get_data_gen(f_diffrax, N, x0_strength = 0.5, tol = 1e-9, 
                 maxdt0=1e-1, dt0=1e-2, save_every = 1, verbose=False, pid_coeffs={}):
    def gen_data(key_init, Tf):
        if verbose:
            progress_meter = TqdmProgressMeter(refresh_steps = 100)
        else:
            progress_meter = NoProgressMeter()
        x0 = random.uniform(key_init, shape=(2*N,), minval=-1, maxval=1) * x0_strength
        solver = Dopri5()
        term = ODETerm(f_diffrax)

        stepsize_controller = PIDController(rtol=tol, atol=tol, dtmax=maxdt0, dtmin=1e-5,
                                            **pid_coeffs)
        base_max_steps = int(Tf/dt0)
        max_steps = 10*base_max_steps
        
        saveat = SaveAt(ts=jnp.arange(0, Tf, save_every))
        
        sol = diffeqsolve(term, solver, t0=0, t1=Tf, 
                        dt0=dt0, y0=x0, saveat=saveat, 
                        stepsize_controller=stepsize_controller, max_steps=max_steps, 
                        progress_meter=progress_meter)
        return sol.ys
    return gen_data

def gen_data_FCN(key_init, N, Tf, save_every=1, verbose=False):
    f_diffrax = setup_FCN(N)
    gen_data = get_data_gen(f_diffrax, N, verbose=verbose, 
                            x0_strength = 0.5, tol = 1e-9, 
                            save_every=save_every,maxdt0=5e-1, dt0=1e-2)
    
    xs = gen_data(key_init, Tf)
    return xs

def load_jax_sparse(fname): 
    jax_dict = jnp.load(fname, allow_pickle=True).item()
    jax_A = BCOO((jax_dict['data'], jax_dict['indices']), shape=jax_dict['shape'])
    return jax_A


def base_lattice(n,m):
    th_1 = jnp.linspace(0, 2*jnp.pi, n, endpoint=False)
    th_2 = jnp.linspace(0, 2*jnp.pi, m, endpoint=False)
    th1_mesh, th2_mesh = jnp.meshgrid(th_1,th_2)
    return th1_mesh, th2_mesh

def base_lattice_pos(n,m):
    th1_mesh, th2_mesh = base_lattice(n,m)
    
    th1_flat = th1_mesh.flatten()
    th2_flat = th2_mesh.flatten()
    embedding = jnp.vstack([jnp.cos(th1_flat), jnp.sin(th1_flat), jnp.cos(th2_flat), jnp.sin(th2_flat)]).T
    return embedding


def get_knn_graph(n_points_per_side, k):
    pos_embeddings = base_lattice_pos(n_points_per_side,n_points_per_side)
    sparse_A = kneighbors_graph(pos_embeddings, k)
    G_knn = nx.from_scipy_sparse_array(sparse_A)
    return G_knn

def get_small_world_graph(key_topo, n_points_per_side, k, p):
    G_knn = get_knn_graph(n_points_per_side, k)
    n_E = G_knn.number_of_edges()
    n_V = G_knn.number_of_nodes()
        
    remove_bool = (random.uniform(key_topo,shape = (n_E,)) < p)
    remove_idx = jnp.argwhere(remove_bool).flatten()
        
    node_list = jnp.arange(n_V)
    clashes = 0
    from tqdm import tqdm
    G_small_world = G_knn.copy()
    edge_list = list(G_small_world.edges)
    for e_idx in tqdm(remove_idx):
        u,v = edge_list[e_idx]
        u_new, v_new = u, v
        clashes -= 1
        while G_small_world.has_edge(u_new,v_new):
            key_topo, _ = random.split(key_topo)
            u_new, v_new = random.choice(key_topo, node_list, shape=(2,), replace=False).tolist()
            clashes += 1
            
        G_small_world.remove_edge(u,v)
        G_small_world.add_edge(u_new,v_new)
        
    print('num clashes: ', clashes)
    return G_small_world


def get_grads(f_diffrax, xs, key_cot, Tf_fwd=100, save_dt = 1, 
              dt0 = 1e-3, dtmax = 5e-3, tol = 1e-12, skip_samples = 2, batch_size=10,
              batch_save_freq = 10, save_name = 'fhn_small_world_grads.npy'):
    output_dim = Tf_fwd // save_dt
    saveat_fwd = SaveAt(ts=jnp.arange(0, Tf_fwd, save_dt))
    solver = Dopri5()
    term = ODETerm(f_diffrax)

    stepsize_controller = PIDController(rtol=tol, atol=tol, dtmax=dtmax, dtmin=1e-5)
    
    def fwd_diffrax(x0):
        base_max_steps = int(Tf_fwd/dt0)
        max_steps = 10*base_max_steps
        sol = diffeqsolve(term, solver, t0=0, t1=Tf_fwd, 
                    dt0=dt0, y0=x0, saveat=saveat_fwd, 
                    stepsize_controller=stepsize_controller, max_steps=max_steps)
        return sol.ys

    def qoi_diffrax(x0):
        return qoi_v(fwd_diffrax(x0))

    def get_jvp(x0, cotangent):
        y, vjp_fn = vjp(qoi_diffrax, x0)
        grad_sample = vjp_fn(cotangent)[0]
        return grad_sample

    get_jvp_v = vmap(get_jvp, in_axes=(0, 0))

    num_samples = len(xs)//skip_samples
    cotangets = random.normal(key_cot, shape=(num_samples, output_dim))

    all_x0s = xs[::len(xs)//num_samples]
    n_batches = num_samples // batch_size


    all_grad_data = []
    for i in tqdm(range(n_batches)):
        cots = cotangets[i*batch_size:(i+1)*batch_size]
        x0s = all_x0s[i*batch_size:(i+1)*batch_size]
        x0s = x0s.to_device(jax.devices()[0]) # put on the correct device
        grad_data = get_jvp_v(x0s, cots)
        all_grad_data.append(grad_data)
        if i % batch_save_freq == batch_save_freq - 1:
            jnp.save(save_name, jnp.concatenate(all_grad_data))
        
    all_grad_data = jnp.array(all_grad_data)
    all_grad_data = jnp.concatenate(all_grad_data)
    return all_grad_data
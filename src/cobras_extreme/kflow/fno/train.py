"""
Fourier Neural Operator (FNO) surrogate for 2D periodic Kolmogorov Flow at Re=40.

Architecture: FNO-2D (Li et al. 2021, https://arxiv.org/abs/2010.08895)
Optimizer:    AdamW with cosine annealing + linear warmup (standard for FNO)
Loss:         Relative L2 (H1 optional) on vorticity snapshots

Usage:
    python train.py --data_path /path/to/snapshots.npy --T 4 --input_res 64

Data format:
    snapshots.npy  shape (N, H, W)  float32 vorticity fields,
                   equally spaced in time by 1 simulation time unit.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state



# ─────────────────────────────────────────────────────────────
# 0.  NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────
# All data stored on disk and passed through the public API is in
# PHYSICAL units (vorticity ω, s⁻¹).  Normalisation to zero-mean /
# unit-std happens only at the network boundary and is always undone
# before any metric or output is returned to the caller.

def compute_norm_stats(data: np.ndarray):
    """Compute global mean and std from a (N, H, W) physical array."""
    mu  = float(data.mean())
    sig = float(data.std()) + 1e-8
    return {'mean': mu, 'std': sig}

def normalise(x: np.ndarray, stats: dict) -> np.ndarray:
    """Physical → normalised.  Works on any shape."""
    return (x - stats['mean']) / stats['std']

def unnormalise(x: np.ndarray, stats: dict) -> np.ndarray:
    """Normalised → physical.  Works on any shape."""
    return x * stats['std'] + stats['mean']

def save_norm_stats(stats: dict, path):
    import json
    with open(path, 'w') as f:
        json.dump(stats, f)

def load_norm_stats(path) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 1.  MODEL
# ─────────────────────────────────────────────────────────────

class SpectralConv2D(nn.Module):
    """
    Fourier integral operator layer.
    Truncates to `modes` lowest-frequency modes in each spatial dimension.
    """
    out_channels: int
    modes1: int   # number of kept Fourier modes in dim 1
    modes2: int   # number of kept Fourier modes in dim 2

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C_in)
        B, H, W, C_in = x.shape
        # Convert traced dimensions to Python ints for static shapes
        H_int = int(H); W_int = int(W)
        C_out = self.out_channels
        m1, m2 = self.modes1, self.modes2
        m2h = m2 // 2 + 1   # half-spectrum width we keep

        # Learnable complex weights via real+imag parameterisation
        scale = 1.0 / (C_in * C_out)
        w_re = self.param('w_re', lambda k, s: jax.random.uniform(k, s) * scale,
                          (C_in, C_out, m1, m2h))
        w_im = self.param('w_im', lambda k, s: jax.random.uniform(k, s) * scale,
                          (C_in, C_out, m1, m2h))
        W = w_re + 1j * w_im   # (C_in, C_out, m1, m2h)

        # rfft2 along spatial dims
        x_t  = jnp.transpose(x, (0, 3, 1, 2))          # (B, C_in, H, W)
        x_ft = jnp.fft.rfft2(x_t, norm='ortho')         # (B, C_in, H, W//2+1)

        # Output buffer in frequency domain
        W_half = W_int // 2 + 1
        out_ft = jnp.zeros((B, C_out, H_int, W_half), dtype=jnp.complex64)

        # Low frequencies: positive H-frequencies
        x_low     = x_ft[:, :, :m1, :m2h]               # (B, C_in, m1, m2h)
        out_low   = jnp.einsum('bimn,iomn->bomn', x_low, W)
        out_ft    = out_ft.at[:, :, :m1, :m2h].add(out_low)

        # Negative H-frequencies (wrap-around modes)
        x_low_neg = x_ft[:, :, H_int-m1:, :m2h]         # (B, C_in, m1, m2h)
        out_low_neg = jnp.einsum('bimn,iomn->bomn', x_low_neg, W)
        out_ft    = out_ft.at[:, :, H_int-m1:, :m2h].add(out_low_neg)

        # irfft2 back to physical space
        out = jnp.fft.irfft2(out_ft, s=(H_int, W_int), norm='ortho')  # (B, C_out, H, W)
        out = jnp.transpose(out, (0, 2, 3, 1))           # (B, H, W, C_out)
        return out


class FNOBlock2D(nn.Module):
    """One FNO layer: spectral conv (W) + pointwise bypass (R) + activation."""
    channels: int
    modes1: int
    modes2: int
    activation: callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C)
        x_spec = SpectralConv2D(self.channels, self.modes1, self.modes2)(x)
        x_res  = nn.Dense(self.channels)(x)      # pointwise linear bypass
        return self.activation(x_spec + x_res)


class FNO2D(nn.Module):
    """
    Full FNO-2D network.

    Args:
        modes1, modes2 : Fourier truncation modes (recommend 12–20 for 64x64)
        width          : channel width in FNO layers (recommend 32–64)
        n_layers       : number of FNO blocks (recommend 4)
        lifting_dim    : intermediate lifting MLP width
        out_channels   : output channels (1 for scalar vorticity)
    """
    modes1: int = 16
    modes2: int = 16
    width: int  = 32
    n_layers: int = 4
    out_channels: int = 1

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 1)  – single vorticity channel
        # Lift to width
        x = nn.Dense(self.width)(x)
        # FNO blocks
        for _ in range(self.n_layers):
            x = FNOBlock2D(self.width, self.modes1, self.modes2)(x)
        # Project back
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_channels)(x)
        return x  # (B, H, W, 1)


# ─────────────────────────────────────────────────────────────
# 2.  LOSS
# ─────────────────────────────────────────────────────────────

def relative_l2(pred, target):
    """Relative L2 loss, averaged over batch."""
    diff  = pred - target
    denom = jnp.sqrt(jnp.mean(target ** 2, axis=(1, 2, 3)) + 1e-8)
    numer = jnp.sqrt(jnp.mean(diff  ** 2, axis=(1, 2, 3)))
    return jnp.mean(numer / denom)


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


# ─────────────────────────────────────────────────────────────
# 3.  TRAIN STATE + OPTIMIZER
# ─────────────────────────────────────────────────────────────

class TrainState(train_state.TrainState):
    pass   # can add batch_stats here if using BatchNorm


def create_train_state(key, model, input_shape, learning_rate, warmup_steps, total_steps, weight_decay=1e-4):
    """
    AdamW with linear warmup + cosine decay annealing.
    Standard schedule from the FNO paper and follow-up work.
    """
    dummy = jnp.zeros(input_shape)
    params = model.init(key, dummy)['params']

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 1e-2,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),           # gradient clipping
        optax.adamw(schedule, weight_decay=weight_decay),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), schedule


# ─────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────

def load_train_data(train_path: str, T: int, input_res: int):
    """
    Load training snapshots, compute normalisation stats, downsample, and return
    shuffleable (x, y) pairs ready for the training loop.

    train_path : .npy file (N_train, H, W) of raw physical vorticity (omega, s^-1).
                 Must be temporally separated from the validation file — no
                 automatic splitting is done here.

    Returns:
        xs, ys     : float32 (N_train-T, res, res, 1), normalised
        norm_stats : dict with 'mean' and 'std' in physical units
                     (computed from training data only)
    """
    print(f"Loading training data from {train_path} ...")
    data = np.load(train_path).astype(np.float32)   # (N, H, W) physical
    N, H, W = data.shape
    print(f"  {N} snapshots at {H}x{W}")

    # Stats computed on training data only — val must not influence normalisation.
    norm_stats = compute_norm_stats(data)
    print(f"  Physical stats: mean={norm_stats['mean']:.4f}  "
          f"std={norm_stats['std']:.4f}  (omega units)")

    # Downsample in physical space before normalising.
    if input_res < H:
        data = fourier_downsample(data, input_res)

    data_norm = normalise(data, norm_stats)
    n_pairs = N - T
    xs = data_norm[:n_pairs, :, :, None]   # (n_pairs, res, res, 1)
    ys = data_norm[T:,       :, :, None]
    print(f"  Training pairs: {n_pairs}")
    return xs, ys, norm_stats


def load_val_data(val_path: str, T: int, input_res: int, norm_stats: dict):
    """
    Load validation snapshots and return temporally-ordered (x, y) pairs.

    val_path   : .npy file (N_val, H, W) of raw physical vorticity (omega, s^-1).
                 Should cover a contiguous time window with no overlap with training.
    norm_stats : stats from load_train_data — never recomputed from val data.

    Returns:
        xs, ys : float32 (N_val-T, res, res, 1), normalised, time-ordered
    """
    print(f"Loading validation data from {val_path} ...")
    data = np.load(val_path).astype(np.float32)   # (N, H, W) physical
    N, H, W = data.shape
    print(f"  {N} snapshots at {H}x{W}")

    if input_res < H:
        data = fourier_downsample(data, input_res)

    data_norm = normalise(data, norm_stats)
    n_pairs = N - T
    xs = data_norm[:n_pairs, :, :, None]
    ys = data_norm[T:,       :, :, None]
    print(f"  Validation pairs: {n_pairs}")
    return xs, ys


def fourier_downsample(arr, target_res):
    """
    Spectrally accurate downsampling for periodic fields.
    arr: (N, H, W)  float32
    Returns: (N, target_res, target_res)
    """
    N, H, W = arr.shape
    k = target_res
    # rfft2 along spatial dims
    arr_f = np.fft.rfft2(arr, axes=(1, 2))  # (N, H, W//2+1)
    # Keep low modes
    arr_f_low = np.zeros((N, k, k // 2 + 1), dtype=np.complex64)
    m1 = min(k // 2, H // 2)
    m2 = min(k // 2 + 1, W // 2 + 1)
    arr_f_low[:, :m1,   :m2] = arr_f[:, :m1,   :m2]
    arr_f_low[:, -m1:,  :m2] = arr_f[:, H-m1:, :m2]
    # Scale to preserve amplitude
    scale = (k * k) / (H * W)
    out = np.fft.irfft2(arr_f_low, s=(k, k), axes=(1, 2)) * scale
    return out.astype(np.float32)


def make_batches(x, y, batch_size, shuffle=True, seed=0):
    """Yield (x_batch, y_batch) pairs."""
    N = x.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        yield x[b], y[b]


# ─────────────────────────────────────────────────────────────
# 5.  TRAIN / EVAL STEPS
# ─────────────────────────────────────────────────────────────

@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        pred = state.apply_fn({'params': params}, x_batch)
        return relative_l2(pred, y_batch), pred

    (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, x_batch, y_batch):
    pred = state.apply_fn({'params': state.params}, x_batch)
    return relative_l2(pred, y_batch)


# ─────────────────────────────────────────────────────────────
# 6.  MULTI-STEP ROLLOUT METRICS
# ─────────────────────────────────────────────────────────────

def rollout_error(state, x_init, targets, n_steps):
    """
    Autoregressively roll out surrogate n_steps steps from x_init.
    Returns per-step relative L2.
    """
    x = x_init
    errors = []
    for s in range(n_steps):
        x = state.apply_fn({'params': state.params}, x)
        err = float(relative_l2(x, targets[s]))
        errors.append(err)
    return errors


def energy_dissipation(vorticity_physical, nu):
    """
    Enstrophy-based energy dissipation rate in PHYSICAL units.
      ε(t) = ν · ⟨ω(t)²⟩
    vorticity_physical: (B, H, W, 1)  in physical ω units (s⁻¹)
    Returns scalar per batch element.
    """
    return nu * jnp.mean(vorticity_physical ** 2, axis=(1, 2, 3))


# ─────────────────────────────────────────────────────────────
# 7.  RESOLUTION SWEEP
# ─────────────────────────────────────────────────────────────

def evaluate_resolution(model_params, apply_fn, val_highres_path, norm_stats, T,
                        resolutions=(32, 64, 128, 256), batch_size=16):
    """
    Resolution sweep on the held-out high-res validation data.

    val_highres_path : .npy file of shape (N_val, H_orig, W_orig) containing
                       only the validation snapshots in physical units.
                       This file is separate from the low-res training data so
                       the full-res array never needs to be in memory during training.
    norm_stats       : dict with 'mean' and 'std' from training (physical units).

    For each resolution, the physical data is Fourier-downsampled, predictions
    are made in physical space via normalise/unnormalise, and relative L2
    (dimensionless) is returned.
    """
    results = {}
    data = np.load(val_highres_path).astype(np.float32)   # (N_val, H, W) physical
    N, H, W = data.shape
    print(f"  Loaded val high-res data: {N} snapshots at {H}×{W}")

    # We need at least T+1 snapshots to form one (input, target) pair
    if N <= T:
        print(f"  Not enough snapshots ({N}) for T={T} — skipping resolution sweep")
        return {}

    n_pairs = N - T
    xs_full = data[:n_pairs]    # physical inputs
    ys_full = data[T:]          # physical targets

    for res in resolutions:
        if res > H:
            print(f"  Skipping res={res} (larger than data resolution {H})")
            continue

        # Downsample in physical space
        xs = fourier_downsample(xs_full, res)[:, :, :, None]   # (n_pairs, res, res, 1)
        ys = fourier_downsample(ys_full, res)[:, :, :, None]

        errs = []
        for start in range(0, n_pairs, batch_size):
            xb = xs[start:start+batch_size]   # physical
            yb = ys[start:start+batch_size]   # physical
            # Normalise → predict → unnormalise
            xb_norm = jnp.array(normalise(xb, norm_stats))
            pred_norm = apply_fn({'params': model_params}, xb_norm)
            pred = unnormalise(np.array(pred_norm), norm_stats)   # physical
            diff  = pred - yb
            numer = np.sqrt(np.mean(diff**2,  axis=(1, 2, 3)))
            denom = np.sqrt(np.mean(yb**2,    axis=(1, 2, 3))) + 1e-8
            errs.extend((numer / denom).tolist())

        results[res] = float(np.mean(errs))
        print(f"  res={res:4d}  val rel-L2 = {results[res]:.4f}")
    return results


# ─────────────────────────────────────────────────────────────
# 8.  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    train_x, train_y, norm_stats = load_train_data(
        args.train_path, args.T, args.input_res,
    )
    val_x, val_y = load_val_data(
        args.val_path, args.T, args.input_res, norm_stats,
    )
    print(f"Train: {train_x.shape}  Val: {val_x.shape}")
    save_norm_stats(norm_stats, out_dir / 'norm_stats.json')

    # ── Model ─────────────────────────────────────────────────
    model = FNO2D(
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        n_layers=args.n_layers,
    )

    n_train = train_x.shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps     = args.max_epochs * steps_per_epoch
    warmup_steps    = int(0.05 * total_steps)   # 5% warmup

    key = jax.random.PRNGKey(args.seed)
    state, schedule = create_train_state(
        key, model,
        input_shape=(1, args.input_res, args.input_res, 1),
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=args.weight_decay,
    )
    print(f"Model params: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")

    # ── Logging ───────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'rollout_train': [], 'rollout_val': [],
        'eps_stats': [],
    }

    best_val  = np.inf
    patience  = 0
    nu        = 1.0 / args.Re   # kinematic viscosity

    # Pre-build rollout targets from val set (first batch only for speed)
    rollout_seed_x = jnp.array(val_x[:args.batch_size])   # (B, H, W, 1)

    print(f"\nStarting training — max {args.max_epochs} epochs, "
          f"early stopping patience {args.patience}\n")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()

        # ── Training ──────────────────────────────────────────
        train_losses = []
        for xb, yb in make_batches(train_x, train_y, args.batch_size,
                                   shuffle=True, seed=epoch):
            xb = jnp.array(xb); yb = jnp.array(yb)
            state, loss = train_step(state, xb, yb)
            train_losses.append(float(loss))
        train_loss = np.mean(train_losses)

        # ── Validation ────────────────────────────────────────
        val_losses = []
        for xb, yb in make_batches(val_x, val_y, args.batch_size, shuffle=False):
            xb = jnp.array(xb); yb = jnp.array(yb)
            val_losses.append(float(eval_step(state, xb, yb)))
        val_loss = np.mean(val_losses)

        # ── Rollout metrics (every `rollout_every` epochs) ────
        rollout_train_err = rollout_val_err = None
        if epoch % args.rollout_every == 0 or epoch == 1:
            # Build multi-step targets from val data
            # targets[s] = val_x shifted by (s+1) steps, wrapped
            n_rollout = min(args.rollout_steps, (val_x.shape[0] - 1))
            rollout_targets = [
                jnp.array(val_x[s+1: s+1 + args.batch_size])
                for s in range(n_rollout)
            ]
            n_available = min(len(rollout_targets),
                              len([t for t in rollout_targets if t.shape[0] == args.batch_size]))
            if n_available > 0:
                rollout_targets = rollout_targets[:n_available]
                seed_x = jnp.array(val_x[:args.batch_size])
                errs   = rollout_error(state, seed_x, rollout_targets, n_available)
                rollout_val_err = errs  # list of per-step errors

                # Energy dissipation stats along rollout — computed in PHYSICAL units.
                # Unnormalise before applying ε = ν <ω²>.
                x_curr = seed_x
                x_phys = unnormalise(np.array(x_curr), norm_stats)
                eps_pred_list = [float(np.mean(nu * x_phys**2))]
                for _ in range(n_available):
                    x_curr = np.array(state.apply_fn({'params': state.params}, jnp.array(x_curr)))
                    x_phys = unnormalise(x_curr, norm_stats)
                    eps_pred_list.append(float(np.mean(nu * x_phys**2)))

                eps_true_list = [
                    float(np.mean(nu * unnormalise(val_x[s:s+args.batch_size], norm_stats)**2))
                    for s in range(n_available + 1)
                ]
                history['eps_stats'].append({
                    'epoch': epoch,
                    'eps_pred': eps_pred_list,
                    'eps_true': eps_true_list,
                })

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        if rollout_val_err is not None:
            history['rollout_val'].append({'epoch': epoch, 'errors': rollout_val_err})

        dt = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.max_epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={float(schedule(state.step)):.2e}  "
              f"  [{dt:.1f}s]")

        # ── Checkpoint ────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            # Save checkpoint as pickle (portable, no orbax path issues)
            # Remove older checkpoints (keep 3)
            old_ckpts = sorted(ckpt_dir.glob('ckpt_epoch*.pkl'))
            for oc in old_ckpts[:-2]:
                oc.unlink()
            import pickle
            ckpt_path = ckpt_dir / f'ckpt_epoch{epoch:04d}.pkl'
            params_np = jax.tree_util.tree_map(np.array, state.params)
            with open(ckpt_path, 'wb') as _f:
                pickle.dump(params_np, _f)
            # Also save as 'best' for easy restoration
            with open(ckpt_dir / 'best_params.pkl', 'wb') as _f:
                pickle.dump(params_np, _f)
            print(f"  ✓ New best val={best_val:.4f} — checkpoint saved")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Save history every epoch
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    # ── Final: resolution sweep (optional, requires high-res val file) ─────
    if args.val_highres_path:
        print("\n=== Resolution sweep ===")
        res_errors = evaluate_resolution(
            state.params, model.apply,
            val_highres_path=args.val_highres_path,
            norm_stats=norm_stats,
            T=args.T,
            resolutions=[32, 64, 128, 256],
            batch_size=args.batch_size,
        )
        history['resolution_errors'] = res_errors
    else:
        print("\n(Skipping resolution sweep — no --val_highres_path provided)")
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete.  Best val rel-L2 = {best_val:.4f}")
    print(f"Outputs saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────
# 9.  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO surrogate for Kolmogorov flow')

    # Data
    parser.add_argument('--train_path', type=str, required=True,
                        help='Training snapshots: .npy file (N_train, H, W), physical ω. '
                             'Must be temporally earlier than (and non-overlapping with) val.')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Validation snapshots: .npy file (N_val, H, W), physical ω. '
                             'Must be a contiguous held-out time window after training data.')
    parser.add_argument('--val_highres_path', type=str, default=None,
                        help='High-res validation snapshots: .npy file (N_val, H_orig, W_orig), '
                             'physical ω.  Used only for the resolution sweep at end of training. '
                             'Optional — sweep is skipped if not provided.')
    parser.add_argument('--T',          type=int,   default=4,
                        help='Prediction horizon in simulation time units')
    parser.add_argument('--input_res',  type=int,   default=64,
                        help='Spatial resolution fed to the network (e.g. 64)')

    # Model
    parser.add_argument('--modes',    type=int, default=16,
                        help='Fourier modes to keep per dim (recommend 12-20)')
    parser.add_argument('--width',    type=int, default=32,
                        help='FNO channel width (recommend 32-64)')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of FNO blocks')

    # Training
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--max_epochs',   type=int,   default=500)
    parser.add_argument('--patience',     type=int,   default=50,
                        help='Early stopping patience in epochs')

    # Rollout evaluation
    parser.add_argument('--rollout_steps', type=int, default=20,
                        help='Number of autoregressive steps for rollout eval')
    parser.add_argument('--rollout_every', type=int, default=10,
                        help='Evaluate rollout every N epochs')

    # Physics
    parser.add_argument('--Re', type=float, default=40.0,
                        help='Reynolds number (sets nu = 1/Re)')

    # Misc
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--seed',    type=int, default=42,
                        help='RNG seed for model initialisation and batch shuffling')

    args = parser.parse_args()
    main(args)
"""
evaluate.py  –  Full post-training evaluation for the FNO Kolmogorov-flow surrogate.

Produces four metric blocks plus publication-quality figures:
  (1) One-step train/val relative L2  (with bootstrap CI)
  (2) Autoregressive rollout error    (mean ± std over many IC seeds)
  (3) Long-term energy dissipation ε  (time series + PDF + Lyapunov-time estimate)
  (4) Resolution convergence sweep    (32 / 64 / 128 / 256)

Plus bonus diagnostics:
  (5) Vorticity snapshot comparison   (true vs pred vs error at several rollout times)
  (6) 2D energy spectra               (surrogate vs DNS)
  (7) Summary metrics JSON

Usage:
    python evaluate.py \\
        --data_path /path/to/snapshots.npy \\
        --ckpt_path outputs/checkpoints/best_params.pkl \\
        --T 4  --input_res 64  --Re 40 \\
        --rollout_steps 50  --n_ic 20 \\
        --out_dir outputs/eval
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from flax import linen as nn

# ── Re-use model definition from train.py ──────────────────────────────────────
# We import the classes directly so evaluate.py stays self-contained when run
# alongside train.py in the same directory.
sys.path.insert(0, str(Path(__file__).parent))
from train import (SpectralConv2D, FNOBlock2D, FNO2D,
                   fourier_downsample, relative_l2,
                   normalise, unnormalise,
                   compute_norm_stats, load_norm_stats)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.labelsize':   12,
    'axes.titlesize':   13,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'lines.linewidth':  1.8,
})
BLUE   = '#2166AC'
RED    = '#D6604D'
GREEN  = '#4DAC26'
ORANGE = '#F4A582'
GREY   = '#888888'


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_params(ckpt_path: str):
    with open(ckpt_path, 'rb') as f:
        return pickle.load(f)


def build_apply_fn(modes, width, n_layers):
    model = FNO2D(modes1=modes, modes2=modes, width=width, n_layers=n_layers)
    return model.apply


def predict_physical(apply_fn, params, omega_physical, norm_stats):
    """
    Physical ω in → physical ω out.  Fully differentiable — no np.array()
    conversions on the hot path, so jax.grad / jax.jacobian can trace through
    the normalise → network → unnormalise chain.

    Args:
        omega_physical: (B, H, W, 1) jnp array in physical units (s⁻¹).
                        Must be a jnp array (not np) if gradients are needed.
        norm_stats:     dict with 'mean' and 'std' (Python floats from training).

    Returns:
        (B, H, W, 1) jnp array in physical units.

    For eval loops that don't need gradients, wrap the call in np.array():
        pred_np = np.array(predict_physical(apply_fn, params, x_jnp, norm_stats))
    """
    x_norm = normalise(omega_physical, norm_stats)          # jnp → jnp
    y_norm = apply_fn({'params': params}, x_norm)           # jnp → jnp
    return unnormalise(y_norm, norm_stats)                  # jnp → jnp


def sensitivity_wrt_input(apply_fn, params, omega_physical, norm_stats,
                           output_fn=None):
    """
    Compute d(output) / d(omega_physical) via jax.jacobian, tracing through
    the full normalise → FNO → unnormalise chain.

    Args:
        omega_physical : (1, H, W, 1) jnp array, single input field (no batch
                         dim > 1 — Jacobian over a batch is expensive).
        norm_stats     : training norm stats dict.
        output_fn      : optional callable (jnp array → scalar) applied to the
                         output field before differentiating.  If None, returns
                         the full Jacobian d(omega_out[i,j]) / d(omega_in[k,l])
                         of shape (H, W, H, W) after squeezing batch/channel dims.
                         Pass e.g. `lambda y: jnp.mean(y)` for a scalar sensitivity
                         map of shape (1, H, W, 1).

    Returns:
        If output_fn is None : jnp array (H, W, H, W) — full spatial Jacobian.
        If output_fn given   : jnp array (1, H, W, 1) — gradient of scalar wrt input.

    Examples:
        # Full Jacobian (expensive — O(H*W) network passes via forward-mode):
        J = sensitivity_wrt_input(apply_fn, params, x, norm_stats)

        # Gradient of mean output wrt input (cheap — single reverse-mode pass):
        grad = sensitivity_wrt_input(apply_fn, params, x, norm_stats,
                                     output_fn=lambda y: jnp.mean(y))

        # Gradient of output at a specific point (i, j) wrt full input:
        grad_ij = sensitivity_wrt_input(apply_fn, params, x, norm_stats,
                                        output_fn=lambda y: y[0, i, j, 0])
    """
    def _forward(x):
        return predict_physical(apply_fn, params, x, norm_stats)

    if output_fn is not None:
        # Scalar output → use grad (single reverse-mode pass, cheap)
        grad_fn = jax.grad(lambda x: output_fn(_forward(x)))
        return grad_fn(omega_physical)
    else:
        # Field output → full Jacobian via jacfwd (forward-mode, O(H*W) passes)
        # squeeze/unsqueeze to get a clean (H, W, H, W) result
        H, W = omega_physical.shape[1], omega_physical.shape[2]
        jac = jax.jacobian(lambda x: _forward(x)[0, :, :, 0])(omega_physical)
        return jac[:, :, 0, :, :, 0]   # (H, W, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_val_data(val_path: str, T: int, input_res: int, norm_stats: dict):
    """
    Load the validation snapshots (a separate, temporally-held-out file) and
    return them in physical units at the requested resolution.

    val_path   : .npy file (N_val, H, W) of raw physical vorticity (omega, s^-1).
    norm_stats : training norm stats — normalisation is applied only at network
                 boundaries; physical data is returned here.

    Returns:
        data    : (N_val, res, res) float32 array in PHYSICAL units
        n_pairs : number of (input, target) pairs = N_val - T
    """
    data = np.load(val_path).astype(np.float32)   # (N_val, H, W) physical
    N, H, W = data.shape
    print(f"  {N} validation snapshots at {H}x{W}")

    if input_res < H:
        data = fourier_downsample(data, input_res)   # still physical

    return data, N - T


def get_consecutive_segment(data, start, length):
    """Return data[start : start+length, :, :, None]  (length, H, W, 1)."""
    seg = data[start : start + length]
    return seg[:, :, :, None].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Core inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_one_step(apply_fn, params, x):
    """x: (B, H, W, 1) → pred (B, H, W, 1)."""
    return apply_fn({'params': params}, x)


@jax.jit
def _jit_predict(apply_fn, params, x):
    return apply_fn({'params': params}, x)


def rollout(apply_fn, params, x0, n_steps, batch_size=32):
    """
    Autoregressive rollout from initial condition x0 (B, H, W, 1).
    Returns array of shape (n_steps+1, B, H, W, 1), step 0 = x0.
    """
    trajectory = [x0]
    x = x0
    for _ in range(n_steps):
        # Process in batches to manage memory
        chunks = []
        for b in range(0, x.shape[0], batch_size):
            xb = jnp.array(x[b:b+batch_size])
            chunks.append(np.array(apply_fn({'params': params}, xb)))
        x = np.concatenate(chunks, axis=0)
        trajectory.append(x)
    return np.stack(trajectory, axis=0)   # (n_steps+1, B, H, W, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metric 1 — One-step L2 with bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def metric1_onestep(apply_fn, params, data, T, norm_stats,
                    batch_size=32, n_bootstrap=500):
    """
    One-step relative L2 across all val pairs.  data is the full (N_val, H, W)
    physical array; pairs are (data[i], data[i+T]) for i in 0..N_val-T-1.
    """
    n_pairs = data.shape[0] - T
    xs = data[:n_pairs, :, :, None]   # physical
    ys = data[T:,       :, :, None]   # physical
    N = n_pairs

    per_sample_err = []
    for start in range(0, N, batch_size):
        xb_phys = jnp.array(xs[start:start+batch_size])
        yb_phys = xs[start:start+batch_size]           # keep np for metric math
        pred_phys = np.array(predict_physical(apply_fn, params, xb_phys, norm_stats))
        diff  = pred_phys - yb_phys
        numer = np.sqrt(np.mean(diff**2, axis=(1,2,3)))
        denom = np.sqrt(np.mean(yb_phys**2, axis=(1,2,3))) + 1e-8
        per_sample_err.extend((numer / denom).tolist())

    err = np.array(per_sample_err)
    rng = np.random.default_rng(0)
    boot_means = [rng.choice(err, size=len(err), replace=True).mean()
                  for _ in range(n_bootstrap)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        'mean':  float(err.mean()),
        'std':   float(err.std()),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
        'per_sample': err.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metric 2 — Rollout error over many ICs
# ─────────────────────────────────────────────────────────────────────────────

def metric2_rollout(apply_fn, params, data, T, n_steps, n_ic,
                    norm_stats, batch_size=32):
    """
    Autoregressive rollout over n_ic ICs sampled from the val data.
    data is the full (N_val, H, W) physical array.
    Operates entirely in physical units via predict_physical().
    Returns per-step mean and std of relative L2 (dimensionless).
    """
    # Any index that leaves room for n_steps*T more steps is a valid IC.
    max_start = data.shape[0] - n_steps * T - 1
    valid_starts = np.arange(max(0, max_start))
    rng = np.random.default_rng(1)
    chosen = rng.choice(valid_starts, size=min(n_ic, len(valid_starts)), replace=False)

    per_ic_errors = []   # list of (n_steps,) arrays
    for ic_start in chosen:
        x = jnp.array(data[ic_start][None, :, :, None])   # (1, H, W, 1) jnp
        errors = []
        for s in range(1, n_steps + 1):
            x = predict_physical(apply_fn, params, x, norm_stats)    # jnp out
            true_s = data[ic_start + s * T][None, :, :, None]        # np, physical
            diff  = np.array(x) - true_s
            numer = np.sqrt(np.mean(diff**2))
            denom = np.sqrt(np.mean(true_s**2)) + 1e-8
            errors.append(float(numer / denom))
        per_ic_errors.append(np.array(errors))

    per_ic_errors = np.stack(per_ic_errors, axis=0)   # (n_ic, n_steps)
    return {
        'steps': list(range(1, n_steps + 1)),
        'T_per_step': T,
        'mean': per_ic_errors.mean(axis=0).tolist(),
        'std':  per_ic_errors.std(axis=0).tolist(),
        'per_ic': per_ic_errors.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Metric 3 — Long-term energy dissipation
# ─────────────────────────────────────────────────────────────────────────────

def metric3_dissipation(apply_fn, params, data, ic_start, n_steps, T, nu,
                        norm_stats, batch_size=1):
    """
    Roll out n_steps steps from ic_start and track ε(t) = ν <ω²> in PHYSICAL units.
    data is in physical units (ω, s⁻¹).  ε has units of s⁻².
    """
    # Ground truth: consecutive snapshots every T steps — already physical
    true_snaps  = [data[ic_start + s * T] for s in range(n_steps + 1)
                   if ic_start + s * T < data.shape[0]]
    n_avail = len(true_snaps)

    eps_true = [float(nu * np.mean(s**2)) for s in true_snaps]   # physical ε

    # Surrogate rollout via predict_physical — in, out both physical
    x = jnp.array(data[ic_start][None, :, :, None])               # physical IC, jnp
    eps_pred = [float(nu * np.mean(np.array(x)**2))]
    for _ in range(n_avail - 1):
        x = predict_physical(apply_fn, params, x, norm_stats)      # jnp out
        eps_pred.append(float(nu * np.mean(np.array(x)**2)))

    times = [s * T for s in range(n_avail)]

    # Time-mean and RMS
    return {
        'times':          times,
        'eps_true':       eps_true,
        'eps_pred':       eps_pred,
        'eps_true_mean':  float(np.mean(eps_true)),
        'eps_pred_mean':  float(np.mean(eps_pred)),
        'eps_true_rms':   float(np.std(eps_true)),
        'eps_pred_rms':   float(np.std(eps_pred)),
        'rel_error_mean': float(abs(np.mean(eps_pred) - np.mean(eps_true))
                               / (abs(np.mean(eps_true)) + 1e-12)),
    }


def dissipation_pdf(apply_fn, params, data, T, nu, norm_stats,
                    n_ic=10, n_steps=40, n_bins=40):
    """
    Collect ε values from many rollout trajectories to compare PDFs.
    data is the full (N_val, H, W) physical array.
    """
    rng = np.random.default_rng(2)
    max_start = data.shape[0] - n_steps * T - 1
    valid = np.arange(max(0, max_start))
    chosen = rng.choice(valid, size=min(n_ic, len(valid)), replace=False)

    eps_true_all = []
    eps_pred_all = []

    for ic in chosen:
        res = metric3_dissipation(apply_fn, params, data, ic, n_steps, T, nu, norm_stats)
        eps_true_all.extend(res['eps_true'])
        eps_pred_all.extend(res['eps_pred'])

    return np.array(eps_true_all), np.array(eps_pred_all)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Metric 4 — Resolution sweep
# ─────────────────────────────────────────────────────────────────────────────

def metric4_resolution(apply_fn, params, val_highres_path, T, norm_stats,
                       resolutions=(32, 64, 128, 256), batch_size=32):
    """
    Resolution sweep on the held-out high-res validation data.

    val_highres_path : .npy file of shape (N_val, H_orig, W_orig) containing
                       only the validation snapshots, in physical units.
                       This is a separate file from the low-res training data —
                       no split logic is needed here since the file is already
                       val-only.
    norm_stats : dict with 'mean' and 'std' from training (physical units).
    """
    raw = np.load(val_highres_path).astype(np.float32)   # (N_val, H, W) physical
    N, H_orig, W_orig = raw.shape
    print(f"  Loaded val high-res data: {N} snapshots at {H_orig}×{W_orig}")

    if N <= T:
        print(f"  Not enough snapshots ({N}) for T={T} — skipping resolution sweep")
        return {}

    n_pairs = N - T
    xs_full = raw[:n_pairs]    # physical inputs
    ys_full = raw[T:]          # physical targets

    results = {}
    for res in resolutions:
        if res > H_orig:
            print(f"  res={res}: skipped (> data resolution {H_orig})")
            results[res] = None
            continue

        # Downsample in physical space
        xs = fourier_downsample(xs_full, res)[:, :, :, None]   # physical
        ys = fourier_downsample(ys_full, res)[:, :, :, None]   # physical

        errs = []
        for start in range(0, n_pairs, batch_size):
            xb = jnp.array(xs[start:start+batch_size])
            yb = ys[start:start+batch_size]                           # np for metric math
            pred = np.array(predict_physical(apply_fn, params, xb, norm_stats))
            diff  = pred - yb
            numer = np.sqrt(np.mean(diff**2, axis=(1, 2, 3)))
            denom = np.sqrt(np.mean(yb**2,   axis=(1, 2, 3))) + 1e-8
            errs.extend((numer / denom).tolist())

        results[res] = float(np.mean(errs))
        print(f"  res={res:4d}  val rel-L2 = {results[res]:.4f}")
    return results


def energy_spectrum_2d(field):
    """
    Isotropically binned 2D kinetic energy spectrum from a vorticity field.
    field: (H, W)  in spectral space ω̂  →  E(k) ∝ Σ_{|k|≈k} |ω̂_k|² / k²
    Returns (k_bins, E_k).
    """
    H, W = field.shape
    omega_hat = np.fft.rfft2(field, norm='ortho')
    kx = np.fft.fftfreq(H, d=1.0/H).astype(int)
    ky = np.arange(W // 2 + 1)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # |ω̂|²
    power = np.abs(omega_hat)**2
    # E(k) = |ω̂|² / k² (relate vorticity to velocity energy)
    with np.errstate(divide='ignore', invalid='ignore'):
        E = np.where(K > 0, power / K**2, 0.0)

    k_max = int(min(H, W) // 2)
    k_bins = np.arange(1, k_max + 1)
    E_k = np.array([E[np.round(K).astype(int) == k].sum() for k in k_bins])
    return k_bins, E_k


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_loss_curves(history_path, out_dir):
    """Loss curves from training history."""
    if not Path(history_path).exists():
        return
    with open(history_path) as f:
        hist = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Train / val loss
    ax = axes[0]
    ep = np.arange(1, len(hist['train_loss']) + 1)
    ax.semilogy(ep, hist['train_loss'], color=BLUE,  label='Train', lw=1.6)
    ax.semilogy(ep, hist['val_loss'],   color=RED,   label='Val',   lw=1.6)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L₂')
    ax.set_title('(a) Training / Validation Loss')
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.35)

    # (b) Rollout at different training epochs
    ax = axes[1]
    rollout_data = hist.get('rollout_val', [])
    if rollout_data:
        cmap = plt.cm.plasma
        n = len(rollout_data)
        for i, rec in enumerate(rollout_data):
            steps  = np.array(rec['errors'])
            t_vals = np.arange(1, len(steps) + 1) * hist.get('T', 4)
            c = cmap(i / max(n - 1, 1))
            ax.semilogy(t_vals, steps, color=c, lw=1.2, alpha=0.7,
                        label=f"ep {rec['epoch']}" if i in (0, n//2, n-1) else '_')
        ax.set_xlabel(r'Simulation time $t/T_0$')
        ax.set_ylabel('Rollout Relative L₂')
        ax.set_title('(b) Rollout Error During Training')
        ax.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(cmap=cmap,
             norm=plt.Normalize(rollout_data[0]['epoch'], rollout_data[-1]['epoch']))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Epoch')
    ax.grid(True, which='both', ls='--', alpha=0.35)

    fig.tight_layout()
    path = out_dir / 'fig1_loss_curves.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_onestep_distribution(m1_results, out_dir):
    """Histogram + violin of per-sample one-step errors."""
    errs = np.array(m1_results['per_sample'])
    mean = m1_results['mean']
    ci_lo, ci_hi = m1_results['ci_lo'], m1_results['ci_hi']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(errs, bins=40, color=BLUE, alpha=0.75, edgecolor='white', lw=0.4)
    ax.axvline(mean,  color=RED,    ls='-',  lw=2,   label=f'Mean = {mean:.3f}')
    ax.axvline(ci_lo, color=ORANGE, ls='--', lw=1.5, label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.axvline(ci_hi, color=ORANGE, ls='--', lw=1.5)
    ax.set_xlabel('Per-sample Relative L₂'); ax.set_ylabel('Count')
    ax.set_title('(a) One-step Error Distribution (Val)')
    ax.legend(); ax.grid(ls='--', alpha=0.35)

    ax = axes[1]
    vp = ax.violinplot([errs], positions=[1], showmedians=True,
                       showextrema=True, widths=0.6)
    vp['bodies'][0].set_facecolor(BLUE); vp['bodies'][0].set_alpha(0.6)
    vp['cmedians'].set_color(RED); vp['cmedians'].set_linewidth(2)
    ax.set_xticks([1]); ax.set_xticklabels(['Val set'])
    ax.set_ylabel('Relative L₂')
    ax.set_title('(b) Violin: One-step Rel L₂')
    ax.grid(ls='--', alpha=0.35, axis='y')

    fig.tight_layout()
    path = out_dir / 'fig2_onestep_errors.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_rollout_error(m2_results, out_dir):
    """Mean ± std rollout error with shaded band + saturation line."""
    steps = np.array(m2_results['steps'])
    t     = steps * m2_results['T_per_step']
    mean  = np.array(m2_results['mean'])
    std   = np.array(m2_results['std'])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(t, mean - std, mean + std,
                    color=BLUE, alpha=0.20, label='± 1 std')
    ax.semilogy(t, mean, color=BLUE, lw=2, label='Mean rel-L₂')

    # Plot individual IC trajectories in light grey
    per_ic = np.array(m2_results['per_ic'])
    for traj in per_ic:
        ax.semilogy(t, traj, color=GREY, lw=0.5, alpha=0.4)

    # Saturation reference (climatological error ≈ 1 for normalised vorticity)
    ax.axhline(1.0, color=RED, ls='--', lw=1.5, label='Saturation (rel-L₂=1)')

    # Mark Lyapunov-time proxy: first time mean error > 0.5
    lyap_mask = mean > 0.5
    if lyap_mask.any():
        t_lyap = t[lyap_mask][0]
        ax.axvline(t_lyap, color=GREEN, ls=':', lw=1.8,
                   label=f'Error>0.5 at t={t_lyap:.0f}')

    ax.set_xlabel(r'Simulation time  $t$  (T$_0$ units)')
    ax.set_ylabel('Relative L₂')
    ax.set_title('Autoregressive Rollout Error (Val, many ICs)')
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.35)
    fig.tight_layout()
    path = out_dir / 'fig3_rollout_error.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_dissipation(m3_results, eps_true_all, eps_pred_all, out_dir):
    """Three-panel: time-series, PDF comparison, long-term mean convergence."""
    fig = plt.figure(figsize=(15, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # (a) Time series
    ax = fig.add_subplot(gs[0])
    t  = np.array(m3_results['times'])
    ax.plot(t, m3_results['eps_true'], color='k',  lw=2,   label='DNS')
    ax.plot(t, m3_results['eps_pred'], color=RED,  lw=1.8, ls='--', label='Surrogate')
    ax.set_xlabel('Simulation time  $t$')
    ax.set_ylabel(r'$\varepsilon(t) = \nu \langle \omega^2 \rangle$  (s$^{-2}$)')
    ax.set_title(r'(a) $\varepsilon(t)$ — one rollout')
    ax.legend(); ax.grid(ls='--', alpha=0.35)

    # (b) PDF
    ax = fig.add_subplot(gs[1])
    bins = np.linspace(min(eps_true_all.min(), eps_pred_all.min()),
                       max(eps_true_all.max(), eps_pred_all.max()), 45)
    ax.hist(eps_true_all, bins=bins, color='k',  alpha=0.55,
            density=True, label='DNS',       edgecolor='white', lw=0.3)
    ax.hist(eps_pred_all, bins=bins, color=RED, alpha=0.50,
            density=True, label='Surrogate', edgecolor='white', lw=0.3)
    ax.set_xlabel(r'$\varepsilon$'); ax.set_ylabel('Density')
    ax.set_title(r'(b) PDF of $\varepsilon$')
    ax.legend(); ax.grid(ls='--', alpha=0.35)

    # (c) Relative error in time-mean eps
    ax = fig.add_subplot(gs[2])
    vals = [m3_results['eps_true_mean'], m3_results['eps_pred_mean']]
    errs = [m3_results['eps_true_rms'],  m3_results['eps_pred_rms']]
    labels = ['DNS', 'Surrogate']
    colors = ['k', RED]
    x = np.array([0, 1])
    bars = ax.bar(x, vals, yerr=errs, color=colors, alpha=0.75,
                  capsize=6, width=0.5, error_kw={'lw': 2})
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r'$\langle \varepsilon \rangle$')
    rel = m3_results['rel_error_mean'] * 100
    ax.set_title(f'(c) Time-mean ε  (rel err = {rel:.1f}%)')
    ax.grid(ls='--', alpha=0.35, axis='y')

    fig.suptitle('Energy Dissipation Rate — DNS vs Surrogate', fontsize=14, y=1.02)
    path = out_dir / 'fig4_dissipation.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_resolution_sweep(m4_results, out_dir):
    """Log-log plot of val error vs. resolution."""
    resolutions = sorted(r for r, v in m4_results.items() if v is not None)
    errors      = [m4_results[r] for r in resolutions]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(resolutions, errors, 'o-', color=BLUE, ms=8, lw=2, zorder=3)
    for r, e in zip(resolutions, errors):
        ax.annotate(f'{e:.3f}', xy=(r, e), xytext=(5, 5),
                    textcoords='offset points', fontsize=10, color=BLUE)

    # Fit power-law slope (log-log regression)
    if len(resolutions) >= 3:
        log_r = np.log(resolutions); log_e = np.log(errors)
        slope, intercept = np.polyfit(log_r, log_e, 1)
        r_fit = np.array(resolutions, dtype=float)
        ax.loglog(r_fit, np.exp(intercept) * r_fit**slope,
                  ls='--', color=GREY, lw=1.4,
                  label=f'Slope = {slope:.2f}  (N⁻ⁿ scaling)')
        ax.legend()

    ax.set_xlabel('Input resolution (pixels per side)')
    ax.set_ylabel('Val Relative L₂')
    ax.set_title('FNO Error vs. Spatial Resolution')
    ax.set_xticks(resolutions)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which='both', ls='--', alpha=0.35)

    fig.tight_layout()
    path = out_dir / 'fig5_resolution_sweep.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_snapshot_comparison(apply_fn, params, data, ic_start, T,
                             rollout_times, out_dir, norm_stats=None):
    """
    Side-by-side: true | surrogate | pointwise error at multiple rollout times.
    data is in physical units.  Colourbars show physical ω (s⁻¹).
    rollout_times: list of integers (number of T-steps).
    """
    n_t = len(rollout_times)
    fig, axes = plt.subplots(3, n_t, figsize=(3.5 * n_t, 9))

    # Roll out in physical space
    x = jnp.array(data[ic_start][None, :, :, None])           # physical IC, jnp
    preds = {0: np.array(x)}
    for step in range(1, max(rollout_times) + 1):
        x = predict_physical(apply_fn, params, x, norm_stats)  # jnp out
        preds[step] = np.array(x)                              # pull back for plotting

    cmap_vort  = 'RdBu_r'
    cmap_err   = 'hot_r'

    for col, step in enumerate(rollout_times):
        true_f = data[ic_start + step * T]              # (H, W)
        pred_f = preds[step][0, :, :, 0]                # (H, W)
        err_f  = np.abs(pred_f - true_f)

        vmax = max(np.abs(true_f).max(), np.abs(pred_f).max())
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        for row, (field, title) in enumerate([
            (true_f, f'DNS  t={step*T}'),
            (pred_f, f'FNO  t={step*T}'),
            (err_f,  f'|Error|  t={step*T}'),
        ]):
            ax = axes[row, col]
            if row < 2:
                im = ax.imshow(field, cmap=cmap_vort, norm=norm, origin='lower')
            else:
                im = ax.imshow(field, cmap=cmap_err, vmin=0, vmax=vmax*0.5, origin='lower')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel('DNS',       fontsize=12)
    axes[1, 0].set_ylabel('Surrogate', fontsize=12)
    axes[2, 0].set_ylabel('|Error|',   fontsize=12)

    fig.suptitle(r'Vorticity $\omega$ (s$^{-1}$): DNS vs FNO Surrogate', fontsize=14)
    fig.tight_layout()
    path = out_dir / 'fig6_snapshots.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


def fig_energy_spectra(apply_fn, params, data, T, norm_stats, n_fields=30, out_dir=None):
    """Compare isotropic 2D energy spectra of DNS vs surrogate fields."""
    rng = np.random.default_rng(3)
    valid = np.arange(data.shape[0] - T)
    chosen = rng.choice(valid, size=min(n_fields, len(valid)), replace=False)

    E_true_list = []
    E_pred_list = []
    for ic in chosen:
        x0 = jnp.array(data[ic][None, :, :, None])
        pred = np.array(predict_physical(apply_fn, params, x0, norm_stats))[0, :, :, 0]
        true = data[ic + T]

        k, Et = energy_spectrum_2d(true)
        _, Ep = energy_spectrum_2d(pred)
        E_true_list.append(Et)
        E_pred_list.append(Ep)

    E_true = np.stack(E_true_list).mean(axis=0)
    E_pred = np.stack(E_pred_list).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(k, E_true, color='k',  lw=2,   label='DNS')
    ax.loglog(k, E_pred, color=RED,  lw=1.8, ls='--', label='Surrogate')

    # Reference k^{-5/3} Kolmogorov scaling
    k_ref = k[k >= 3]
    C = E_true[k >= 3][0] * k_ref[0]**(5/3)
    ax.loglog(k_ref, C * k_ref**(-5/3), color=GREY, ls=':', lw=1.4,
              label=r'$k^{-5/3}$')

    ax.set_xlabel('Wavenumber $k$'); ax.set_ylabel(r'$E(k)$')
    ax.set_title('Isotropic Energy Spectrum — DNS vs FNO')
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.35)
    fig.tight_layout()
    path = out_dir / 'fig7_energy_spectra.pdf'
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nu = 1.0 / args.Re
    print(f"\nRe={args.Re}  ν={nu:.5f}  T={args.T}  res={args.input_res}")

    # ── Load norm stats (saved by train.py alongside checkpoint) ──
    norm_stats_path = Path(args.ckpt_path).parent.parent / 'norm_stats.json'
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"norm_stats.json not found at {norm_stats_path}.\n"
            "This file is written by train.py alongside the checkpoint. "
            "Make sure you are pointing --ckpt_path at a checkpoint produced "
            "by the current version of train.py."
        )
    norm_stats = load_norm_stats(norm_stats_path)
    print(f"  Norm stats: mean={norm_stats['mean']:.4f}  std={norm_stats['std']:.4f}  (physical ω units)")

    # ── Load model ────────────────────────────────────────────
    print("\nLoading checkpoint ...")
    params   = load_params(args.ckpt_path)
    apply_fn = build_apply_fn(args.modes, args.width, args.n_layers)

    # Warm-up JIT: normalise a dummy physical input
    dummy_phys = jnp.zeros((1, args.input_res, args.input_res, 1))
    _ = predict_physical(apply_fn, params, dummy_phys, norm_stats)
    print("  Model loaded and JIT-compiled.")

    # ── Load data (physical units) ────────────────────────────
    print("\nLoading validation data ...")
    data, n_pairs = load_val_data(
        args.val_path, args.T, args.input_res, norm_stats,
    )
    print(f"  Val array: {data.shape}  ({n_pairs} pairs, physical ω, s⁻¹)")

    # Pick a stable IC for long-rollout plots — any index with enough room ahead.
    rng = np.random.default_rng(7)
    max_start = data.shape[0] - args.rollout_steps * args.T - 2
    if max_start > 0:
        ic_long = int(rng.integers(0, max_start))
    else:
        ic_long = 0

    # ─────────────────────────────────────────────────────────
    print("\n=== Metric 1: One-step L2 (with bootstrap CI) ===")
    m1 = metric1_onestep(apply_fn, params, data, args.T, norm_stats,
                         batch_size=args.batch_size)
    print(f"  mean={m1['mean']:.4f}  std={m1['std']:.4f}  "
          f"95% CI=[{m1['ci_lo']:.4f}, {m1['ci_hi']:.4f}]")

    # ─────────────────────────────────────────────────────────
    print("\n=== Metric 2: Rollout error over many ICs ===")
    m2 = metric2_rollout(apply_fn, params, data, args.T,
                         n_steps=args.rollout_steps, n_ic=args.n_ic,
                         norm_stats=norm_stats, batch_size=args.batch_size)
    # Print horizon where error first exceeds 0.5
    mean_arr = np.array(m2['mean'])
    if (mean_arr > 0.5).any():
        t_half = int(np.argmax(mean_arr > 0.5) + 1) * args.T
        print(f"  Error > 0.5 at t ≈ {t_half} sim-time units")
    else:
        print(f"  Error stays < 0.5 for all {args.rollout_steps} steps")

    # ─────────────────────────────────────────────────────────
    print("\n=== Metric 3: Energy dissipation ===")
    m3 = metric3_dissipation(apply_fn, params, data, ic_long,
                             n_steps=args.rollout_steps, T=args.T, nu=nu,
                             norm_stats=norm_stats)
    eps_true_all, eps_pred_all = dissipation_pdf(
        apply_fn, params, data, args.T, nu, norm_stats,
        n_ic=args.n_ic, n_steps=args.rollout_steps,
    )
    print(f"  DNS  <ε> = {m3['eps_true_mean']:.5f} ± {m3['eps_true_rms']:.5f}")
    print(f"  FNO  <ε> = {m3['eps_pred_mean']:.5f} ± {m3['eps_pred_rms']:.5f}")
    print(f"  Relative error in mean ε: {m3['rel_error_mean']*100:.2f}%")

    # ─────────────────────────────────────────────────────────
    print("\n=== Metric 4: Resolution sweep ===")
    if args.val_highres_path:
        m4 = metric4_resolution(apply_fn, params, args.val_highres_path, args.T,
                                norm_stats=norm_stats,
                                resolutions=[32, 64, 128, 256],
                                batch_size=args.batch_size)
    else:
        print("  (Skipping — no --val_highres_path provided)")
        m4 = {}

    # ─────────────────────────────────────────────────────────
    # Save summary JSON
    summary = {
        'metric1_onestep': {k: v for k, v in m1.items() if k != 'per_sample'},
        'metric2_rollout': {k: v for k, v in m2.items() if k != 'per_ic'},
        'metric3_dissipation': {k: v for k, v in m3.items()
                                if k not in ('times', 'eps_true', 'eps_pred')},
        'metric4_resolution': {str(k): v for k, v in m4.items()},
        'config': vars(args),
    }
    with open(out_dir / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {out_dir}/metrics_summary.json")

    # ─────────────────────────────────────────────────────────
    print("\n=== Generating figures ===")

    # Training curves (if history exists)
    history_path = Path(args.out_dir).parent / 'history.json'
    fig_loss_curves(str(history_path), out_dir)

    fig_onestep_distribution(m1, out_dir)
    fig_rollout_error(m2, out_dir)
    fig_dissipation(m3, eps_true_all, eps_pred_all, out_dir)
    fig_resolution_sweep(m4, out_dir)

    # Snapshot comparison at 1, 5, 10 rollout steps (if enough data)
    snap_steps = [s for s in [1, 5, 10, 20] if s <= args.rollout_steps
                  and ic_long + s * args.T < data.shape[0]]
    if snap_steps:
        fig_snapshot_comparison(apply_fn, params, data, ic_long,
                                args.T, snap_steps[:4], out_dir,
                                norm_stats=norm_stats)

    fig_energy_spectra(apply_fn, params, data, args.T,
                       norm_stats=norm_stats, n_fields=30, out_dir=out_dir)

    print(f"\nAll outputs written to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FNO Kolmogorov-flow evaluation')

    parser.add_argument('--val_path', type=str, required=True,
                        help='Validation snapshots: low-res .npy file (N_val, H_lo, W_lo), '
                             'physical ω.  Same resolution as --val_path in train.py. '
                             'Used for metrics 1–3, snapshots, and energy spectra.')
    parser.add_argument('--val_highres_path', type=str, default=None,
                        help='High-res validation snapshots: .npy file (N_val, H_orig, W_orig), '
                             'physical ω.  Used only for metric 4 (resolution sweep). '
                             'Optional — metric 4 is skipped if not provided.')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to best_params.pkl saved by train.py')
    parser.add_argument('--T',         type=int,   default=4)
    parser.add_argument('--input_res', type=int,   default=64)
    parser.add_argument('--Re',        type=float, default=40.0)

    # Model (must match training config)
    parser.add_argument('--modes',    type=int, default=16)
    parser.add_argument('--width',    type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)

    parser.add_argument('--rollout_steps', type=int, default=50,
                        help='Autoregressive steps for rollout error + dissipation metrics')
    parser.add_argument('--n_ic',          type=int, default=20,
                        help='Number of ICs for rollout / PDF averaging')
    parser.add_argument('--batch_size',    type=int, default=32)
    parser.add_argument('--out_dir',       type=str, default='outputs/eval')

    main(parser.parse_args())
"""
plot_metrics.py  –  Visualise training history produced by train.py

Usage:
    python plot_metrics.py --history outputs/history.json --out_dir outputs/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_loss_curves(history, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.semilogy(epochs, history['train_loss'], label='Train rel-L2', lw=1.5)
    ax.semilogy(epochs, history['val_loss'],   label='Val rel-L2',   lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Relative L2')
    ax.set_title('Training / Validation Loss')
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'loss_curves.png', dpi=150)
    plt.close(fig)
    print("  loss_curves.png")


def plot_rollout_errors(history, out_dir):
    rollout = history.get('rollout_val', [])
    if not rollout:
        print("  No rollout data found.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    # Plot last few rollout evaluations with colour gradient
    cmap = plt.cm.viridis
    n = len(rollout)
    for i, rec in enumerate(rollout):
        steps  = np.arange(1, len(rec['errors']) + 1)
        colour = cmap(i / max(n - 1, 1))
        ax.semilogy(steps, rec['errors'],
                    color=colour, lw=1.2, alpha=0.8,
                    label=f"epoch {rec['epoch']}" if i in (0, n//2, n-1) else None)
    ax.set_xlabel('Rollout step  (×T simulation units)')
    ax.set_ylabel('Relative L2')
    ax.set_title('Autoregressive Rollout Error (validation)')
    ax.legend(loc='upper left'); ax.grid(True, which='both', ls='--', alpha=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(rollout[0]['epoch'], rollout[-1]['epoch']))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Epoch')
    fig.tight_layout()
    fig.savefig(out_dir / 'rollout_errors.png', dpi=150)
    plt.close(fig)
    print("  rollout_errors.png")


def plot_eps_stats(history, out_dir):
    eps = history.get('eps_stats', [])
    if not eps:
        print("  No eps_stats data found.")
        return
    # Use the final recorded epoch
    last = eps[-1]
    steps_pred = np.arange(len(last['eps_pred']))
    steps_true = np.arange(len(last['eps_true']))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_true, last['eps_true'], 'k-',  lw=1.8, label='True ε')
    ax.plot(steps_pred, last['eps_pred'], 'r--', lw=1.8, label='Surrogate ε')
    ax.set_xlabel('Rollout step'); ax.set_ylabel('ε = ν ⟨ω²⟩')
    ax.set_title(f"Energy Dissipation Rate (epoch {last['epoch']})")
    ax.legend(); ax.grid(ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'energy_dissipation.png', dpi=150)
    plt.close(fig)
    print("  energy_dissipation.png")

    # Also time-mean eps over all epochs
    pred_means = [np.mean(e['eps_pred']) for e in eps]
    true_mean  = np.mean(eps[-1]['eps_true'])
    epochs     = [e['epoch'] for e in eps]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(epochs, pred_means, 'r-o', ms=4, label='Surrogate ⟨ε⟩')
    ax.axhline(true_mean, color='k', ls='--', label='True ⟨ε⟩')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Time-mean ε')
    ax.set_title('Long-term Energy Dissipation (training progress)')
    ax.legend(); ax.grid(ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'eps_convergence.png', dpi=150)
    plt.close(fig)
    print("  eps_convergence.png")


def plot_resolution_errors(history, out_dir):
    res_err = history.get('resolution_errors', {})
    if not res_err:
        print("  No resolution sweep data found.")
        return
    resolutions = sorted(int(k) for k in res_err.keys())
    errors      = [res_err[str(r)] for r in resolutions]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(resolutions, errors, 'b-o', ms=7, lw=2)
    for r, e in zip(resolutions, errors):
        ax.annotate(f'{e:.3f}', (r, e), textcoords='offset points',
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel('Input resolution (pixels per side)')
    ax.set_ylabel('Validation Relative L2')
    ax.set_title('FNO Error vs. Spatial Resolution')
    ax.set_xticks(resolutions)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which='both', ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / 'resolution_sweep.png', dpi=150)
    plt.close(fig)
    print("  resolution_sweep.png")


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load(args.history)
    print("Generating plots:")

    plot_loss_curves(history, out_dir)
    plot_rollout_errors(history, out_dir)
    plot_eps_stats(history, out_dir)
    plot_resolution_errors(history, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, required=True,
                        help='Path to history.json produced by train.py')
    parser.add_argument('--out_dir', type=str, default='outputs/plots')
    main(parser.parse_args())

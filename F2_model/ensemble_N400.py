#!/usr/bin/env python3
"""
Ensemble of 10 particle simulations at N=400, T=0.4
====================================================

Measures D_rot from each realization to characterize
the quenched disorder fluctuations at the same N
used in the first paper.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brownian_s2_simulation import BrownianParticlesS2, SimulationParams
import matplotlib.pyplot as plt
import time as clock

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_ring_normal(positions):
    cov = positions.T @ positions / len(positions)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_hat = eigvecs[:, np.argmin(eigvals)]
    return n_hat, eigvals


def run_one(realization_idx, N=400, T=0.4, sigma=1.0, dt=0.005,
            n_equil=2000, n_track=8000, track_every=10):
    """Run one realization, return D_rot and diagnostics."""
    coupling_seed = 42 + realization_idx * 17
    init_seed = 123 + realization_idx * 31

    params = SimulationParams(
        N=N, temperature=T, sigma=sigma, coupling_type='gaussian',
        dt=dt, coupling_seed=coupling_seed, init_seed=init_seed)

    sim = BrownianParticlesS2(params)

    # Equilibrate
    t0 = clock.time()
    sim.run(n_equil, record_every=n_equil, verbose=False)
    t_equil = clock.time() - t0

    # Check ring
    n_hat, eigvals = extract_ring_normal(sim.positions)
    sorted_ev = np.sort(eigvals)
    ratio = sorted_ev[1] / sorted_ev[0]

    if ratio < 3:
        return {'D_rot': np.nan, 'tau_c': np.nan,
                'ratio': ratio, 'failed': True,
                'time_equil': t_equil, 'time_track': 0}

    # Track orientation
    t0 = clock.time()
    n_records = n_track // track_every
    n_hats = np.zeros((n_records, 3))
    idx = 0

    for step in range(n_track):
        sim.step()
        if step % track_every == 0 and idx < n_records:
            n_hat, _ = extract_ring_normal(sim.positions)
            if idx > 0 and np.dot(n_hat, n_hats[idx-1]) < 0:
                n_hat = -n_hat
            n_hats[idx] = n_hat
            idx += 1

    t_track = clock.time() - t0
    n_hats = n_hats[:idx]
    dt_track = dt * track_every

    # Autocorrelation
    max_lag = min(300, idx // 3)
    C = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(n_hats[:idx-lag] * n_hats[lag:idx], axis=1)
        C[lag] = np.mean(dots)

    tau = np.arange(max_lag) * dt_track
    fit_mask = C > 0.2
    if np.sum(fit_mask) < 5:
        return {'D_rot': np.nan, 'tau_c': np.nan,
                'ratio': ratio, 'failed': True,
                'C': C, 'tau': tau,
                'time_equil': t_equil, 'time_track': t_track}

    log_C = np.log(np.clip(C[fit_mask], 1e-10, None))
    slope, _ = np.polyfit(tau[fit_mask], log_C, 1)
    D_rot = -slope / 2.0
    tau_c = 1.0 / (2.0 * D_rot) if D_rot > 0 else np.nan

    return {'D_rot': D_rot, 'tau_c': tau_c,
            'ratio': ratio, 'failed': False,
            'C': C, 'tau': tau, 'n_hats': n_hats,
            'time_equil': t_equil, 'time_track': t_track}


def main():
    N = 400
    n_realizations = 10

    print(f"Running {n_realizations} realizations at N = {N}")
    print(f"=" * 60)

    results = []
    for i in range(n_realizations):
        print(f"\nRealization {i+1}/{n_realizations} "
              f"(seed={42 + i*17})...", flush=True)
        r = run_one(i, N=N)
        results.append(r)

        if r['failed']:
            print(f"  FAILED (ratio={r['ratio']:.1f})")
        else:
            print(f"  D_rot = {r['D_rot']:.5f}, "
                  f"tau_c = {r['tau_c']:.1f}, "
                  f"ratio = {r['ratio']:.0f}, "
                  f"time = {r['time_equil'] + r['time_track']:.0f}s")

    # Summary
    D_rots = [r['D_rot'] for r in results if not r['failed']]
    tau_cs = [r['tau_c'] for r in results if not r['failed']]
    n_good = len(D_rots)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE SUMMARY: N = {N}, {n_good}/{n_realizations} successful")
    print(f"{'='*60}")
    print(f"D_rot: {np.mean(D_rots):.5f} +/- {np.std(D_rots):.5f} "
          f"(sem {np.std(D_rots)/np.sqrt(n_good):.5f})")
    print(f"  range: [{np.min(D_rots):.5f}, {np.max(D_rots):.5f}]")
    print(f"  median: {np.median(D_rots):.5f}")
    print(f"  coeff of variation: {np.std(D_rots)/np.mean(D_rots):.2f}")
    print(f"tau_c: {np.mean(tau_cs):.1f} +/- {np.std(tau_cs):.1f}")
    print(f"  range: [{np.min(tau_cs):.1f}, {np.max(tau_cs):.1f}]")

    # Individual results
    print(f"\nIndividual realizations:")
    print(f"{'#':>3}  {'D_rot':>10}  {'tau_c':>8}  {'ratio':>6}")
    for i, r in enumerate(results):
        if not r['failed']:
            print(f"{i+1:3d}  {r['D_rot']:10.5f}  {r['tau_c']:8.1f}  "
                  f"{r['ratio']:6.0f}")
        else:
            print(f"{i+1:3d}  {'FAILED':>10}  {'':>8}  {r['ratio']:6.1f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) D_rot histogram
    ax = axes[0]
    ax.hist(D_rots, bins=max(5, n_good//2), color='#3498db',
            edgecolor='#2980b9', alpha=0.7)
    ax.axvline(np.mean(D_rots), color='red', linewidth=2,
               label=f'Mean = {np.mean(D_rots):.4f}')
    ax.axvline(np.median(D_rots), color='green', linewidth=2,
               linestyle='--', label=f'Median = {np.median(D_rots):.4f}')
    ax.set_xlabel(r'$D_{\rm rot}$')
    ax.set_ylabel('Count')
    ax.set_title(f'(a) $D_{{rot}}$ distribution ($N={N}$, '
                 f'$n={n_good}$ realizations)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Autocorrelation for each realization
    ax = axes[1]
    for i, r in enumerate(results):
        if not r['failed'] and 'C' in r:
            ax.plot(r['tau'], r['C'], linewidth=0.8, alpha=0.6,
                    label=f"$D_{{rot}}={r['D_rot']:.4f}$" if i < 5 else None)
    ax.plot(r['tau'], np.exp(-2*np.mean(D_rots)*r['tau']), 'k--',
            linewidth=2, label=f'Mean: $e^{{-2\\cdot{np.mean(D_rots):.4f}\\tau}}$')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    ax.set_title('(b) Autocorrelation per realization')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) D_rot per realization
    ax = axes[2]
    ax.bar(range(1, n_good+1),
           sorted(D_rots), color='#3498db', edgecolor='#2980b9')
    ax.axhline(np.mean(D_rots), color='red', linewidth=1.5,
               label=f'Mean')
    ax.fill_between([0.5, n_good+0.5],
                    np.mean(D_rots)-np.std(D_rots),
                    np.mean(D_rots)+np.std(D_rots),
                    color='red', alpha=0.1, label='$\\pm 1\\sigma$')
    ax.set_xlabel('Realization (sorted)')
    ax.set_ylabel(r'$D_{\rm rot}$')
    ax.set_title('(c) Sample-to-sample variation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Ensemble of {n_good} disorder realizations '
                 f'($N = {N}$, $T = 0.4$, $\\sigma = 1$)',
                 fontsize=13)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'ensemble_N400.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

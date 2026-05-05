#!/usr/bin/env python3
"""
N-scaling study for D_rot
=========================

Runs particle simulations at N = 50, 100, 200, 300, 400, 500
and measures D_rot from the orientation autocorrelation.

For each N, runs 3 disorder realizations to estimate the mean
and spread of D_rot.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brownian_s2_simulation import BrownianParticlesS2, SimulationParams
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_ring_normal(positions):
    """Ring normal = smallest eigenvector of position covariance."""
    cov = positions.T @ positions / len(positions)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_hat = eigvecs[:, np.argmin(eigvals)]
    return n_hat, eigvals


def run_and_measure_Drot(N, T, sigma, dt, n_equil, n_track, track_every,
                          coupling_seed, init_seed):
    """Run one simulation and return D_rot."""
    params = SimulationParams(
        N=N, temperature=T, sigma=sigma, coupling_type='gaussian',
        dt=dt, coupling_seed=coupling_seed, init_seed=init_seed)

    sim = BrownianParticlesS2(params)

    # Equilibrate
    sim.run(n_equil, record_every=n_equil, verbose=False)

    # Check ring formation
    n_hat, eigvals = extract_ring_normal(sim.positions)
    sorted_ev = np.sort(eigvals)
    ratio = sorted_ev[1] / sorted_ev[0]
    if ratio < 3:
        return np.nan

    # Track orientation
    n_records = n_track // track_every
    n_hats = np.zeros((n_records, 3))
    record_idx = 0

    for step in range(n_track):
        sim.step()
        if step % track_every == 0 and record_idx < n_records:
            n_hat, _ = extract_ring_normal(sim.positions)
            if record_idx > 0 and np.dot(n_hat, n_hats[record_idx-1]) < 0:
                n_hat = -n_hat
            n_hats[record_idx] = n_hat
            record_idx += 1

    n_hats = n_hats[:record_idx]
    dt_track = dt * track_every

    # Compute autocorrelation
    max_lag = min(200, record_idx // 3)
    C = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(n_hats[:record_idx-lag] * n_hats[lag:record_idx], axis=1)
        C[lag] = np.mean(dots)

    # Fit D_rot from exponential decay
    tau = np.arange(max_lag) * dt_track
    fit_mask = C > 0.2
    if np.sum(fit_mask) < 5:
        return np.nan

    log_C = np.log(np.clip(C[fit_mask], 1e-10, None))
    slope, _ = np.polyfit(tau[fit_mask], log_C, 1)
    D_rot = -slope / 2.0

    return D_rot


def main():
    T = 0.4
    sigma = 1.0
    dt = 0.001

    N_values = [50, 100, 200, 300, 400, 500]
    n_realizations = 3

    n_track = 80000
    track_every = 20

    results = {}

    for N in N_values:
        n_equil = max(20000, N * 50)
        D_rots = []

        print(f"\n{'='*50}")
        print(f"N = {N} (equil: {n_equil} steps, track: {n_track} steps)")
        print(f"{'='*50}")

        for r in range(n_realizations):
            coupling_seed = 42 + r * 17
            init_seed = 123 + r * 31

            print(f"  Realization {r+1}/{n_realizations} "
                  f"(seed={coupling_seed})...", end=' ', flush=True)

            D_rot = run_and_measure_Drot(
                N, T, sigma, dt, n_equil, n_track, track_every,
                coupling_seed, init_seed)

            if not np.isnan(D_rot):
                print(f"D_rot = {D_rot:.4f}, tau_c = {1/(2*D_rot):.1f}")
                D_rots.append(D_rot)
            else:
                print("FAILED (no ring)")

        if D_rots:
            results[N] = {
                'D_rots': D_rots,
                'mean': np.mean(D_rots),
                'std': np.std(D_rots),
                'sem': np.std(D_rots) / np.sqrt(len(D_rots)),
                'tau_c': 1.0 / (2.0 * np.mean(D_rots)),
            }
        else:
            results[N] = {'D_rots': [], 'mean': np.nan,
                          'std': np.nan, 'sem': np.nan, 'tau_c': np.nan}

    # Print summary
    print("\n" + "=" * 60)
    print("N-SCALING SUMMARY")
    print("=" * 60)
    print(f"{'N':>5s}  {'D_rot (mean)':>12s}  {'± sem':>8s}  {'tau_c':>8s}  {'n_good':>6s}")
    print("-" * 50)
    for N in N_values:
        r = results[N]
        print(f"{N:5d}  {r['mean']:12.4f}  {r['sem']:8.4f}  "
              f"{r['tau_c']:8.1f}  {len(r['D_rots']):6d}/{n_realizations}")

    N_arr = np.array([N for N in N_values if not np.isnan(results[N]['mean'])])
    D_arr = np.array([results[N]['mean'] for N in N_values
                       if not np.isnan(results[N]['mean'])])
    D_err = np.array([results[N]['sem'] for N in N_values
                       if not np.isnan(results[N]['mean'])])

    if len(N_arr) > 2:
        log_N = np.log(N_arr)
        log_D = np.log(D_arr)
        alpha_fit, log_A = np.polyfit(log_N, log_D, 1)
        print(f"\nPower-law fit: D_rot ~ N^{alpha_fit:.2f}")
        print(f"  (theory predicts alpha = -1)")

    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(N_arr, D_arr, yerr=D_err, fmt='o-', color='#3498db',
                markersize=8, capsize=5, linewidth=2,
                label='Simulation')
    N_ref = np.linspace(40, 550, 100)
    if len(N_arr) > 0:
        D_ref = D_arr[0] * N_arr[0] / N_ref
        ax.plot(N_ref, D_ref, 'r--', linewidth=1.5,
                label=r'$\propto 1/N$ (theory)')
    if len(N_arr) > 2:
        D_fit = np.exp(log_A) * N_ref**alpha_fit
        ax.plot(N_ref, D_fit, 'g:', linewidth=1.5,
                label=f'Fit: $N^{{{alpha_fit:.2f}}}$')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$D_{\rm rot}$')
    ax.set_title(r'(a) $D_{\rm rot}$ vs $N$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(N_arr, D_arr, yerr=D_err, fmt='o', color='#3498db',
                markersize=8, capsize=5)
    if len(N_arr) > 0:
        ax.plot(N_ref, D_ref, 'r--', linewidth=1.5, label=r'$\propto 1/N$')
    if len(N_arr) > 2:
        ax.plot(N_ref, D_fit, 'g:', linewidth=1.5,
                label=f'$\\propto N^{{{alpha_fit:.2f}}}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$D_{\rm rot}$')
    ax.set_title(r'(b) Log-log scale')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(r'$N$-scaling of rotational diffusion ($T = 0.4$, $\sigma = 1$)',
                 fontsize=14)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'n_scaling_Drot.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {figpath}")
    plt.close()


if __name__ == '__main__':
    main()

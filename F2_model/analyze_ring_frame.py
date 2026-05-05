#!/usr/bin/env python3
"""
Ring-Frame Analysis: Density Profile and Longitudinal Velocity
==============================================================

Analyzes multi-snapshot data from the simulator to extract:
1. Transversal density profile f_0(theta) across the ring
2. Longitudinal angular velocity distribution omega_phi along the ring
3. Time evolution of ring normal n_hat(t) from snapshots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gaussian_profile(theta, A, mu, sigma):
    """Gaussian ring profile centered at mu."""
    return A * np.exp(-(theta - mu)**2 / (2 * sigma**2))


def main():
    # Load snapshot data
    snap_path = os.path.join(PROJECT_DIR,
                             'data/sphere/Snapshots_ringframe_sphere_n41.csv')
    print(f"Loading {snap_path}...")
    df = pd.read_csv(snap_path)

    n_snapshots = df['snapshot'].nunique()
    n_particles = df[df['snapshot'] == 0].shape[0]
    times = df.groupby('snapshot')['time'].first().values

    print(f"  {n_snapshots} snapshots, {n_particles} particles each")
    print(f"  Time range: {times[0]:.1f} to {times[-1]:.1f}")

    # ============================================================
    # 1. Aggregate density profile f_0(theta)
    # ============================================================
    all_theta = df['theta_RF'].values

    # Histogram with proper area normalization
    n_bins = 60
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = theta_bins[1] - theta_bins[0]

    counts, _ = np.histogram(all_theta, bins=theta_bins)
    # Normalize: integral of f_0(theta) * sin(theta) * dtheta = 1
    # (probability density on S^2 per unit solid angle)
    area = 2 * np.pi * np.sin(theta_centers) * dtheta
    area[area < 1e-8] = 1e-8
    total_samples = len(all_theta)
    f0 = counts / (total_samples * dtheta)  # density per unit theta
    # Also compute per solid angle
    f0_solid = counts / (total_samples * area / (2 * np.pi))

    # Fit Gaussian to f_0(theta)
    try:
        # Initial guess
        peak_idx = np.argmax(f0)
        p0 = [f0[peak_idx], theta_centers[peak_idx], 0.1]
        popt, pcov = curve_fit(gaussian_profile, theta_centers, f0,
                               p0=p0, maxfev=5000)
        A_fit, mu_fit, sigma_fit = popt
        fwhm_fit = 2.355 * abs(sigma_fit) * 180 / np.pi
        fit_success = True
        print(f"\nDensity profile f_0(theta):")
        print(f"  Peak at theta = {mu_fit * 180/np.pi:.1f} deg")
        print(f"  Gaussian sigma = {abs(sigma_fit) * 180/np.pi:.1f} deg")
        print(f"  FWHM = {fwhm_fit:.1f} deg")
    except Exception as e:
        print(f"  Gaussian fit failed: {e}")
        fit_success = False

    # ============================================================
    # 2. Longitudinal velocity distribution
    # ============================================================
    all_omega = df['omega_phi'].values

    # Restrict to ring particles (near equator in rotated frame)
    ring_mask = np.abs(df['theta_RF'].values - np.pi/2) < 0.3
    omega_ring = df.loc[ring_mask, 'omega_phi'].values

    mean_omega = np.mean(omega_ring)
    std_omega = np.std(omega_ring)
    sem_omega = std_omega / np.sqrt(len(omega_ring))

    print(f"\nLongitudinal velocity (ring particles, |theta-pi/2| < 0.3):")
    print(f"  N_ring = {len(omega_ring)} (of {len(all_omega)} total)")
    print(f"  Mean omega_phi = {mean_omega:.4f} +/- {sem_omega:.4f}")
    print(f"  Std omega_phi  = {std_omega:.3f}")
    print(f"  |mean/std| = {abs(mean_omega/std_omega):.4f}")
    print(f"  -> {'Consistent with zero drift' if abs(mean_omega/std_omega) < 0.1 else 'Possible nonzero drift'}")

    # Per-snapshot mean omega (to check for time dependence)
    omega_per_snap = []
    for s in range(n_snapshots):
        mask = (df['snapshot'] == s) & (np.abs(df['theta_RF'] - np.pi/2) < 0.3)
        omega_per_snap.append(df.loc[mask, 'omega_phi'].mean())
    omega_per_snap = np.array(omega_per_snap)

    # ============================================================
    # 3. Ring normal evolution from snapshots
    # ============================================================
    nhat_data = df.groupby('snapshot')[['nHat_x', 'nHat_y', 'nHat_z']].first().values

    # ============================================================
    # 4. Theta distribution per snapshot (stability check)
    # ============================================================
    theta_std_per_snap = []
    for s in range(n_snapshots):
        th = df.loc[df['snapshot'] == s, 'theta_RF'].values
        theta_std_per_snap.append(np.std(th))
    theta_std_per_snap = np.array(theta_std_per_snap)

    # ============================================================
    # 5. Generate figure
    # ============================================================
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) Density profile f_0(theta)
    ax = axes[0, 0]
    ax.bar(theta_centers * 180/np.pi, f0, width=dtheta * 180/np.pi * 0.9,
           color='#3498db', alpha=0.7, edgecolor='#2980b9',
           label=f'Simulation ({n_snapshots} snapshots)')
    if fit_success:
        theta_fine = np.linspace(0, np.pi, 300)
        ax.plot(theta_fine * 180/np.pi,
                gaussian_profile(theta_fine, *popt), 'r-', linewidth=2,
                label=f'Gaussian fit: $\\mu={mu_fit*180/np.pi:.1f}^\\circ$, '
                      f'FWHM$={fwhm_fit:.0f}^\\circ$')
    ax.axvline(90, color='gray', linestyle=':', alpha=0.5, label='Equator')
    ax.set_xlabel(r'$\theta$ [deg] (rotated frame)')
    ax.set_ylabel(r'$f_0(\theta)$ [density per unit $\theta$]')
    ax.set_title(r'(a) Transversal density profile $f_0(\theta)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Longitudinal velocity distribution
    ax = axes[0, 1]
    ax.hist(omega_ring, bins=60, density=True, alpha=0.7,
            color='#3498db', edgecolor='#2980b9',
            label=f'Ring particles (n={len(omega_ring)})')
    # Gaussian overlay
    x_range = np.linspace(mean_omega - 4*std_omega,
                          mean_omega + 4*std_omega, 200)
    ax.plot(x_range,
            np.exp(-(x_range - mean_omega)**2 / (2*std_omega**2)) /
            (std_omega * np.sqrt(2*np.pi)),
            'r-', linewidth=2,
            label=f'Gaussian: $\\mu={mean_omega:.3f}$, $\\sigma={std_omega:.1f}$')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(mean_omega, color='red', linestyle='-', alpha=0.7)
    ax.set_xlabel(r'$\omega_\phi$ (angular velocity about $\hat{n}$)')
    ax.set_ylabel('Probability density')
    ax.set_title(r'(b) Longitudinal velocity $\omega_\phi$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Mean omega_phi per snapshot (time series)
    ax = axes[0, 2]
    ax.plot(times, omega_per_snap, 'o-', color='#3498db',
            markersize=3, linewidth=0.8)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(np.mean(omega_per_snap), color='red', linestyle='-',
               alpha=0.7, label=f'Mean = {np.mean(omega_per_snap):.3f}')
    ax.fill_between(times,
                    np.mean(omega_per_snap) - np.std(omega_per_snap),
                    np.mean(omega_per_snap) + np.std(omega_per_snap),
                    color='red', alpha=0.1)
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\langle \omega_\phi \rangle$ per snapshot')
    ax.set_title('(c) Mean longitudinal velocity vs time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Ring normal components vs time
    ax = axes[1, 0]
    for i, (comp, label, color) in enumerate(
            zip([0, 1, 2], ['$n_x$', '$n_y$', '$n_z$'],
                ['#e74c3c', '#2ecc71', '#3498db'])):
        ax.plot(times, nhat_data[:, i], 'o-', color=color,
                markersize=3, linewidth=0.8, label=label)
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\hat{n}_i(t)$')
    ax.set_title(r'(d) Ring normal $\hat{n}(t)$ from snapshots')
    ax.legend(fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # (e) Ring width (theta std) vs time
    ax = axes[1, 1]
    ax.plot(times, theta_std_per_snap * 180/np.pi, 'o-',
            color='#9b59b6', markersize=3, linewidth=0.8)
    ax.axhline(np.mean(theta_std_per_snap) * 180/np.pi, color='red',
               linestyle='-', alpha=0.7,
               label=f'Mean = {np.mean(theta_std_per_snap)*180/np.pi:.1f}$^\\circ$')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\sigma_\theta$ [deg]')
    ax.set_title('(e) Ring width vs time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) 2D scatter: theta vs phi colored by omega_phi (one snapshot)
    ax = axes[1, 2]
    snap0 = df[df['snapshot'] == n_snapshots // 2]  # middle snapshot
    vmax = np.percentile(np.abs(snap0['omega_phi']), 95)
    scatter = ax.scatter(snap0['phi_RF'] * 180/np.pi,
                         snap0['theta_RF'] * 180/np.pi,
                         c=snap0['omega_phi'], cmap='coolwarm',
                         s=12, vmin=-vmax, vmax=vmax)
    plt.colorbar(scatter, ax=ax, label=r'$\omega_\phi$')
    ax.axhline(90, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\phi$ [deg] (along ring)')
    ax.set_ylabel(r'$\theta$ [deg] (across ring)')
    ax.set_title(f'(f) Snapshot at t = {times[n_snapshots//2]:.1f}')
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Ring-frame analysis: $f_0(\\theta)$ and $\\omega_\\phi$ '
        f'({n_snapshots} snapshots, N = {n_particles})',
        fontsize=14)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'ring_frame_analysis.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {figpath}")
    plt.close()

    # ============================================================
    # Print summary table
    # ============================================================
    print("\n" + "=" * 60)
    print("RING STRUCTURE PARAMETERS")
    print("=" * 60)
    if fit_success:
        print(f"  Ring center:    theta_0 = {mu_fit*180/np.pi:.1f} deg")
        print(f"  Gaussian width: sigma   = {abs(sigma_fit)*180/np.pi:.1f} deg")
        print(f"  FWHM:                   = {fwhm_fit:.0f} deg")
    print(f"  Ring width (std):       = {np.mean(theta_std_per_snap)*180/np.pi:.1f} deg")
    print(f"  Particles in ring:      = {np.sum(ring_mask)//n_snapshots}/{n_particles} per snapshot")
    print(f"\nLONGITUDINAL DYNAMICS")
    print(f"  Mean omega_phi (ring):  = {mean_omega:.4f} +/- {sem_omega:.4f}")
    print(f"  Std omega_phi:          = {std_omega:.2f}")
    print(f"  |mean/std|:             = {abs(mean_omega/std_omega):.4f}")
    print(f"  Drift:                  {'None detected' if abs(mean_omega/std_omega) < 0.1 else 'Possible'}")


if __name__ == '__main__':
    main()

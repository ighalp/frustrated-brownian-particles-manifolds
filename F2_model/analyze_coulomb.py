#!/usr/bin/env python3
"""
Analyze Coulomb gas simulation data and compare with F2 model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analyze_snapshots(snap_path, label):
    """Analyze ring-frame snapshots."""
    df = pd.read_csv(snap_path)
    n_snapshots = df['snapshot'].nunique()
    n_particles = df[df['snapshot'] == 0].shape[0]
    times = df.groupby('snapshot')['time'].first().values

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"  {n_snapshots} snapshots, {n_particles} particles")
    print(f"  t = {times[0]:.1f} to {times[-1]:.1f}")

    # Ring normal evolution
    nhat = df.groupby('snapshot')[['nHat_x','nHat_y','nHat_z']].first().values

    # Density profile
    all_theta = df['theta_RF'].values
    n_bins = 60
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    dtheta = theta_bins[1] - theta_bins[0]
    counts, _ = np.histogram(all_theta, bins=theta_bins)
    f0 = counts / (len(all_theta) * dtheta)

    # Gaussian fit
    try:
        peak_idx = np.argmax(f0)
        p0 = [f0[peak_idx], theta_centers[peak_idx], 0.1]
        popt, _ = curve_fit(lambda t, A, mu, s: A*np.exp(-(t-mu)**2/(2*s**2)),
                           theta_centers, f0, p0=p0, maxfev=5000)
        A, mu, sigma = popt
        fwhm = 2.355 * abs(sigma) * 180/np.pi
        print(f"  Ring center: {mu*180/np.pi:.1f} deg")
        print(f"  Gaussian sigma: {abs(sigma)*180/np.pi:.1f} deg")
        print(f"  FWHM: {fwhm:.0f} deg")
        fit_ok = True
    except:
        print(f"  Gaussian fit failed")
        fit_ok = False
        mu, sigma, fwhm = np.pi/2, 0.5, 90
        popt = None

    # Ring width per snapshot (check if ring is stable)
    widths = []
    for s in range(n_snapshots):
        th = df.loc[df['snapshot'] == s, 'theta_RF'].values
        widths.append(np.std(th))
    widths = np.array(widths)
    print(f"  Ring width (std): {np.mean(widths)*180/np.pi:.1f} +/- {np.std(widths)*180/np.pi:.1f} deg")

    # Angular velocity
    ring_mask = np.abs(df['theta_RF'] - np.pi/2) < 0.5
    omega_ring = df.loc[ring_mask, 'omega_phi'].values
    mean_omega = np.mean(omega_ring)
    std_omega = np.std(omega_ring)
    print(f"  Particles near equator: {np.sum(ring_mask)//n_snapshots}/{n_particles}")
    print(f"  Mean omega_phi: {mean_omega:.4f} +/- {std_omega/np.sqrt(len(omega_ring)):.4f}")
    print(f"  |mean/std|: {abs(mean_omega/std_omega):.4f}")

    # Anisotropy check
    eigval_ratios = []
    for s in range(n_snapshots):
        nhat_s = nhat[s]
        # The ring normal should be well-defined if ring exists
        pass

    # Check if ring exists by looking at the concentration
    fraction_near_equator = np.mean(np.abs(all_theta - np.pi/2) < 0.3)
    print(f"  Fraction within 0.3 rad of equator: {fraction_near_equator:.2f}")
    is_ring = fraction_near_equator > 0.7
    print(f"  Ring formed: {'YES' if is_ring else 'NO'}")

    return {
        'theta_centers': theta_centers, 'f0': f0,
        'popt': popt, 'fit_ok': fit_ok,
        'times': times, 'nhat': nhat, 'widths': widths,
        'omega_ring': omega_ring, 'is_ring': is_ring,
        'fwhm': fwhm if fit_ok else np.nan,
        'all_theta': all_theta,
    }


def analyze_lseries(lpath, label):
    """Analyze L-series for orientation dynamics."""
    df = pd.read_csv(lpath)
    time = df['time'].values
    Lx, Ly, Lz = df['Lx'].values, df['Ly'].values, df['Lz'].values
    L_mag = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    valid = L_mag > 1e-10
    n_hats = np.zeros((len(time), 3))
    n_hats[valid, 0] = Lx[valid] / L_mag[valid]
    n_hats[valid, 1] = Ly[valid] / L_mag[valid]
    n_hats[valid, 2] = Lz[valid] / L_mag[valid]

    # Skip early
    start = np.searchsorted(time, 10.0)
    time = time[start:]
    n_hats = n_hats[start:]
    dt = time[1] - time[0] if len(time) > 1 else 0.1
    N = len(time)

    # Autocorrelation
    max_lag = min(150, N // 3)
    tau = np.arange(max_lag) * dt
    C = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(n_hats[:N-lag] * n_hats[lag:N], axis=1)
        C[lag] = np.mean(dots)

    # Fit D_rot
    fit_mask = C > 0.1
    if np.sum(fit_mask) > 5:
        log_C = np.log(np.clip(C[fit_mask], 1e-10, None))
        slope, _ = np.polyfit(tau[fit_mask], log_C, 1)
        D_rot = -slope / 2.0
        tau_c = 1.0 / (2.0 * D_rot)
    else:
        D_rot = np.nan
        tau_c = np.nan

    print(f"  D_rot (from L/|L|): {D_rot:.5f}")
    print(f"  tau_c: {tau_c:.1f}")

    return {
        'time': time, 'n_hats': n_hats,
        'tau': tau, 'C': C, 'D_rot': D_rot, 'tau_c': tau_c,
    }


def main():
    # Paths
    f2_snap = os.path.join(PROJECT_DIR, 'data/sphere/Snapshots_ringframe_sphere_n41.csv')
    f2_lseries = os.path.join(PROJECT_DIR, 'data/sphere/L_series_3D_sphere.csv')

    cg_snap = os.path.join(PROJECT_DIR, 'data/sphere/Snapshots_ringframe_sphere_n41_Coulomb_gas.csv')
    cg_lseries = os.path.join(PROJECT_DIR, 'data/sphere/L_series_3D_sphere_Coulomb_gas.csv')
    cg_coords = os.path.join(PROJECT_DIR, 'data/sphere/Coords_sphere_t50_00-5_Coulomb_gas.csv')

    # Analyze F2 model
    print("ANALYZING F2 MODEL (linear potential, T=0.4)")
    r_f2 = analyze_snapshots(f2_snap, "F2: Ring-frame snapshots")
    l_f2 = analyze_lseries(f2_lseries, "F2: L-series")

    # Analyze Coulomb gas
    print("\n\nANALYZING COULOMB GAS (log potential, d_min=3.0, T=0.2)")
    r_cg = analyze_snapshots(cg_snap, "Coulomb: Ring-frame snapshots")
    l_cg = analyze_lseries(cg_lseries, "Coulomb: L-series")

    # Also check Coords snapshot for ring structure
    print("\n  Coords snapshot analysis:")
    df_coords = pd.read_csv(cg_coords)
    pos = df_coords[['x','y','z']].values
    cov = pos.T @ pos / len(pos)
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    ratio = eigvals[1] / eigvals[0]
    print(f"    Eigenvalues: {eigvals}")
    print(f"    Anisotropy ratio: {ratio:.1f}")
    print(f"    {'Ring detected' if ratio > 3 else 'No clear ring'}")

    # ============================================================
    # Generate comparison figure
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) Density profile comparison
    ax = axes[0, 0]
    ax.bar(r_f2['theta_centers']*180/np.pi, r_f2['f0'],
           width=3, alpha=0.5, color='#3498db', label='F2 (linear)')
    ax.bar(r_cg['theta_centers']*180/np.pi, r_cg['f0'],
           width=3, alpha=0.5, color='#e74c3c', label='Coulomb (log)')
    ax.axvline(90, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$f_0(\theta)$')
    ax.set_title('(a) Density profile comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Ring width vs time
    ax = axes[0, 1]
    ax.plot(r_f2['times'], r_f2['widths']*180/np.pi, 'o-',
            color='#3498db', markersize=3, label='F2')
    ax.plot(r_cg['times'], r_cg['widths']*180/np.pi, 's-',
            color='#e74c3c', markersize=3, label='Coulomb')
    ax.set_xlabel('$t$')
    ax.set_ylabel(r'$\sigma_\theta$ [deg]')
    ax.set_title('(b) Ring width vs time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Autocorrelation comparison
    ax = axes[0, 2]
    ax.plot(l_f2['tau'], l_f2['C'], color='#3498db', linewidth=2,
            label=f"F2: $D_{{rot}}={l_f2['D_rot']:.4f}$")
    ax.plot(l_cg['tau'], l_cg['C'], color='#e74c3c', linewidth=2,
            label=f"Coulomb: $D_{{rot}}={l_cg['D_rot']:.4f}$")
    if not np.isnan(l_f2['D_rot']):
        ax.plot(l_f2['tau'], np.exp(-2*l_f2['D_rot']*l_f2['tau']),
                'b--', linewidth=1, alpha=0.5)
    if not np.isnan(l_cg['D_rot']):
        ax.plot(l_cg['tau'], np.exp(-2*l_cg['D_rot']*l_cg['tau']),
                'r--', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C(\tau)$')
    ax.set_title('(c) Orientation autocorrelation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) n_z(t) comparison
    ax = axes[1, 0]
    ax.plot(l_f2['time'], l_f2['n_hats'][:, 2], color='#3498db',
            linewidth=0.5, alpha=0.7, label='F2')
    ax.plot(l_cg['time'], l_cg['n_hats'][:, 2], color='#e74c3c',
            linewidth=0.5, alpha=0.7, label='Coulomb')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$n_z(t)$')
    ax.set_title('(d) $n_z$ component')
    ax.legend(fontsize=9)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # (e) Angular velocity distribution
    ax = axes[1, 1]
    ax.hist(r_f2['omega_ring'], bins=50, density=True, alpha=0.5,
            color='#3498db', label='F2')
    ax.hist(r_cg['omega_ring'], bins=50, density=True, alpha=0.5,
            color='#e74c3c', label='Coulomb')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\omega_\phi$')
    ax.set_ylabel('Density')
    ax.set_title(r'(e) Longitudinal velocity $\omega_\phi$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) Theta distribution (histogram)
    ax = axes[1, 2]
    ax.hist(r_f2['all_theta']*180/np.pi, bins=60, density=True, alpha=0.5,
            color='#3498db', label='F2')
    ax.hist(r_cg['all_theta']*180/np.pi, bins=60, density=True, alpha=0.5,
            color='#e74c3c', label='Coulomb')
    ax.axvline(90, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel('Density')
    ax.set_title(r'(f) $\theta$ distribution (all snapshots)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('F2 model (linear, T=0.4) vs Coulomb gas (log, d_min=3.0, T=0.2)',
                 fontsize=14)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'coulomb_comparison.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {figpath}")
    plt.close()

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'':20s}  {'F2 (linear)':>15s}  {'Coulomb (log)':>15s}")
    print("-"*55)
    print(f"{'Ring formed':20s}  {'YES' if r_f2['is_ring'] else 'NO':>15s}  {'YES' if r_cg['is_ring'] else 'NO':>15s}")
    print(f"{'FWHM [deg]':20s}  {r_f2['fwhm']:15.0f}  {r_cg['fwhm']:15.0f}")
    print(f"{'D_rot':20s}  {l_f2['D_rot']:15.5f}  {l_cg['D_rot']:15.5f}")
    print(f"{'tau_c':20s}  {l_f2['tau_c']:15.1f}  {l_cg['tau_c']:15.1f}")


if __name__ == '__main__':
    main()

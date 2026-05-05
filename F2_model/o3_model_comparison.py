#!/usr/bin/env python3
"""
O(3) Model Direct Simulation and Comparison with Particle Simulations
=====================================================================

1. Loads 3D angular momentum data (Lx, Ly, Lz) from particle simulation
2. Computes orientation n_hat(t) = L(t)/|L(t)|
3. Simulates the O(3) SDE with fitted D_rot
4. Compares observables
5. Extracts ring profile rho(theta) and angular speed from Coords snapshot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# O(3) SDE SIMULATION
# ============================================================
def simulate_o3_sde(D_rot, dt, n_steps, n_hat0=None, seed=42):
    """Brownian motion on S^2 (Ito form)."""
    rng = np.random.default_rng(seed)
    if n_hat0 is None:
        n_hat0 = np.array([0.0, 0.0, 1.0])
    n_hats = np.zeros((n_steps + 1, 3))
    n_hats[0] = n_hat0 / np.linalg.norm(n_hat0)
    sqrt_2D_dt = np.sqrt(2 * D_rot * dt)
    for k in range(n_steps):
        n = n_hats[k]
        dW = rng.standard_normal(3)
        dW_tan = dW - np.dot(dW, n) * n
        drift = -2 * D_rot * n * dt
        n_new = n + drift + sqrt_2D_dt * dW_tan
        n_hats[k + 1] = n_new / np.linalg.norm(n_new)
    return n_hats


# ============================================================
# OBSERVABLES
# ============================================================
def compute_autocorrelation(n_hats, max_lag):
    N = len(n_hats)
    max_lag = min(max_lag, N // 3)
    C = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(n_hats[:N - lag] * n_hats[lag:N], axis=1)
        C[lag] = np.mean(dots)
    return C


def compute_msd_chord(n_hats, max_lag):
    N = len(n_hats)
    max_lag = min(max_lag, N // 3)
    msd = np.zeros(max_lag)
    for lag in range(max_lag):
        diff = n_hats[lag:N] - n_hats[:N - lag]
        msd[lag] = np.mean(np.sum(diff**2, axis=1))
    return msd


# ============================================================
# RING PROFILE AND ANGULAR SPEED FROM COORDS SNAPSHOT
# ============================================================
def analyze_ring_structure(coords_path):
    """
    From a snapshot of particle positions and velocities:
    1. Find ring normal n_hat via inertia tensor
    2. Rotate to frame where n_hat = z-axis
    3. Compute density profile rho(theta)
    4. Compute angular speed along the ring
    """
    df = pd.read_csv(coords_path)
    pos = df[['x', 'y', 'z']].values
    vel = df[['vx', 'vy', 'vz']].values
    N = len(pos)

    # --- Ring normal from inertia tensor ---
    cov = pos.T @ pos / N
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Ring normal = smallest eigenvalue direction
    idx_sort = np.argsort(eigvals)
    n_hat = eigvecs[:, idx_sort[0]]
    # Ensure n_hat points "up" (positive z component)
    if n_hat[2] < 0:
        n_hat = -n_hat

    print(f"Ring normal: n_hat = ({n_hat[0]:.3f}, {n_hat[1]:.3f}, {n_hat[2]:.3f})")
    print(f"Eigenvalues: {eigvals[idx_sort]}")
    print(f"Anisotropy ratio: {eigvals[idx_sort[1]]/eigvals[idx_sort[0]]:.1f}")

    # --- Rotation matrix to align n_hat with z-axis ---
    z_axis = np.array([0, 0, 1.0])
    if np.allclose(n_hat, z_axis):
        R = np.eye(3)
    elif np.allclose(n_hat, -z_axis):
        R = np.diag([1, -1, -1])
    else:
        v = np.cross(n_hat, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(n_hat, z_axis)
        vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / s**2

    # Rotate positions and velocities
    pos_rot = (R @ pos.T).T
    vel_rot = (R @ vel.T).T

    # --- Spherical coordinates in rotated frame ---
    r = np.linalg.norm(pos_rot, axis=1)
    theta = np.arccos(np.clip(pos_rot[:, 2] / r, -1, 1))  # polar angle from n_hat
    phi = np.arctan2(pos_rot[:, 1], pos_rot[:, 0])         # azimuthal angle

    # --- Density profile rho(theta) ---
    theta_bins = np.linspace(0, np.pi, 37)  # 5-degree bins
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    # Weight by 1/sin(theta) to get density per unit angle
    counts, _ = np.histogram(theta, bins=theta_bins)
    # Area element: 2*pi*sin(theta)*dtheta
    dtheta = theta_bins[1] - theta_bins[0]
    area = 2 * np.pi * np.sin(theta_centers) * dtheta
    # Avoid division by zero at poles
    area[area < 1e-6] = 1e-6
    rho_theta = counts / (area * N)  # normalize so integral = 1

    # --- Angular velocity along the ring ---
    # In rotated frame, angular velocity about n_hat (= z-axis):
    # omega_z = (x * vy - y * vx) / r^2 for each particle
    x_rot, y_rot, z_rot = pos_rot.T
    vx_rot, vy_rot, vz_rot = vel_rot.T

    r_perp_sq = x_rot**2 + y_rot**2
    valid = r_perp_sq > 1e-6
    omega_z = np.zeros(N)
    omega_z[valid] = (x_rot[valid] * vy_rot[valid] -
                      y_rot[valid] * vx_rot[valid]) / r_perp_sq[valid]

    # Also compute omega for ring particles only (near equator)
    in_ring = np.abs(theta - np.pi/2) < 0.3  # within ~17 deg of equator
    omega_ring = omega_z[in_ring]

    print(f"\nParticles in ring (|theta - pi/2| < 0.3): {np.sum(in_ring)}/{N}")
    print(f"Mean angular velocity (ring): {np.mean(omega_ring):.4f} +/- {np.std(omega_ring)/np.sqrt(len(omega_ring)):.4f}")
    print(f"Std angular velocity (ring): {np.std(omega_ring):.4f}")
    print(f"Mean angular velocity (all): {np.mean(omega_z):.4f}")

    # Ring width (FWHM)
    peak_idx = np.argmax(rho_theta)
    half_max = rho_theta[peak_idx] / 2
    above_half = rho_theta > half_max
    if np.any(above_half):
        fwhm = dtheta * np.sum(above_half) * 180 / np.pi
    else:
        fwhm = np.nan
    print(f"Ring center: theta = {theta_centers[peak_idx]*180/np.pi:.1f} deg")
    print(f"Ring FWHM: {fwhm:.1f} deg")

    return {
        'n_hat': n_hat,
        'theta': theta,
        'theta_centers': theta_centers,
        'rho_theta': rho_theta,
        'omega_z': omega_z,
        'omega_ring': omega_ring,
        'in_ring': in_ring,
        'pos_rot': pos_rot,
        'fwhm': fwhm,
        'N': N,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    # Parameters
    D_rot = 0.028
    T_total = 50.0
    dt_sde = 0.1  # Match particle data time step
    n_steps = int(T_total / dt_sde)

    # ============================================================
    # 1. Load particle simulation data
    # ============================================================
    l_series_path = os.path.join(PROJECT_DIR, 'data/sphere/L_series_3D_sphere.csv')
    coords_path = os.path.join(PROJECT_DIR, 'data/sphere/Coords_sphere_t50_17.csv')

    print("Loading 3D angular momentum data...")
    df = pd.read_csv(l_series_path)
    time_sim = df['time'].values
    Lx = df['Lx'].values
    Ly = df['Ly'].values
    Lz = df['Lz'].values
    L_mag = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Compute n_hat = L / |L|
    valid = L_mag > 1e-10
    n_hats_sim = np.zeros((len(time_sim), 3))
    n_hats_sim[valid, 0] = Lx[valid] / L_mag[valid]
    n_hats_sim[valid, 1] = Ly[valid] / L_mag[valid]
    n_hats_sim[valid, 2] = Lz[valid] / L_mag[valid]

    # Skip t=0 where L=0
    start_idx = np.argmax(valid)
    time_sim = time_sim[start_idx:]
    n_hats_sim = n_hats_sim[start_idx:]
    dt_sim = time_sim[1] - time_sim[0]

    print(f"  {len(time_sim)} data points, dt = {dt_sim:.2f}, "
          f"t = {time_sim[0]:.1f} to {time_sim[-1]:.1f}")

    # Fit D_rot from simulation autocorrelation
    max_lag_sim = min(150, len(time_sim) // 3)
    tau_sim = np.arange(max_lag_sim) * dt_sim
    C_sim = compute_autocorrelation(n_hats_sim, max_lag_sim)
    msd_sim = compute_msd_chord(n_hats_sim, max_lag_sim)

    fit_range = C_sim > 0.2
    if np.sum(fit_range) > 5:
        tau_fit = tau_sim[fit_range]
        log_C = np.log(np.clip(C_sim[fit_range], 1e-10, None))
        slope, _ = np.polyfit(tau_fit, log_C, 1)
        D_rot_sim = -slope / 2
        tau_c_sim = 1 / (2 * D_rot_sim)
    else:
        D_rot_sim = D_rot
        tau_c_sim = 1 / (2 * D_rot)

    print(f"  D_rot from C(tau) fit: {D_rot_sim:.4f}")
    print(f"  tau_c = {tau_c_sim:.1f}")

    # Use the fitted D_rot for the O(3) model
    D_rot_use = D_rot_sim

    # ============================================================
    # 2. Simulate O(3) model
    # ============================================================
    print(f"\nSimulating O(3) model with D_rot = {D_rot_use:.4f}...")
    n_realizations = 10
    all_n_hats_sde = []
    for i in range(n_realizations):
        n_hats_sde = simulate_o3_sde(D_rot_use, dt_sim, n_steps, seed=42 + i * 13)
        all_n_hats_sde.append(n_hats_sde)

    times_sde = np.arange(n_steps + 1) * dt_sim

    # Ensemble observables
    max_lag_sde = max_lag_sim
    tau_sde = np.arange(max_lag_sde) * dt_sim
    C_sde_all = np.array([compute_autocorrelation(nh, max_lag_sde) for nh in all_n_hats_sde])
    msd_sde_all = np.array([compute_msd_chord(nh, max_lag_sde) for nh in all_n_hats_sde])
    C_sde_mean = np.mean(C_sde_all, axis=0)
    C_sde_std = np.std(C_sde_all, axis=0)
    msd_sde_mean = np.mean(msd_sde_all, axis=0)
    msd_sde_std = np.std(msd_sde_all, axis=0)

    # Theory
    C_theory = np.exp(-2 * D_rot_use * tau_sde)
    msd_theory = 2 * (1 - np.exp(-2 * D_rot_use * tau_sde))

    # Power spectra
    nperseg = min(256, len(n_hats_sim) // 4)
    fs = 1.0 / dt_sim
    freq_sim, psd_sim = welch(n_hats_sim[:, 2], fs=fs, nperseg=nperseg)
    freq_sde, psd_sde = welch(all_n_hats_sde[0][:, 2], fs=fs, nperseg=nperseg)

    # ============================================================
    # 3. Ring structure analysis
    # ============================================================
    print(f"\nAnalyzing ring structure from {coords_path}...")
    ring = analyze_ring_structure(coords_path)

    # ============================================================
    # 4. Generate figure (2 rows x 3 columns)
    # ============================================================
    print("\nGenerating comparison figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    component_labels = [r'$n_x(t)$', r'$n_y(t)$', r'$n_z(t)$']
    component_titles = ['(a) $n_x(t)$', '(b) $n_y(t)$', '(c) $n_z(t)$']

    # --- Row 1: components n_x, n_y, n_z ---
    for i in range(3):
        ax = axes[0, i]
        ax.plot(time_sim, n_hats_sim[:, i], color='#3498db', linewidth=0.8,
                alpha=0.8, label='Particle sim. (inertia tensor)')
        ax.plot(times_sde, all_n_hats_sde[0][:, i], color='#e74c3c',
                linewidth=0.8, alpha=0.6, label='O(3) model SDE')
        ax.set_xlabel('$t$')
        ax.set_ylabel(component_labels[i])
        ax.set_title(component_titles[i])
        if i == 0:
            ax.legend(fontsize=8, loc='lower left')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    # --- Row 2: (d) autocorrelation, (e) MSD, (f) power spectrum ---

    # (d) Autocorrelation
    ax = axes[1, 0]
    ax.plot(tau_sim, C_sim, color='#3498db', linewidth=2,
            label='Particle sim.')
    ax.fill_between(tau_sde, C_sde_mean - C_sde_std,
                    C_sde_mean + C_sde_std, color='#e74c3c', alpha=0.2)
    ax.plot(tau_sde, C_sde_mean, color='#e74c3c', linewidth=1.5,
            label='O(3) model')
    ax.plot(tau_sde, C_theory, 'k--', linewidth=1,
            label=fr'$e^{{-2D_{{\rm rot}}\tau}}$, $D_{{\rm rot}}={D_rot_use:.4f}$')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$q(\tau) = \langle\hat{n}(t)\cdot\hat{n}(t+\tau)\rangle$')
    ax.set_title(fr'(d) Autocorrelation ($\tau_c = {tau_c_sim:.0f}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) MSD (chord distance)
    ax = axes[1, 1]
    ax.plot(tau_sim[1:], msd_sim[1:], color='#3498db', linewidth=2,
            label='Particle sim.')
    ax.fill_between(tau_sde[1:], (msd_sde_mean - msd_sde_std)[1:],
                    (msd_sde_mean + msd_sde_std)[1:],
                    color='#e74c3c', alpha=0.2)
    ax.plot(tau_sde[1:], msd_sde_mean[1:], color='#e74c3c',
            linewidth=1.5, label='O(3) model')
    ax.plot(tau_sde[1:], msd_theory[1:], 'k--', linewidth=1,
            label=r'$2(1 - e^{-2D_{\rm rot}\tau})$')
    ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5,
               label='Saturation (= 2)')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'MSD$(\tau)$')
    ax.set_title('(e) Mean squared displacement')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Power spectrum of n_z
    ax = axes[1, 2]
    ax.loglog(freq_sim[1:], psd_sim[1:], color='#3498db', linewidth=1.5,
              alpha=0.8, label='Particle sim.')
    ax.loglog(freq_sde[1:], psd_sde[1:], color='#e74c3c', linewidth=1,
              alpha=0.7, label='O(3) model')
    gamma_lor = 2 * D_rot_use
    psd_lor = (2 * gamma_lor) / (gamma_lor**2 + (2 * np.pi * freq_sde[1:])**2)
    psd_lor *= psd_sde[1] / psd_lor[0]
    ax.loglog(freq_sde[1:], psd_lor, 'k--', linewidth=1,
              label=r'Lorentzian ($\sim\omega^{-2}$)')
    ax.set_xlabel(r'frequency $f$')
    ax.set_ylabel(r'$S_{n_z}(f)$')
    ax.set_title('(f) Power spectrum of $n_z$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Long-run analysis: N={ring["N"]}, T=0.4, t={time_sim[-1]:.1f}, '
        fr'$D_{{\rm rot}}={D_rot_use:.4f}$, $\tau_c={tau_c_sim:.0f}$ '
        '(inertia tensor method)',
        fontsize=13)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'o3_model_comparison.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {figpath}")
    plt.close()

    # ============================================================
    # 5. Print summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: O(3) MODEL PARAMETERS FROM SIMULATION")
    print("=" * 60)
    print(f"\nOrientation dynamics (from L-series):")
    print(f"  D_rot = {D_rot_use:.4f}")
    print(f"  tau_c = {tau_c_sim:.1f}")
    print(f"  kappa = 1/(2*D_rot) = {1/(2*D_rot_use):.1f}")
    print(f"\nRing structure (from Coords snapshot):")
    print(f"  N particles = {ring['N']}")
    print(f"  Ring FWHM = {ring['fwhm']:.0f} deg")
    print(f"  Particles in ring = {np.sum(ring['in_ring'])}")
    print(f"\nAngular velocity along ring:")
    print(f"  Mean omega_z (ring) = {np.mean(ring['omega_ring']):.4f} "
          f"+/- {np.std(ring['omega_ring'])/np.sqrt(len(ring['omega_ring'])):.4f}")
    print(f"  Std omega_z (ring)  = {np.std(ring['omega_ring']):.3f}")
    print(f"  Mean omega_z (all)  = {np.mean(ring['omega_z']):.4f}")
    print(f"  |mean/std| = {abs(np.mean(ring['omega_ring'])/np.std(ring['omega_ring'])):.3f}")
    print(f"  -> {'Consistent with zero' if abs(np.mean(ring['omega_ring'])/np.std(ring['omega_ring'])) < 0.3 else 'Significant'} mean drift")


if __name__ == '__main__':
    main()

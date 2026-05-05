#!/usr/bin/env python3
"""
Extract the memory kernel kappa(tau) from particle simulation data.

Method:
  1. Compute n_hat(t) = L(t)/|L(t)| from the 3D angular momentum
  2. Compute velocity v(t) = dn_hat/dt by finite differences
  3. Compute the velocity autocorrelation function (VACF)
  4. Compute the velocity power spectrum S_v(omega)
  5. Invert: kappa_hat(omega) = 2 / S_v(omega)
  6. Inverse FFT to get kappa(tau)

For the Markovian case, kappa(tau) = kappa * delta(tau), so
kappa_hat(omega) = const and S_v is white noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Load 3D angular momentum data
    csv_path = os.path.join(PROJECT_DIR,
                            'data/sphere/L_series_3D_sphere.csv')
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    time = df['time'].values
    Lx, Ly, Lz = df['Lx'].values, df['Ly'].values, df['Lz'].values
    L_mag = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    # Compute n_hat = L/|L|
    valid = L_mag > 1e-10
    n_hats = np.zeros((len(time), 3))
    n_hats[valid, 0] = Lx[valid] / L_mag[valid]
    n_hats[valid, 1] = Ly[valid] / L_mag[valid]
    n_hats[valid, 2] = Lz[valid] / L_mag[valid]

    # Skip formation phase (use t > 10)
    start_idx = np.searchsorted(time, 10.0)
    time = time[start_idx:]
    n_hats = n_hats[start_idx:]
    dt = time[1] - time[0]
    N = len(time)

    print(f"  Using {N} points from t = {time[0]:.1f} to {time[-1]:.1f}, dt = {dt:.2f}")

    # ============================================================
    # 1. Compute velocity by finite differences
    # ============================================================
    v = np.diff(n_hats, axis=0) / dt  # (N-1, 3)
    t_v = 0.5 * (time[:-1] + time[1:])  # midpoints

    # ============================================================
    # 2. Velocity autocorrelation function (VACF)
    # ============================================================
    max_lag = min(100, len(v) // 3)
    tau_vals = np.arange(max_lag) * dt
    Cv = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(v[:len(v)-lag] * v[lag:], axis=1)
        Cv[lag] = np.mean(dots)

    # For comparison: Markovian prediction
    # VACF for Markovian diffusion on S^2:
    # C_v(tau) = 4 D_rot * delta(tau)  (white noise)
    # In discrete time: C_v(0) = 4 D_rot / dt, C_v(lag>0) ~ 0

    # Fit D_rot from orientation autocorrelation
    Cq = np.zeros(max_lag)
    for lag in range(max_lag):
        dots = np.sum(n_hats[:N-lag] * n_hats[lag:N], axis=1)
        Cq[lag] = np.mean(dots)

    fit_mask = Cq > 0.2
    if np.sum(fit_mask) > 5:
        log_Cq = np.log(np.clip(Cq[fit_mask], 1e-10, None))
        slope, _ = np.polyfit(tau_vals[fit_mask], log_Cq, 1)
        D_rot = -slope / 2
    else:
        D_rot = 0.028
    print(f"  D_rot from q(tau): {D_rot:.4f}")

    # ============================================================
    # 3. Velocity power spectrum
    # ============================================================
    fs = 1.0 / dt
    nperseg = min(128, len(v) // 4)

    # Sum over 3 embedding-space components
    freq, Sv_total = np.zeros(nperseg // 2 + 1), np.zeros(nperseg // 2 + 1)
    for comp in range(3):
        f, psd = welch(v[:, comp], fs=fs, nperseg=nperseg)
        Sv_total += psd
    freq = f

    # Also compute n_z power spectrum for comparison
    freq_n, Sn = welch(n_hats[:, 2], fs=fs, nperseg=nperseg)

    # ============================================================
    # 4. Memory kernel in frequency domain
    # ============================================================
    # kappa_hat(omega) = 2 / S_v(omega)
    # (factor 2 for the 2 tangent-plane degrees of freedom;
    #  S_v_total sums 3 embedding components, but constraint
    #  n.dn = 0 means effectively 2 DOF)
    # Use S_v_total directly (it includes the 2 tangent DOF
    # plus the constrained component which is ~0)
    kappa_hat = 2.0 / np.clip(Sv_total[1:], 1e-20, None)
    freq_k = freq[1:]

    # Markovian prediction: kappa_hat = kappa = 1/(D_rot)
    # (since D_rot = 1/(2*kappa) => kappa = 1/(2*D_rot)...
    # wait, need to be careful with conventions)
    # From the action S = kappa/2 * int |dn|^2 dt,
    # the MSRJD relation gives D_rot = 1/(2*kappa).
    # The Markovian VACF: S_v(omega) = 2/kappa = 4*D_rot
    kappa_markov = 1.0 / (2 * D_rot)
    Sv_markov = 4 * D_rot  # constant (white noise)

    # ============================================================
    # 5. Memory kernel in time domain (inverse FFT)
    # ============================================================
    # Use the full (two-sided) FFT approach
    # Compute VACF -> FFT -> invert
    # kappa(tau) from VACF: kappa_hat = 2/S_v, then IFFT

    # Pad VACF symmetrically
    Cv_sym = np.concatenate([Cv, Cv[-2:0:-1]])  # symmetric extension
    Cv_fft = np.fft.rfft(Cv_sym)

    # kappa in frequency domain from VACF
    # S_v(omega) = FFT of C_v(tau) (Wiener-Khinchin)
    # kappa_hat = 2 / S_v
    Sv_from_vacf = np.abs(Cv_fft) * dt  # normalize
    Sv_from_vacf[Sv_from_vacf < 1e-20] = 1e-20
    kappa_hat_from_vacf = 2.0 / Sv_from_vacf

    # Inverse FFT to get kappa(tau)
    kappa_tau = np.fft.irfft(kappa_hat_from_vacf) / dt
    tau_kappa = np.arange(len(kappa_tau)) * dt

    # ============================================================
    # 6. Generate figure
    # ============================================================
    print("Generating figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) VACF
    ax = axes[0, 0]
    ax.plot(tau_vals, Cv, 'b-', linewidth=2, label='Simulation')
    # Markovian: delta function (show as spike at tau=0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.plot(0, Cv[0], 'ro', markersize=10,
            label=f'$C_v(0) = {Cv[0]:.2f}$')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C_v(\tau) = \langle \dot{\hat{n}}(t) \cdot \dot{\hat{n}}(t+\tau) \rangle$')
    ax.set_title('(a) Velocity autocorrelation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Velocity power spectrum
    ax = axes[0, 1]
    ax.loglog(freq[1:], Sv_total[1:], 'b-', linewidth=1.5,
              label=r'$S_v(\omega)$ (simulation)')
    ax.axhline(Sv_markov, color='r', linestyle='--', linewidth=1.5,
               label=f'Markovian: $4D_{{rot}} = {Sv_markov:.3f}$')
    # Power-law fit
    mid = len(freq) // 4
    if mid > 2:
        log_f = np.log(freq[2:mid])
        log_S = np.log(Sv_total[2:mid])
        beta, _ = np.polyfit(log_f, log_S, 1)
        ax.loglog(freq[2:mid],
                  np.exp(_) * freq[2:mid]**beta, 'g:',
                  linewidth=2,
                  label=f'Fit: $\\omega^{{{beta:.2f}}}$')
    ax.set_xlabel(r'$f$')
    ax.set_ylabel(r'$S_v(f)$')
    ax.set_title(r'(b) Velocity power spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) n_z power spectrum (for reference)
    ax = axes[0, 2]
    ax.loglog(freq_n[1:], Sn[1:], 'b-', linewidth=1.5,
              label=r'$S_{n_z}(\omega)$')
    # Lorentzian reference
    gamma = 2 * D_rot
    Sn_lor = (2*gamma) / (gamma**2 + (2*np.pi*freq_n[1:])**2)
    Sn_lor *= Sn[1] / Sn_lor[0]
    ax.loglog(freq_n[1:], Sn_lor, 'r--', linewidth=1,
              label=r'Lorentzian ($\omega^{-2}$)')
    ax.set_xlabel(r'$f$')
    ax.set_ylabel(r'$S_{n_z}(f)$')
    ax.set_title(r'(c) Orientation power spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Memory kernel in frequency domain
    ax = axes[1, 0]
    ax.loglog(freq_k, kappa_hat, 'b-', linewidth=1.5,
              label=r'$\hat{\kappa}(\omega) = 2/S_v(\omega)$')
    ax.axhline(kappa_markov, color='r', linestyle='--',
               linewidth=1.5,
               label=f'Markovian: $\\kappa = {kappa_markov:.1f}$')
    ax.set_xlabel(r'$f$')
    ax.set_ylabel(r'$\hat{\kappa}(f)$')
    ax.set_title(r'(d) Memory kernel $\hat{\kappa}(\omega)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Memory kernel in time domain
    ax = axes[1, 1]
    n_show = min(30, len(kappa_tau) // 2)
    ax.plot(tau_kappa[:n_show], kappa_tau[:n_show], 'b-o',
            linewidth=1.5, markersize=4, label=r'$\kappa(\tau)$')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$\kappa(\tau)$')
    ax.set_title(r'(e) Memory kernel $\kappa(\tau)$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) Orientation autocorrelation with non-Markovian fit
    ax = axes[1, 2]
    ax.plot(tau_vals, Cq, 'b-', linewidth=2, label='Simulation')
    ax.plot(tau_vals, np.exp(-2*D_rot*tau_vals), 'r--',
            linewidth=1.5,
            label=f'Markovian: $e^{{-2D_{{rot}}\\tau}}$')
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$q(\tau)$')
    ax.set_title(r'(f) Orientation autocorrelation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Memory kernel extraction ($N = 400$, $T = 0.4$, '
        f'$D_{{rot}} = {D_rot:.3f}$)',
        fontsize=14)
    plt.tight_layout()

    figpath = os.path.join(PROJECT_DIR, 'memory_kernel.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {figpath}")
    plt.close()

    # ============================================================
    # Print summary
    # ============================================================
    print("\n" + "=" * 60)
    print("MEMORY KERNEL ANALYSIS")
    print("=" * 60)
    print(f"\nD_rot = {D_rot:.4f}")
    print(f"Markovian kappa = 1/(2*D_rot) = {kappa_markov:.1f}")
    print(f"VACF at tau=0: C_v(0) = {Cv[0]:.3f}")
    print(f"Markovian prediction: C_v(0) = 4*D_rot/dt = {4*D_rot/dt:.3f}")
    print(f"VACF decay to 1/e: ~ {tau_vals[np.searchsorted(-Cv[:max_lag//2], -Cv[0]/np.e)] if Cv[0] > 0 else 'N/A':.2f}")

    # Check if velocity is white noise (Markovian test)
    Cv_ratio = Cv[1] / Cv[0] if Cv[0] > 0 else 0
    print(f"\nMarkovianity test:")
    print(f"  C_v(dt)/C_v(0) = {Cv_ratio:.4f}")
    print(f"  (= 0 for white noise, > 0 for colored noise)")
    if abs(Cv_ratio) > 0.1:
        print(f"  -> NON-MARKOVIAN: velocity has significant memory")
    else:
        print(f"  -> Approximately Markovian")


if __name__ == '__main__':
    main()

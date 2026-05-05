"""
Density-density correlator on S^2 for the F2 ring-formed state.

Reconstructs a smooth density field rho(x,t) from particle positions using a
Fisher-von Mises (spherical Gaussian) KDE, then evaluates the unequal-time
correlator C_rho(gamma, tau) = <rho(x,t) rho(x', t+tau)>, where gamma is the
angle between x and x' on S^2 and tau is the time lag. The measurement is
compared against the O(3) effective-theory prediction

    C_rho(gamma, tau) = sum_l  c_l^2/(2l+1) * q(tau)^{l(l+1)/2} P_l(cos gamma),

with q(tau) = <nHat(t) . nHat(t+tau)> measured from the same runs and c_l the
Legendre coefficients of the ring profile f_0 (also measured).

Input: CSVs of the form
    snapshot,time,nHat_x,nHat_y,nHat_z,particle,theta_RF,phi_RF,omega_phi
Each snapshot gives N particle positions in the ring frame plus the ring
orientation vector nHat(t) in the lab frame; we rotate back to get the lab
positions.

The simulations are already post-ring-formation (first snapshot at t = 10,
well after the transient), so every recorded snapshot is usable.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import eval_legendre, sph_harm


# ---------------------------------------------------------------------------
# Data loading and lab-frame reconstruction
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    time: float
    nhat: np.ndarray         # (3,) lab-frame ring normal
    pos_lab: np.ndarray      # (N, 3) lab-frame positions on the unit sphere
    cos_theta_rf: np.ndarray # (N,) = nhat . x_lab, the ring-frame colatitude cosine


def _rot_z_to(n: np.ndarray) -> np.ndarray:
    """Rotation matrix R with R @ (0,0,1) = n."""
    nz = float(n[2])
    if nz > 1 - 1e-12:
        return np.eye(3)
    if nz < -1 + 1e-12:
        # 180 deg rotation about x
        return np.diag([1.0, -1.0, -1.0])
    axis = np.array([-n[1], n[0], 0.0])  # (0,0,1) x n
    axis /= np.linalg.norm(axis)
    sin_a = np.sqrt(1 - nz * nz)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sin_a * K + (1 - nz) * K @ K


def load_snapshots(csv_path: str) -> list[Snapshot]:
    df = pd.read_csv(csv_path)
    snaps: list[Snapshot] = []
    for _, g in df.groupby("snapshot", sort=True):
        nhat = np.array(
            [g["nHat_x"].iloc[0], g["nHat_y"].iloc[0], g["nHat_z"].iloc[0]],
            dtype=float,
        )
        nhat = nhat / np.linalg.norm(nhat)
        theta_rf = g["theta_RF"].to_numpy(dtype=float)
        phi_rf = g["phi_RF"].to_numpy(dtype=float)
        x_rf = np.sin(theta_rf) * np.cos(phi_rf)
        y_rf = np.sin(theta_rf) * np.sin(phi_rf)
        z_rf = np.cos(theta_rf)
        pos_rf = np.stack([x_rf, y_rf, z_rf], axis=-1)
        R = _rot_z_to(nhat)
        pos_lab = pos_rf @ R.T
        snaps.append(
            Snapshot(
                time=float(g["time"].iloc[0]),
                nhat=nhat,
                pos_lab=pos_lab,
                cos_theta_rf=np.cos(theta_rf),
            )
        )
    return snaps


# ---------------------------------------------------------------------------
# Grid on S^2 (equal-area along cos theta)
# ---------------------------------------------------------------------------

def _sphere_grid(n_cos: int = 30, n_phi: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return grid points on S^2 and per-point area weights (sum to 4 pi)."""
    u_edges = np.linspace(-1.0, 1.0, n_cos + 1)
    u = 0.5 * (u_edges[:-1] + u_edges[1:])           # (n_cos,)
    sin_t = np.sqrt(np.clip(1 - u * u, 0.0, None))   # (n_cos,)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    X = np.outer(sin_t, cphi).ravel()
    Y = np.outer(sin_t, sphi).ravel()
    Z = np.repeat(u, n_phi)
    xyz = np.stack([X, Y, Z], axis=-1)               # (M, 3)
    # area per cell: (du) * (dphi), sum = 2 * 2 pi = 4 pi
    du = u_edges[1] - u_edges[0]
    dphi = 2 * np.pi / n_phi
    area = np.full(xyz.shape[0], du * dphi)
    return xyz, area


# ---------------------------------------------------------------------------
# Fisher-von Mises KDE on S^2
# ---------------------------------------------------------------------------

def _logC_vMF(kappa: float) -> float:
    # Log-normalization of 3D Fisher-von Mises:
    #   p(x) = C(kappa) exp(kappa mu . x), int p dOmega = 1
    #   C(kappa) = kappa / (4 pi sinh(kappa)) (stable form)
    if kappa < 1e-6:
        return -np.log(4 * np.pi)
    # log(kappa / (4 pi sinh kappa)) = log kappa - log(4 pi) - (kappa + log(1 - exp(-2 kappa)) - log 2)
    return (
        np.log(kappa)
        - np.log(4 * np.pi)
        - kappa
        - np.log1p(-np.exp(-2 * kappa))
        + np.log(2.0)
    )


def kde_density(grid_xyz: np.ndarray, particles_xyz: np.ndarray, kappa: float) -> np.ndarray:
    """
    Fisher-von Mises KDE evaluated on a spherical grid.

        rho(x) = (1/N) sum_n C(kappa) exp(kappa x . x_n),

    so that int rho dOmega = 1 for any particle count. Returns a vector of
    shape (M,) with one density value per grid point.
    """
    dots = grid_xyz @ particles_xyz.T  # (M, N)
    # For numerical stability, subtract the per-grid-point max before exp.
    m = dots.max(axis=1, keepdims=True)
    kern = np.exp(kappa * (dots - m))                # (M, N)
    log_norm = _logC_vMF(kappa)
    rho = kern.mean(axis=1) * np.exp(log_norm + kappa * m.squeeze(-1))
    return rho


# ---------------------------------------------------------------------------
# Empirical two-time correlator C(gamma, tau) from a run
# ---------------------------------------------------------------------------

def _ylm_matrix(grid_xyz: np.ndarray, l_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Real spherical-harmonic matrix Y[m_idx, i] evaluated at grid points.
    Returns (Y_real, l_of_mode, m_of_mode) where Y_real has shape (n_modes, M),
    n_modes = (l_max+1)^2, and Y_real is real-valued orthonormal on S^2.

    We project onto real spherical harmonics so all coefficients are real.
    """
    r = np.linalg.norm(grid_xyz, axis=1)
    theta = np.arccos(np.clip(grid_xyz[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0))
    phi = np.arctan2(grid_xyz[:, 1], grid_xyz[:, 0])
    rows = []
    l_list = []
    m_list = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                Y = np.sqrt(2) * sph_harm(abs(m), l, phi, theta).imag
            elif m == 0:
                Y = sph_harm(0, l, phi, theta).real
            else:
                Y = np.sqrt(2) * sph_harm(m, l, phi, theta).real
            rows.append(Y.real)
            l_list.append(l)
            m_list.append(m)
    Y_real = np.stack(rows, axis=0)                  # (n_modes, M)
    return Y_real, np.array(l_list), np.array(m_list)


def empirical_correlator_sh(
    rho_grid: np.ndarray,           # (T, M)
    grid_xyz: np.ndarray,           # (M, 3)
    area: np.ndarray,               # (M,)
    n_gamma_bins: int = 25,
    max_tau: int | None = None,
    l_max: int = 30,
    weight: np.ndarray | None = None,  # per-time-origin weight (T,) if pooling many traj
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast evaluation of C(gamma, tau) via spherical-harmonic decomposition.

    By rotational isotropy of the disorder-averaged ensemble the correlator
    depends only on (gamma, tau), and it admits the Legendre decomposition

        C(gamma, tau) = sum_l (2l+1)/(4 pi) * A_l(tau) * P_l(cos gamma)

    where A_l(tau) = (1 / (2l+1)) * sum_m < a_{lm}(t) a_{lm}(t+tau) >,
    with a_{lm}(t) the real-SH coefficients of rho(.,t).

    We first compute a_{lm}(t) for every snapshot, then average their
    tau-shifted products to get A_l(tau), then reconstruct C(gamma,tau) on
    the requested angular grid.
    """
    T, M = rho_grid.shape
    if max_tau is None:
        max_tau = T - 1

    # Real-SH coefficients (T, n_modes)
    Y_real, l_list, _ = _ylm_matrix(grid_xyz, l_max)       # (n_modes, M)
    A_coef = (rho_grid * area[None, :]) @ Y_real.T          # (T, n_modes)

    # For each l, collect the sum_m a_{lm}(t) a_{lm}(t+tau) averaged over t
    taus = np.arange(0, max_tau + 1)
    A_l_tau = np.zeros((len(taus), l_max + 1))
    # Index of modes belonging to each l
    idx_by_l = [np.where(l_list == l)[0] for l in range(l_max + 1)]
    for k, tau in enumerate(taus):
        if tau == 0:
            prods = (A_coef * A_coef).sum(axis=0)           # (n_modes,)
            count = T
        else:
            prods = (A_coef[:-tau] * A_coef[tau:]).sum(axis=0)
            count = T - tau
        for l in range(l_max + 1):
            A_l_tau[k, l] = prods[idx_by_l[l]].sum() / (count * (2 * l + 1))

    # Reconstruct on the gamma grid
    gamma_edges = np.linspace(0.0, np.pi, n_gamma_bins + 1)
    gamma_centers = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    cos_g = np.cos(gamma_centers)
    C = np.zeros((len(taus), n_gamma_bins))
    for l in range(l_max + 1):
        Pl = eval_legendre(l, cos_g)
        C += np.outer(A_l_tau[:, l], ((2 * l + 1) / (4 * np.pi)) * Pl)
    counts = np.array([T - tau for tau in taus], dtype=float)
    return gamma_centers, taus, C, counts


def empirical_correlator(
    rho_grid: np.ndarray,           # (T, M) density at each snapshot
    grid_xyz: np.ndarray,           # (M, 3)
    area: np.ndarray,               # (M,)
    n_gamma_bins: int = 25,
    max_tau: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute C(gamma, tau) by binning grid-point pair angles."""
    T, M = rho_grid.shape
    if max_tau is None:
        max_tau = T - 1
    # Pairwise angle between grid points
    cos_gamma = np.clip(grid_xyz @ grid_xyz.T, -1.0, 1.0)
    gamma_flat = np.arccos(cos_gamma).ravel()
    gamma_edges = np.linspace(0.0, np.pi, n_gamma_bins + 1)
    gamma_centers = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    bin_idx = np.clip(
        np.digitize(gamma_flat, gamma_edges) - 1, 0, n_gamma_bins - 1
    )
    weights_flat = np.outer(area, area).ravel()
    # Normalize bin weights so the spherical average within each bin is unbiased
    bin_weight = np.bincount(bin_idx, weights=weights_flat, minlength=n_gamma_bins)

    taus = np.arange(0, max_tau + 1)
    C = np.zeros((len(taus), n_gamma_bins))
    counts = np.zeros(len(taus))
    for k, tau in enumerate(taus):
        num = 0.0
        n_pairs = 0
        # average over time origins
        binsum = np.zeros(n_gamma_bins)
        for t in range(0, T - tau):
            outer = np.outer(rho_grid[t], rho_grid[t + tau]).ravel()
            binsum += np.bincount(
                bin_idx, weights=outer * weights_flat, minlength=n_gamma_bins
            )
            n_pairs += 1
        if n_pairs > 0:
            mean_in_bin = np.where(bin_weight > 0, binsum / np.maximum(bin_weight, 1e-15), 0.0)
            C[k] = mean_in_bin / n_pairs
            counts[k] = n_pairs
    return gamma_centers, taus, C, counts


# ---------------------------------------------------------------------------
# O(3) theoretical prediction (eq. Crho_reduced of the paper)
# ---------------------------------------------------------------------------

def legendre_coeffs(cos_theta_rf_pooled: np.ndarray, l_max: int, sigma_smooth: float = 0.05) -> np.ndarray:
    """
    Estimate the Legendre coefficients c_l of the ring profile f_0(u),
    u = cos(theta_RF), by a histogram + polynomial projection.

    f_0(u) = (1 / (2 pi)) d rho_RF / du  where rho_RF is the ring-frame
    angular density normalized to int rho_RF dOmega = 1. Pooled over particles
    and snapshots gives a smooth estimate (the ring profile is stationary in
    the ring frame).
    """
    # Histogram in u
    n_u = 400
    u_edges = np.linspace(-1.0, 1.0, n_u + 1)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    du = u_edges[1] - u_edges[0]
    hist, _ = np.histogram(cos_theta_rf_pooled, bins=u_edges)
    # f_0(u) = P(u) / (2 pi), with int_{-1}^{1} P(u) du = 1
    P = hist / (len(cos_theta_rf_pooled) * du)
    f0 = P / (2 * np.pi)
    # Optional Gaussian smoothing along u
    if sigma_smooth > 0:
        x = np.arange(-4 * sigma_smooth, 4 * sigma_smooth + du, du)
        k = np.exp(-0.5 * (x / sigma_smooth) ** 2)
        k /= k.sum()
        f0 = np.convolve(f0, k, mode="same")
    c = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        Pl = eval_legendre(l, u_centers)
        c[l] = (2 * l + 1) / 2 * np.trapz(f0 * Pl, u_centers)
    return c


def o3_correlator(
    gamma_centers: np.ndarray,
    taus_steps: np.ndarray,
    D_rot: float,
    c_l: np.ndarray,
    dt_per_step: float,
) -> np.ndarray:
    """
    Sum_{l=0}^{l_max} c_l^2/(2 l + 1) * q(tau)^{l(l+1)/2} P_l(cos gamma),
    with q(tau) = exp(-2 D_rot |tau|) and tau = dt_per_step * tau_step.
    """
    tau_phys = dt_per_step * taus_steps
    q = np.exp(-2.0 * D_rot * np.abs(tau_phys))
    C = np.zeros((len(taus_steps), len(gamma_centers)))
    cos_g = np.cos(gamma_centers)
    for l, cl in enumerate(c_l):
        Pl = eval_legendre(l, cos_g)
        amp = cl * cl / (2 * l + 1)
        # q**(l(l+1)/2)
        q_pow = q ** (l * (l + 1) / 2)
        C += amp * np.outer(q_pow, Pl)
    return C


def q_of_tau_measured(snaps: list[Snapshot], dt_per_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (tau_phys, q(tau)) measured from the nHat autocorrelation."""
    n = np.stack([s.nhat for s in snaps], axis=0)  # (T, 3)
    T = n.shape[0]
    taus = np.arange(0, T)
    q = np.zeros_like(taus, dtype=float)
    for k, tau in enumerate(taus):
        if tau == 0:
            q[k] = 1.0
        else:
            dots = np.sum(n[:-tau] * n[tau:], axis=-1)
            q[k] = dots.mean()
    return dt_per_step * taus, q


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _fit_D_rot(tau_phys: np.ndarray, q: np.ndarray) -> float:
    mask = (tau_phys > 0) & (q > 1e-3)
    if mask.sum() >= 3:
        slope, _ = np.polyfit(tau_phys[mask], np.log(q[mask]), 1)
        return -0.5 * slope
    return 0.003


def run_on_csvs(
    csv_paths: list[str],
    *,
    sigma_kde_rad: float = 0.15,
    n_cos: int = 30,
    n_phi: int = 60,
    n_gamma_bins: int = 25,
    l_max: int = 25,
    D_rot: float | None = None,
    label: str | None = None,
):
    """Pool several independent trajectories (same potential) into one estimate.

    The empirical C(gamma, tau) is obtained by summing per-trajectory bin
    accumulators (numerator and weight) and then dividing. The orientation
    autocorrelator q(tau) is the time-weighted average across trajectories.
    The ring profile c_l and the D_rot fit use pooled data.
    """
    assert csv_paths, "need at least one csv"
    label = label or "+".join(
        os.path.splitext(os.path.basename(p))[0] for p in csv_paths
    )
    print(f"\n=== {label} ===")

    # Load all trajectories
    traj_snaps = []
    for p in csv_paths:
        snaps = load_snapshots(p)
        traj_snaps.append(snaps)
        print(
            f"  {os.path.basename(p)}: {len(snaps)} snapshots, "
            f"t = {snaps[0].time:.2f}..{snaps[-1].time:.2f}"
        )

    # Assume all trajectories share the same dt
    s0 = traj_snaps[0]
    dt_per_step = (s0[-1].time - s0[0].time) / max(len(s0) - 1, 1)

    # Pool ring-frame u values for f_0 and c_l
    u_pool = np.concatenate([s.cos_theta_rf for snaps in traj_snaps for s in snaps])
    c_l = legendre_coeffs(u_pool, l_max=l_max)

    # Pooled q(tau): pair counts per tau weight runs by (T - tau) automatically
    max_T = max(len(snaps) for snaps in traj_snaps)
    taus_steps = np.arange(0, max_T)
    q_num = np.zeros(max_T)
    q_den = np.zeros(max_T)
    for snaps in traj_snaps:
        n = np.stack([s.nhat for s in snaps], axis=0)
        T = n.shape[0]
        for k in range(T):
            if k == 0:
                q_num[k] += T * 1.0
                q_den[k] += T
            else:
                d = (n[:-k] * n[k:]).sum(axis=-1)
                q_num[k] += d.sum()
                q_den[k] += d.size
    q_meas = np.where(q_den > 0, q_num / np.maximum(q_den, 1), 0.0)
    tau_phys = dt_per_step * taus_steps

    if D_rot is None:
        D_rot = _fit_D_rot(tau_phys, q_meas)
        print(f"  D_rot (pooled fit of q): {D_rot:.4f}")
    else:
        print(f"  D_rot (supplied): {D_rot:.4f}")

    # KDE grid and bandwidth
    kappa = 1.0 / sigma_kde_rad ** 2
    grid, area = _sphere_grid(n_cos=n_cos, n_phi=n_phi)
    print(f"  grid size: {grid.shape[0]}  (n_cos={n_cos}, n_phi={n_phi})", flush=True)

    # Real-SH basis evaluated on the grid (n_modes, M)
    Y_real, l_list, _ = _ylm_matrix(grid, l_max)
    idx_by_l = [np.where(l_list == l)[0] for l in range(l_max + 1)]
    print(f"  SH basis: l_max={l_max}, {Y_real.shape[0]} modes", flush=True)

    # Pool per-trajectory SH coefficients into A_l_tau accumulators
    gamma_edges = np.linspace(0.0, np.pi, n_gamma_bins + 1)
    gamma_centers = 0.5 * (gamma_edges[:-1] + gamma_edges[1:])
    cos_g = np.cos(gamma_centers)
    taus_steps = np.arange(0, max_T)

    # Per-l numerator: sum over (traj, t-origins) of sum_m a_lm(t) a_lm(t+tau)
    num_l = np.zeros((max_T, l_max + 1))
    den_count = np.zeros(max_T)   # number of (traj, t) pairs for each tau

    total_int = 0.0
    total_cnt = 0
    for ti, snaps in enumerate(traj_snaps):
        T_i = len(snaps)
        rho_t = np.stack(
            [kde_density(grid, s.pos_lab, kappa) for s in snaps], axis=0
        )  # (T_i, M)
        integ = (rho_t * area[None, :]).sum(axis=1)
        total_int += integ.sum()
        total_cnt += integ.size
        # SH coefficients: A_coef[t, mode] = sum_i area_i * rho_t[i] * Y_mode(i)
        A_coef = (rho_t * area[None, :]) @ Y_real.T   # (T_i, n_modes)
        # Accumulate tau-shifted products into num_l[tau, l]
        for k in range(T_i):
            if k == 0:
                prods = (A_coef * A_coef).sum(axis=0)
            else:
                prods = (A_coef[:-k] * A_coef[k:]).sum(axis=0)
            den_count[k] += T_i - k
            for l in range(l_max + 1):
                num_l[k, l] += prods[idx_by_l[l]].sum()
        if (ti + 1) % 20 == 0:
            print(f"  processed {ti+1}/{len(traj_snaps)} trajectories",
                  flush=True)

    print(f"  mean integral of rho over sphere (pooled): {total_int/total_cnt:.3f}", flush=True)

    # A_l(tau) = num_l / ((2l+1) * count)
    A_l_tau = np.zeros((max_T, l_max + 1))
    for l in range(l_max + 1):
        A_l_tau[:, l] = num_l[:, l] / np.maximum(den_count * (2 * l + 1), 1e-15)

    # Reconstruct C_emp(gamma, tau) = sum_l (2l+1)/(4 pi) A_l(tau) P_l(cos gamma)
    C_emp = np.zeros((max_T, n_gamma_bins))
    for l in range(l_max + 1):
        Pl = eval_legendre(l, cos_g)
        C_emp += np.outer(A_l_tau[:, l], ((2 * l + 1) / (4 * np.pi)) * Pl)

    # Theoretical O(3) C(gamma, tau) with Markovian q(tau) and measured c_l
    C_th = o3_correlator(
        gamma_centers, taus_steps, D_rot=D_rot, c_l=c_l, dt_per_step=dt_per_step
    )

    return {
        "label": label,
        "n_traj": len(traj_snaps),
        "dt": dt_per_step,
        "D_rot": D_rot,
        "c_l": c_l,
        "tau_phys": tau_phys,
        "gamma_centers": gamma_centers,
        "C_emp": C_emp,
        "C_th": C_th,
        "q_meas": q_meas,
        "C_den": den_count,
        "A_l_tau": A_l_tau,
    }


# Backwards-compatible single-file entry
def run_on_csv(csv_path: str, **kwargs):
    return run_on_csvs([csv_path], **kwargs)


def plot_results(results: list[dict], out_path: str):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n), squeeze=False)
    for row, r in enumerate(results):
        ax_static = axes[row, 0]
        ax_time = axes[row, 1]
        ax_q = axes[row, 2]

        n_traj = r.get("n_traj", 1)

        # (a) Static correlator C(gamma, 0)
        ax_static.plot(np.degrees(r["gamma_centers"]), r["C_emp"][0], "o-", label="empirical (KDE)")
        ax_static.plot(np.degrees(r["gamma_centers"]), r["C_th"][0], "--", label="O(3) prediction")
        ax_static.set_xlabel(r"$\gamma$ (deg)")
        ax_static.set_ylabel(r"$C_\rho(\gamma, 0)$")
        ax_static.set_title(r"Static density correlator $C_\rho(\gamma, 0)$")
        ax_static.legend(fontsize=8)
        ax_static.grid(alpha=0.3)

        # (b) time decay at a few gamma slices
        sel = [1, len(r["gamma_centers"]) // 3, len(r["gamma_centers"]) // 2, -2]
        for s in sel:
            g_deg = np.degrees(r["gamma_centers"][s])
            line, = ax_time.plot(
                r["tau_phys"], r["C_emp"][:, s], "o", markersize=3,
                label=f"emp γ={g_deg:.0f}°",
            )
            ax_time.plot(
                r["tau_phys"], r["C_th"][:, s], "-", color=line.get_color(), alpha=0.7,
            )
        ax_time.set_xlabel(r"$\tau$")
        ax_time.set_ylabel(r"$C_\rho(\gamma, \tau)$")
        ax_time.set_title(
            r"$\tau$-decay slices $C_\rho(\gamma_k,\tau)$"
            "\n(points = empirical, solid = O(3) prediction)"
        )
        ax_time.legend(fontsize=7, ncol=2)
        ax_time.grid(alpha=0.3)

        # (c) q(tau): measured vs Markovian
        ax_q.plot(r["tau_phys"], r["q_meas"], "o", label=r"measured $q(\tau)$")
        ax_q.plot(
            r["tau_phys"],
            np.exp(-2 * r["D_rot"] * r["tau_phys"]),
            "-", label=fr"$\exp(-2 D_{{\rm rot}}\,\tau)$, $D_{{\rm rot}}$={r['D_rot']:.4f}",
        )
        ax_q.set_xlabel(r"$\tau$")
        ax_q.set_ylabel(r"$q(\tau)$")
        ax_q.set_title(r"Orientation autocorrelation $q(\tau)$")
        ax_q.legend(fontsize=8)
        ax_q.grid(alpha=0.3)

        # Optional row label on the y-axis of the first panel if we have
        # multiple groups; keeps a distinguishing marker without cluttering
        # the main titles.
        if n > 1:
            ax_static.set_ylabel(
                ax_static.get_ylabel() + f"\n[{r['label']}]"
            )

    # Show the dataset label once as a figure-level suptitle so the panel
    # titles can stay content-focused.
    if len(results) == 1:
        r = results[0]
        suptitle = r.get("label", "")
        if suptitle:
            suptitle = f"{suptitle},  $D_{{\\rm rot}}$={r['D_rot']:.4f}"
            fig.suptitle(suptitle, fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"\nwrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    # Default groupings: pool trajectories with the same potential.
    default_groups = [
        (
            "F2 (linear, pooled)",
            [
                "data/sphere/Snapshots_ringframe_sphere_n41.csv",
                "data/sphere/Snapshots_ringframe_sphere_n41-2.csv",
            ],
        ),
        (
            "Truncated-log",
            ["data/sphere/Snapshots_ringframe_sphere_n41_trunc_log.csv"],
        ),
    ]
    p.add_argument(
        "csvs", nargs="*",
        help="(optional) explicit CSVs; if given, they are pooled as one group",
    )
    p.add_argument("--sigma", type=float, default=0.15, help="KDE bandwidth (rad)")
    p.add_argument("--n_cos", type=int, default=30)
    p.add_argument("--n_phi", type=int, default=60)
    p.add_argument("--n_gamma", type=int, default=25)
    p.add_argument("--l_max", type=int, default=25)
    p.add_argument("--D_rot", type=float, default=None)
    p.add_argument("--out", type=str, default="figures/density_correlator.png")
    args = p.parse_args()

    if args.csvs:
        # When CSVs are passed explicitly, build a concise label from the
        # common parent directory of the inputs so the suptitle is
        # informative rather than generic.
        parents = sorted({os.path.dirname(p) for p in args.csvs})
        if len(parents) == 1 and parents[0]:
            label = f"F2 ring ensemble ({os.path.basename(parents[0])}, {len(args.csvs)} trajectories)"
        else:
            label = f"F2 ring ensemble ({len(args.csvs)} trajectories)"
        groups = [(label, list(args.csvs))]
    else:
        groups = default_groups

    results = []
    for label, paths in groups:
        existing = [p for p in paths if os.path.exists(p)]
        missing = [p for p in paths if not os.path.exists(p)]
        for m in missing:
            print(f"  skip (missing): {m}", file=sys.stderr)
        if not existing:
            continue
        results.append(
            run_on_csvs(
                existing,
                sigma_kde_rad=args.sigma,
                n_cos=args.n_cos,
                n_phi=args.n_phi,
                n_gamma_bins=args.n_gamma,
                l_max=args.l_max,
                D_rot=args.D_rot,
                label=label,
            )
        )
    if not results:
        print("no data; nothing to do", file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_results(results, args.out)


if __name__ == "__main__":
    main()

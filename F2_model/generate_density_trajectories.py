"""
Generate multiple post-ring-formation trajectories suitable for the
density-density correlator analysis of density_correlator.py and for
future probability-current / NESS diagnostics.

Per disorder realization we:
  1. equilibrate for n_equil steps (ring formation + transient relaxation),
  2. verify that a ring has formed (inertia-tensor eigenvalue ratio),
  3. record lab-frame particle positions and the ring orientation n_hat(t)
     every record_every steps for n_track steps.

Two outputs:
  - one CSV per realization in --out_dir,
  - optionally, a single combined CSV in --combined_out.

Each CSV has columns:
    realization, snapshot, time, nHat_x, nHat_y, nHat_z, particle,
    x, y, z, theta_RF, phi_RF

where (x, y, z) are the lab-frame particle coordinates on the unit sphere,
(theta_RF, phi_RF) are the ring-frame spherical coordinates (theta_RF =
angle from n_hat), and nHat is sign-continuated across snapshots to give a
smooth trajectory on S^2.
"""

from __future__ import annotations

import argparse
import os
import sys
import time as clock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brownian_s2_simulation import BrownianParticlesS2, SimulationParams  # noqa: E402


def _ring_normal(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = positions.T @ positions / len(positions)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)
    return vecs[:, order[0]], vals[order]


def _ring_frame_coords(positions: np.ndarray, nhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_rf = np.clip(positions @ nhat, -1.0, 1.0)
    theta_rf = np.arccos(z_rf)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(nhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - (tmp @ nhat) * nhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nhat, e1)
    x_rf = positions @ e1
    y_rf = positions @ e2
    phi_rf = np.arctan2(y_rf, x_rf)
    phi_rf = np.where(phi_rf < 0, phi_rf + 2 * np.pi, phi_rf)
    return theta_rf, phi_rf


def run_one_trajectory(
    realization_idx: int,
    *,
    N: int = 400,
    temperature: float = 0.4,
    sigma: float = 1.0,
    dt: float = 0.0025,
    n_equil: int = 4000,
    n_track: int = 16000,
    record_every: int = 100,
    coupling_type: str = "gaussian",
    potential_type: str = "linear",
    min_ratio: float = 3.0,
) -> pd.DataFrame | None:
    """Run one realization; return a DataFrame of snapshots or None on failure."""
    params = SimulationParams(
        N=N,
        temperature=temperature,
        sigma=sigma,
        coupling_type=coupling_type,
        potential_type=potential_type,
        dt=dt,
        coupling_seed=42 + 17 * realization_idx,
        init_seed=123 + 31 * realization_idx,
    )
    sim = BrownianParticlesS2(params)

    sim.run(n_equil, record_every=n_equil + 1, verbose=False)
    nhat, eig = _ring_normal(sim.positions)
    ratio = eig[1] / max(eig[0], 1e-12)
    if ratio < min_ratio:
        return None

    n_records = n_track // record_every
    N_out = n_records * N
    realization_col = np.full(N_out, realization_idx, dtype=np.int32)
    snapshot_col = np.empty(N_out, dtype=np.int32)
    time_col = np.empty(N_out, dtype=np.float32)
    nhx = np.empty(N_out, dtype=np.float32)
    nhy = np.empty(N_out, dtype=np.float32)
    nhz = np.empty(N_out, dtype=np.float32)
    particle_col = np.empty(N_out, dtype=np.int32)
    xs = np.empty(N_out, dtype=np.float32)
    ys = np.empty(N_out, dtype=np.float32)
    zs = np.empty(N_out, dtype=np.float32)
    theta_rf_col = np.empty(N_out, dtype=np.float32)
    phi_rf_col = np.empty(N_out, dtype=np.float32)

    prev_nhat = None
    for rec in range(n_records):
        sim.run(record_every, record_every=record_every + 1, verbose=False)
        nhat, _ = _ring_normal(sim.positions)
        if prev_nhat is not None and np.dot(nhat, prev_nhat) < 0:
            nhat = -nhat
        prev_nhat = nhat
        theta_rf, phi_rf = _ring_frame_coords(sim.positions, nhat)
        t_now = sim.current_time
        offset = rec * N
        snapshot_col[offset:offset + N] = rec
        time_col[offset:offset + N] = t_now
        nhx[offset:offset + N] = nhat[0]
        nhy[offset:offset + N] = nhat[1]
        nhz[offset:offset + N] = nhat[2]
        particle_col[offset:offset + N] = np.arange(N, dtype=np.int32)
        xs[offset:offset + N] = sim.positions[:, 0]
        ys[offset:offset + N] = sim.positions[:, 1]
        zs[offset:offset + N] = sim.positions[:, 2]
        theta_rf_col[offset:offset + N] = theta_rf
        phi_rf_col[offset:offset + N] = phi_rf

    return pd.DataFrame(
        {
            "realization": realization_col,
            "snapshot": snapshot_col,
            "time": time_col,
            "nHat_x": nhx,
            "nHat_y": nhy,
            "nHat_z": nhz,
            "particle": particle_col,
            "x": xs,
            "y": ys,
            "z": zs,
            "theta_RF": theta_rf_col,
            "phi_RF": phi_rf_col,
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_realizations", type=int, default=100)
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--n_equil", type=int, default=4000)
    ap.add_argument("--n_track", type=int, default=16000)
    ap.add_argument("--record_every", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="index of first realization (resume)")
    ap.add_argument(
        "--out_dir", type=str, default="data/sphere/density_runs_100",
        help="directory where one CSV per realization is written",
    )
    ap.add_argument(
        "--combined_out", type=str, default="data/sphere/density_ensemble_100.csv",
        help="path for the final combined CSV; pass empty string to skip",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_total = clock.time()
    dur_phys = args.n_track * args.dt
    print(
        f"plan: {args.n_realizations} realizations, "
        f"dt={args.dt}, n_equil={args.n_equil} (t_eq={args.n_equil*args.dt:.2f}), "
        f"n_track={args.n_track} (duration={dur_phys:.2f} per traj), "
        f"record_every={args.record_every} "
        f"({args.n_track//args.record_every} snapshots per traj)",
        flush=True,
    )

    n_ok = 0
    n_fail = 0
    print_every = 10  # summary every N realizations
    for i in range(args.start, args.n_realizations):
        t0 = clock.time()
        df = run_one_trajectory(
            i,
            N=args.N,
            temperature=args.temperature,
            sigma=args.sigma,
            dt=args.dt,
            n_equil=args.n_equil,
            n_track=args.n_track,
            record_every=args.record_every,
        )
        dur = clock.time() - t0
        if df is None:
            n_fail += 1
        else:
            out = os.path.join(args.out_dir, f"density_traj_{i:03d}.csv")
            df.to_csv(out, index=False, float_format="%.6g")
            n_ok += 1

        done_in_batch = (i - args.start + 1)
        if done_in_batch % print_every == 0 or i == args.n_realizations - 1:
            elapsed = clock.time() - t_total
            remaining = args.n_realizations - (i + 1)
            eta_sec = elapsed / done_in_batch * remaining
            print(
                f"[{i+1}/{args.n_realizations}]  "
                f"ok={n_ok}  failed={n_fail}  "
                f"last_dur={dur:.1f}s  "
                f"elapsed={elapsed/60:.1f} min  "
                f"ETA={eta_sec/60:.1f} min",
                flush=True,
            )

    if args.combined_out:
        print("\nconcatenating into combined CSV...")
        files = sorted(
            os.path.join(args.out_dir, f)
            for f in os.listdir(args.out_dir)
            if f.startswith("density_traj_") and f.endswith(".csv")
        )
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        os.makedirs(os.path.dirname(args.combined_out) or ".", exist_ok=True)
        combined.to_csv(args.combined_out, index=False, float_format="%.6g")
        print(f"wrote {args.combined_out} ({len(combined)} rows, {len(files)} realizations)")

    print(f"\nDone: {n_ok} realizations with rings (total {(clock.time()-t_total)/60:.1f} min)")


if __name__ == "__main__":
    main()

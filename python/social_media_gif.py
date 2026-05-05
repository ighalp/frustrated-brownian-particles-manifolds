"""
Side-by-side animated GIF for social-media announcement.

Layout
------
Two F2 runs are concatenated into one looping GIF. Each run starts
from a small Gaussian blob (Big Bang IC) at a different random
location on S^2, and uses an independent disorder realization.

Sampling is logarithmic in time so the early stages are clearly
visible: rapid spread from the blob, then ring formation. Final
ring then slowly precesses.

Left panel: F2 particles on S^2, colored by per-particle potential
energy E_i = sum_j phi_ij d(x_i, x_j) using the same blue->cyan->
green->yellow->red rainbow as the HTML simulator
(simulator/brownian_with_3d_angular_momentum.html).

Right panel: signed orientation vector hat{n}(t) (inertia-tensor
smallest eigenvector with continuity convention). The initial
hat{n} is taken to be the antipode of the blob centroid; the
inertia tensor convention is then applied with continuity flipping
from that anchor onward. The director line from origin to hat{n}
and the great-circle ring perpendicular to hat{n} are colored
by the total potential energy of the configuration, normalized
across both runs, with the same rainbow map.
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

from brownian_s2_simulation import BrownianParticlesS2, SimulationParams


# =========================================================
# Configuration
# =========================================================

N = 200
TEMPERATURE = 0.4
SIGMA = 1.0
DT = 0.005
T_MIN = 0.02            # first recorded time (log axis)
T_MAX = 35.0            # last recorded time per run
N_FRAMES_PER_RUN = 110  # log-spaced frame times
FPS = 14
INIT_GAUSS_VARIANCE = 0.05

RUN_SEEDS = [
    # (coupling_seed, init_seed)
    (42, 7),
    (101, 23),
]

# =========================================================
# HTML rainbow color map: blue -> cyan -> green -> yellow -> red
# =========================================================

def energy_to_rgb(value, vmin, vmax):
    """Match energyToColor() in the HTML simulator. value is a scalar
    or 1D array of length M. Returns (M,3) array in [0,1]."""
    arr = np.atleast_1d(value).astype(float)
    if vmax > vmin:
        t = (arr - vmin) / (vmax - vmin)
    else:
        t = np.full_like(arr, 0.5)
    t = np.clip(t, 0.0, 1.0)
    r = np.zeros_like(t); g = np.zeros_like(t); b = np.zeros_like(t)

    m1 = t < 0.25
    m2 = (t >= 0.25) & (t < 0.5)
    m3 = (t >= 0.5) & (t < 0.75)
    m4 = t >= 0.75

    r[m1] = 30
    g[m1] = 80 + 175 * t[m1] / 0.25
    b[m1] = 255

    r[m2] = 30 + 70 * (t[m2] - 0.25) / 0.25
    g[m2] = 255
    b[m2] = 255 - 155 * (t[m2] - 0.25) / 0.25

    r[m3] = 100 + 155 * (t[m3] - 0.5) / 0.25
    g[m3] = 255
    b[m3] = 100 - 100 * (t[m3] - 0.5) / 0.25

    r[m4] = 255
    g[m4] = 255 - 200 * (t[m4] - 0.75) / 0.25
    b[m4] = 0
    return np.stack([r, g, b], axis=-1) / 255.0


# =========================================================
# Helpers
# =========================================================

def per_particle_energy(positions, phi, R=1.0):
    """E_i = sum_{j!=i} phi_{ij} d(x_i, x_j) on S^2 with d the
    geodesic distance."""
    dots = positions @ positions.T
    cos_a = np.clip(dots / (R * R), -1.0, 1.0)
    d = R * np.arccos(cos_a)
    np.fill_diagonal(d, 0.0)
    return (phi * d).sum(axis=1)


def total_potential_energy(positions, phi, R=1.0):
    dots = positions @ positions.T
    cos_a = np.clip(dots / (R * R), -1.0, 1.0)
    d = R * np.arccos(cos_a)
    return float(np.sum(np.triu(phi * d, k=1)))


def signed_n_inertia(positions, prev=None):
    cov = positions.T @ positions / len(positions)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, np.argmin(eigvals)]
    if prev is not None and float(n @ prev) < 0:
        n = -n
    return n


def make_sphere_mesh(R=1.0, res=24):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def great_circle_perp(n_hat, R=1.0, n_pts=120):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(n_hat @ ref) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ n_hat) * n_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_hat, e1)
    phi = np.linspace(0, 2 * np.pi, n_pts)
    return R * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)


# =========================================================
# Run one simulation, recording frames at log-spaced times
# =========================================================

def run_one(coupling_seed, init_seed):
    params = SimulationParams(
        N=N, R=1.0,
        temperature=TEMPERATURE, gamma=1.0, dt=DT,
        coupling_type="gaussian",
        potential_type="linear",
        sigma=SIGMA,
        coupling_seed=coupling_seed,
        init_seed=init_seed,
        init_type="gaussian",
        init_gauss_variance=INIT_GAUSS_VARIANCE,
    )
    sim = BrownianParticlesS2(params)

    # blob centroid (mean direction at t=0); antipode = initial n_hat
    blob_dir = sim.positions.mean(axis=0)
    blob_dir /= np.linalg.norm(blob_dir)
    n_anchor = -blob_dir  # antipodal point as the starting orientation

    # log-spaced target times
    target_times = np.geomspace(T_MIN, T_MAX, N_FRAMES_PER_RUN)
    n_steps_total = int(np.ceil(T_MAX / DT)) + 5

    # frame buffers
    pos_frames, energy_frames, n_frames, totE_frames, time_frames = \
        [], [], [], [], []

    # Record the first frame (t = 0+) using initial positions to anchor n_hat
    e_i = per_particle_energy(sim.positions, sim.phi, R=1.0)
    pos_frames.append(sim.positions.copy())
    energy_frames.append(e_i)
    # First inertia-tensor n_hat: align with anchor (antipode of blob)
    n0 = signed_n_inertia(sim.positions, prev=n_anchor)
    n_frames.append(n0)
    totE_frames.append(total_potential_energy(sim.positions, sim.phi))
    time_frames.append(0.0)
    prev_n = n0

    next_idx = 0  # index into target_times
    for s in range(n_steps_total):
        sim.step()
        if next_idx < len(target_times) and \
                sim.current_time >= target_times[next_idx]:
            ei = per_particle_energy(sim.positions, sim.phi, R=1.0)
            n = signed_n_inertia(sim.positions, prev=prev_n)
            totE = total_potential_energy(sim.positions, sim.phi)
            pos_frames.append(sim.positions.copy())
            energy_frames.append(ei)
            n_frames.append(n)
            totE_frames.append(totE)
            time_frames.append(sim.current_time)
            prev_n = n
            next_idx += 1
        if next_idx >= len(target_times):
            break

    return {
        "positions": pos_frames,
        "energies": energy_frames,
        "n_hats": np.array(n_frames),
        "tot_energies": np.array(totE_frames),
        "times": np.array(time_frames),
        "blob_dir": blob_dir,
    }


# =========================================================
# Build the side-by-side animation over both runs
# =========================================================

def make_animation(runs, raw_path):
    # Concatenate runs end-to-end. Track which run each frame belongs to.
    all_pos, all_E, all_n, all_totE, all_t, all_run = [], [], [], [], [], []
    for ri, r in enumerate(runs):
        for k in range(len(r["times"])):
            all_pos.append(r["positions"][k])
            all_E.append(r["energies"][k])
            all_n.append(r["n_hats"][k])
            all_totE.append(r["tot_energies"][k])
            all_t.append(r["times"][k])
            all_run.append(ri + 1)
    all_n = np.array(all_n)
    all_totE = np.array(all_totE)
    n_frames = len(all_t)

    # Global per-particle energy normalization (use 5/95 percentiles
    # of the union, so colors are stable across the whole loop).
    E_concat = np.concatenate(all_E)
    e_lo, e_hi = np.percentile(E_concat, [5, 95])
    if e_hi <= e_lo:
        e_hi = e_lo + 1.0
    # Total-energy normalization for ring color
    totE_lo, totE_hi = np.percentile(all_totE, [5, 95])
    if totE_hi <= totE_lo:
        totE_hi = totE_lo + 1.0

    # ------------- figure ----------------
    fig = plt.figure(figsize=(10.5, 5.9), dpi=100,
                     facecolor="#0f172a")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.96,
                        bottom=0.10, wspace=0.0)

    ax_L = fig.add_subplot(1, 2, 1, projection="3d",
                           facecolor="#0f172a")
    ax_R = fig.add_subplot(1, 2, 2, projection="3d",
                           facecolor="#0f172a")

    # Tighter sphere fit
    LIM = 1.02
    for ax in (ax_L, ax_R):
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_zlim(-LIM, LIM)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        # squeeze away matplotlib's 3D padding
        try:
            ax.set_proj_type("persp", focal_length=0.6)
        except Exception:
            pass

    # Sphere meshes (drawn once, low alpha)
    xs, ys, zs = make_sphere_mesh(R=1.0, res=22)
    ax_L.plot_surface(xs, ys, zs, alpha=0.10, color="#3b82f6",
                      linewidth=0, antialiased=True)
    ax_R.plot_surface(xs, ys, zs, alpha=0.08, color="#3b82f6",
                      linewidth=0, antialiased=True)

    # Coordinate axes on right panel
    for vec in [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]:
        ax_R.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],
                  color="0.45", linewidth=0.6, alpha=0.6)

    # ---- artists ----
    pos0 = all_pos[0]
    col0 = energy_to_rgb(all_E[0], e_lo, e_hi)
    scatter_L = ax_L.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2],
                             c=col0, s=22, alpha=0.95,
                             edgecolors="none", depthshade=True)

    n0 = all_n[0]
    ring_color0 = energy_to_rgb(all_totE[0], totE_lo, totE_hi)[0]

    dot_R = ax_R.scatter([n0[0]], [n0[1]], [n0[2]],
                         c=[ring_color0], s=110,
                         edgecolors="white", linewidths=0.8,
                         depthshade=False, zorder=10)
    director_line, = ax_R.plot([0, n0[0]], [0, n0[1]], [0, n0[2]],
                               color=ring_color0, linewidth=2.5)
    ring_pts = great_circle_perp(n0)
    ring_line, = ax_R.plot(ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2],
                           color=ring_color0, linewidth=2.0, alpha=0.95)
    trail_line, = ax_R.plot([], [], [], color="0.55",
                            linewidth=1.0, alpha=0.55)

    # Per-panel labels at the bottom of each panel
    fig.text(0.25, 0.955,
             r"F2 particles on $S^2$ (Big Bang IC)",
             ha="center", va="top",
             color="#e2e8f0", fontsize=12)
    fig.text(0.75, 0.955,
             r"Signed orientation $\hat{\mathbf{n}}(t)$ on $S^2$",
             ha="center", va="top",
             color="#e2e8f0", fontsize=12)

    fig.text(0.5, 0.05,
             f"$N={N}$, $T={TEMPERATURE}$, "
             f"$\\sigma={SIGMA}$, $V(d)=d$  |  "
             f"log-time sampling  |  color: per-particle "
             f"energy (left), total energy (right ring)",
             ha="center", va="bottom",
             color="#94a3b8", fontsize=9)
    info_text = fig.text(
        0.5, 0.015, "", ha="center", va="bottom",
        color="#cbd5e1", fontsize=11, family="monospace",
    )

    # ---- update ----
    def update(frame):
        pos = all_pos[frame]
        scatter_L._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        col = energy_to_rgb(all_E[frame], e_lo, e_hi)
        scatter_L.set_color(col)

        n = all_n[frame]
        ring_color = energy_to_rgb(all_totE[frame], totE_lo, totE_hi)[0]

        dot_R._offsets3d = ([n[0]], [n[1]], [n[2]])
        dot_R.set_color([ring_color])

        director_line.set_data([0, n[0]], [0, n[1]])
        director_line.set_3d_properties([0, n[2]])
        director_line.set_color(ring_color)

        ring = great_circle_perp(n)
        ring_line.set_data(ring[:, 0], ring[:, 1])
        ring_line.set_3d_properties(ring[:, 2])
        ring_line.set_color(ring_color)

        # trail of recent n_hat (only within the current run)
        run_id = all_run[frame]
        # find first index of this run
        run_start = next(i for i, r in enumerate(all_run) if r == run_id)
        start = max(run_start, frame - 60)
        trail = all_n[start: frame + 1]
        trail_line.set_data(trail[:, 0], trail[:, 1])
        trail_line.set_3d_properties(trail[:, 2])

        # gentle camera rotation
        az = 25 + frame * 0.35
        ax_L.view_init(elev=18, azim=az)
        ax_R.view_init(elev=18, azim=az)

        info_text.set_text(
            f"run {run_id}/{len(runs)}    "
            f"t = {all_t[frame]:6.2f}"
        )
        return (scatter_L, dot_R, director_line, ring_line,
                trail_line, info_text)

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / FPS, blit=False)

    print(f"Writing raw GIF: {raw_path} ({n_frames} frames at {FPS} fps)")
    writer = PillowWriter(fps=FPS)
    anim.save(raw_path, writer=writer, dpi=92,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)


# =========================================================
# ffmpeg compression
# =========================================================

def compress_gif(raw_path, final_path, height=340, fps=FPS, max_colors=128):
    cmd = [
        "ffmpeg", "-y", "-i", raw_path,
        "-filter_complex",
        f"[0:v]scale=-1:{height}:flags=lanczos,fps={fps}[v];"
        f"[v]split[v0][v1];"
        f"[v0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        f"[v1][p]paletteuse=dither=bayer:bayer_scale=4",
        final_path,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(final_path) / 1e6
    print(f"Final GIF: {final_path}  ({sz:.2f} MB)")
    return sz


def main():
    out_dir = os.path.join(PROJECT_DIR, "gifs")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "f2_signed_orientation_raw.gif")
    final_path = os.path.join(out_dir, "f2_signed_orientation.gif")

    runs = []
    for i, (cs, isd) in enumerate(RUN_SEEDS):
        print(f"Run {i+1}/{len(RUN_SEEDS)}: coupling_seed={cs}, "
              f"init_seed={isd}")
        runs.append(run_one(cs, isd))
        print(f"  collected {len(runs[-1]['times'])} frames "
              f"(t in [{runs[-1]['times'][0]:.3f}, "
              f"{runs[-1]['times'][-1]:.2f}])")

    print("Building animation ...")
    make_animation(runs, raw_path)

    print("Compressing ...")
    try:
        sz = compress_gif(raw_path, final_path,
                          height=340, max_colors=128)
        if sz > 9.5:
            print("Large; reducing colors/height ...")
            sz = compress_gif(raw_path, final_path,
                              height=300, max_colors=96)
        if sz > 9.5:
            print("Still large; reducing further ...")
            compress_gif(raw_path, final_path,
                         height=260, max_colors=64)
    except Exception as e:
        print("ffmpeg failed:", e)
        return

    if os.path.exists(final_path) \
            and os.path.getsize(final_path) <= 10 * 1024 * 1024:
        try:
            os.remove(raw_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()

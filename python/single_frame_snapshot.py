"""
Single high-resolution PNG snapshot of the F2 side-by-side
visualization (left: particles on S^2, right: signed orientation
hat{n} with director and perpendicular ring).

The state is taken at a target time at which the ring is already
well-formed, using the same seeds as Run 1 of the social-media GIF.
The output PNG is suitable as a static illustration in the
manuscript; the full animated version is in
gifs/f2_signed_orientation.gif.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

from brownian_s2_simulation import BrownianParticlesS2, SimulationParams
from social_media_gif import (
    energy_to_rgb, per_particle_energy, total_potential_energy,
    signed_n_inertia, make_sphere_mesh, great_circle_perp,
    N, TEMPERATURE, SIGMA, DT, INIT_GAUSS_VARIANCE,
)


T_TARGET = 60.0          # well-formed ring, mid-precession
COUPLING_SEED = 42
INIT_SEED = 7
OUT_PATH = os.path.join(PROJECT_DIR, "ring_snapshot.png")


def run_to_time(coupling_seed, init_seed, t_target):
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

    # Antipode of blob centroid as the n_hat continuity anchor
    blob_dir = sim.positions.mean(axis=0)
    blob_dir /= np.linalg.norm(blob_dir)
    n_anchor = -blob_dir
    prev_n = signed_n_inertia(sim.positions, prev=n_anchor)

    n_steps = int(np.ceil(t_target / DT))
    # Track n_hat with a coarse stride for continuity-flip protection
    stride = 25
    for s in range(n_steps):
        sim.step()
        if s % stride == 0:
            prev_n = signed_n_inertia(sim.positions, prev=prev_n)

    n_hat = signed_n_inertia(sim.positions, prev=prev_n)
    e_i = per_particle_energy(sim.positions, sim.phi)
    totE = total_potential_energy(sim.positions, sim.phi)
    return sim.positions, e_i, n_hat, totE, sim.current_time


def main():
    print(f"Running to t = {T_TARGET} ...")
    pos, e_i, n_hat, totE, t = run_to_time(COUPLING_SEED, INIT_SEED, T_TARGET)
    print(f"  reached t = {t:.3f}")

    e_lo, e_hi = np.percentile(e_i, [5, 95])
    if e_hi <= e_lo:
        e_hi = e_lo + 1.0
    colors = energy_to_rgb(e_i, e_lo, e_hi)
    ring_color = energy_to_rgb(totE, totE - 1.0, totE + 1.0)[0]
    # use a fixed cool color (post-relaxation total energy is low)
    ring_color = np.array([30, 200, 255]) / 255.0

    fig = plt.figure(figsize=(11.5, 6.0), dpi=160,
                     facecolor="#0f172a")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93,
                        bottom=0.07, wspace=0.0)

    ax_L = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#0f172a")
    ax_R = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#0f172a")
    LIM = 1.02
    for ax in (ax_L, ax_R):
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_zlim(-LIM, LIM)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        try:
            ax.set_proj_type("persp", focal_length=0.6)
        except Exception:
            pass

    xs, ys, zs = make_sphere_mesh(R=1.0, res=28)
    ax_L.plot_surface(xs, ys, zs, alpha=0.10, color="#3b82f6",
                      linewidth=0, antialiased=True)
    ax_R.plot_surface(xs, ys, zs, alpha=0.08, color="#3b82f6",
                      linewidth=0, antialiased=True)

    for vec in [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]:
        ax_R.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],
                  color="0.45", linewidth=0.6, alpha=0.6)

    ax_L.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                 c=colors, s=28, alpha=0.95,
                 edgecolors="none", depthshade=True)

    ax_R.scatter([n_hat[0]], [n_hat[1]], [n_hat[2]],
                 c=[ring_color], s=130,
                 edgecolors="white", linewidths=0.9,
                 depthshade=False, zorder=10)
    ax_R.plot([0, n_hat[0]], [0, n_hat[1]], [0, n_hat[2]],
              color=ring_color, linewidth=2.6)
    ring = great_circle_perp(n_hat)
    ax_R.plot(ring[:, 0], ring[:, 1], ring[:, 2],
              color=ring_color, linewidth=2.2, alpha=0.95)

    az = 35
    el = 18
    ax_L.view_init(elev=el, azim=az)
    ax_R.view_init(elev=el, azim=az)

    fig.text(0.25, 0.945,
             r"F2 particles on $S^2$",
             ha="center", va="top", color="#e2e8f0", fontsize=13)
    fig.text(0.75, 0.945,
             r"Signed orientation $\hat{\mathbf{n}}$ on $S^2$",
             ha="center", va="top", color="#e2e8f0", fontsize=13)
    fig.text(0.5, 0.04,
             f"$N={N}$, $T={TEMPERATURE}$, $\\sigma={SIGMA}$, "
             f"$V(d)=d$, $t={t:.1f}$ "
             f"(well-formed ring; per-particle energy on left, "
             "perpendicular great circle on right)",
             ha="center", va="bottom",
             color="#94a3b8", fontsize=10)

    fig.savefig(OUT_PATH, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

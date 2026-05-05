# F2 model: scripts for paper figures

This folder contains the Python scripts that produce the seven figures
of the paper *Frustrated Fields: Statistical Field Theory for
Frustrated Brownian Particles on 2D Manifolds* (arXiv:2601.18653),
together with the simulation modules they depend on. Each figure is
described below with the exact recipe to recreate its underlying data.

The scripts assume they are kept inside this folder of the repository,
so that the relative paths `data/sphere/...` resolve to the
project-root data directory and figures are written to the project
root. All commands below are written from the repository root.

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `pandas`, `matplotlib`
- The HTML simulator at `simulator/brownian_with_3d_angular_momentum.html`
  for the long single-trajectory CSV exports
  (`Coords_*`, `Snapshots_ringframe_*`, `L_series_3D_*`)

## File map

```
F2_model/
  brownian_s2_simulation.py        core S^2 simulation class (used by the
                                   Python-only generators below)
  ensemble_N400.py                 -> ensemble_N400.png
  n_scaling_study.py               -> n_scaling_Drot.png
  analyze_ring_frame.py            -> ring_frame_analysis.png
  o3_model_comparison.py           -> o3_model_comparison.png
  extract_memory_kernel.py         -> memory_kernel.png
  density_correlator.py            -> density_correlator_100.png
  analyze_coulomb.py               -> coulomb_comparison.png
  generate_density_trajectories.py generates the 100 ring-trajectory CSVs
                                   read by density_correlator.py
```

## Data sources

The figure scripts split into two groups by how their data is produced.

**Self-contained (run an internal simulation).** The scripts
`ensemble_N400.py`, `n_scaling_study.py`, and
`generate_density_trajectories.py` each instantiate
`BrownianParticlesS2` and run their own simulations at the paper
parameters ($T = 0.4$, $\sigma = 1$, Gaussian quenched couplings,
linear potential $V(d) = d$). Nothing else is needed beyond a Python
environment.

**CSV-driven.** The remaining scripts read CSVs that were exported
from the WebGL simulator at
`simulator/brownian_with_3d_angular_momentum.html`. That simulator
reproduces the exact long single-trajectory run used in the paper at
$N = 400$, $T = 0.4$, $\sigma = 1$, $\mathrm{d}t = 0.005$. After
running the simulator in a browser, the buttons "Coords",
"L\_series", and "Snapshots" produce the three CSV families used by
the figures:

| Export type | File name pattern | Content |
|---|---|---|
| Coords | `Coords_sphere_t<time>_<seed>.csv` | full $(x,y,z)$ history of all $N$ particles, one row per timestep |
| L\_series | `L_series_3D_sphere.csv` | total angular-momentum vector $L(t)$ at every recording step |
| Snapshots | `Snapshots_ringframe_sphere_n<n>.csv` | $n$ co-moving ring-frame snapshots evenly spaced along the trajectory |

The Coulomb-gas variants
(`*_Coulomb_gas.csv`) are obtained by running the same simulator with
the potential dropdown set to "Coulomb gas".

Place the exported CSVs in `data/sphere/` under the project root. Once
they are in place, the figure scripts find them automatically.

## How to reproduce each figure

All commands below are run from the repository root after activating
the Python environment. Output PNGs are written to the repository
root.

### Fig. ensemble\_N400.png (rotational diffusion ensemble)

Runs $10$ disorder realizations at $N = 400$, $T = 0.4$, $\sigma = 1$,
extracts the ring-frame $D_{\text{rot}}$ from each.

```
python F2_model/ensemble_N400.py
```

Self-contained. Runtime is a few minutes on a laptop.

### Fig. n\_scaling\_Drot.png ($N$-scaling of $D_{\text{rot}}$)

Sweeps $N \in \{50, 100, 200, 300, 400, 500\}$ with three disorder
realizations per $N$.

```
python F2_model/n_scaling_study.py
```

Self-contained. Runtime grows roughly as $N^2$, mostly from the
$N = 500$ runs.

### Fig. ring\_frame\_analysis.png (ring-profile diagnostics)

Reads the snapshot file
`data/sphere/Snapshots_ringframe_sphere_n41.csv` (41 snapshots from
the long single-trajectory run) and fits the ring profile.

```
python F2_model/analyze_ring_frame.py
```

To recreate the input CSV, open
`simulator/brownian_with_3d_angular_momentum.html` in a browser, set
$N = 400$, $T = 0.4$, $\sigma = 1$, $\mathrm{d}t = 0.005$, run for at
least $10000$ steps so $41$ snapshots accumulate, then click
"Snapshots" and save the export under
`data/sphere/Snapshots_ringframe_sphere_n41.csv`.

### Fig. o3\_model\_comparison.png (effective SO(3) NLSM comparison)

Compares the long-time $L(t)$ from the F2 simulation to a free
Brownian motion on $S^2$ with the matching $D_{\text{rot}}$. Reads
`data/sphere/L_series_3D_sphere.csv` and
`data/sphere/Coords_sphere_t50_17.csv` from the simulator export.

```
python F2_model/o3_model_comparison.py
```

To recreate the inputs, run the WebGL simulator with the same
parameters as above through to $t = 50$ ($10000$ steps), click
"L\_series" and save as
`data/sphere/L_series_3D_sphere.csv`, then click "Coords" and save
the latest snapshot as `data/sphere/Coords_sphere_t50_17.csv` (the
suffix `17` is the random seed used for the run reported in the
paper; any equivalent run with the same $T$, $\sigma$, $N$, $\mathrm{d}t$
will work).

### Fig. memory\_kernel.png (memory kernel from $L(t)$)

Reads `data/sphere/L_series_3D_sphere.csv` and extracts the
non-Markovian memory kernel via the Volterra-equation method.

```
python F2_model/extract_memory_kernel.py
```

Uses the same `L_series_3D_sphere.csv` produced for
`o3_model_comparison.png`, no additional data needed.

### Fig. density\_correlator\_100.png (disorder-averaged density correlator)

This is a two-step pipeline. First generate $100$ ring-trajectory
realizations:

```
python F2_model/generate_density_trajectories.py \
    --n_realizations 100 \
    --N 400 --temperature 0.4 --sigma 1.0 --dt 0.0025 \
    --n_equil 4000 --n_track 16000 --record_every 100 \
    --out_dir data/sphere/density_runs_100
```

Then compute and plot the disorder-averaged correlator:

```
python F2_model/density_correlator.py \
    data/sphere/density_runs_100/density_traj_*.csv \
    --out density_correlator_100.png
```

The first step is the costly one (a few hours on a laptop with $100$
realizations of $400$ particles). It can be parallelized over
realizations by splitting the index range with the `--start` flag.

### Fig. coulomb\_comparison.png (potential-shape comparison)

Compares the F2 ring observables with two reference systems sharing
the same particle count and noise: a Coulomb gas and a truncated-log
potential. Reads six CSV files: the F2 snapshot and $L$-series
exports listed above, plus
`data/sphere/Snapshots_ringframe_sphere_n41_Coulomb_gas.csv`,
`data/sphere/L_series_3D_sphere_Coulomb_gas.csv`, and
`data/sphere/Coords_sphere_t50_00-5_Coulomb_gas.csv`.

```
python F2_model/analyze_coulomb.py
```

To recreate the Coulomb-gas inputs, set the potential dropdown in the
WebGL simulator to "Coulomb gas", keep $N = 400$, $T = 0.4$,
$\sigma = 1$, $\mathrm{d}t = 0.005$, run a long trajectory matching
the F2 run, and export "Coords", "L\_series", and "Snapshots" to the
filenames above under `data/sphere/`.

## Reproducibility notes

The Python-only scripts seed the disorder and the initial condition
explicitly inside `BrownianParticlesS2`, so reruns at the same
parameters give bit-identical CSVs. The WebGL exports do not seed the
RNG, so reruns produce statistically equivalent but not bit-identical
trajectories. Any subsequent CSV-driven figure is statistically
robust to this; only fine details such as the precise position of a
single trajectory in the $L(t)$ plot will shift.

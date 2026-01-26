# Order Out of Noise and Disorder: Fate of the Frustrated Manifold

This repository contains simulation code, visualizations, and data for the paper "Order Out of Noise and Disorder: Fate of the Frustrated Manifold" by Igor Halperin.

The work studies frustrated Brownian particles on curved 2D manifolds (sphere, cylinder, and torus).

## Overview

This work studies **frustrated Brownian particles** on 2D Riemannian manifolds (sphere, cylinder, torus) with **spin-glass-like disordered interactions**. Each pair of particles has a random coupling (either attractive or repulsive), creating frustration analogous to conflicting spin alignments in disordered magnets.

### Main Findings

The central phenomenon is **dynamic dimension reduction** accompanied by **spontaneous breaking of rotational symmetry** and emergence of **non-equilibrium steady states (NESS)**. From three minimal ingredients—Brownian noise, quenched random couplings, and manifold geometry—ordered structures emerge spontaneously:

- **Sphere**: Particles form a great-circle band (SO(3) → SO(2) symmetry breaking)
- **Torus**: Particles organize into two minor-circle rings (SO(2)×SO(2) → SO(2)×Z₂)
- **Cylinder**: Particles localize into discrete clusters near boundaries (SO(2) → Z₂)

The type of symmetry breaking is determined by the **geometry and topology** of the manifold: closed manifolds produce extended structures wrapping closed geodesics, while bounded manifolds produce localized clusters.

### Applications

**Soft Matter and Biophysics**: The model serves as a benchmark for investigating collective behavior of particles on curved surfaces. Experimental realizations include colloidal particles confined to curved liquid interfaces, molecules adsorbed on nanotubes or vesicles, proteins diffusing on cell membranes, and active matter on curved substrates. The phenomena documented—frustration-induced dimensional reduction, slow collective modes, geometry-dependent anomalous diffusion—should be directly accessible to experiment.

**Theoretical Physics**: The model provides a toy system for studying non-perturbative phenomena in condensed matter physics and quantum field theory (QFT). Connections include:
- Instanton-like transitions between configurations
- Analogies to quark confinement in QCD (linear potential, dimensional confinement)
- Magnetic monopole configurations on S²
- Dissipative Nambu-Goldstone modes in open systems
- Classical stochastic realizations of symmetry breaking patterns

## Repository Structure

```
├── paper/                          # Research paper
│   └── Frustrated_Brownian_particles_on_manifolds.pdf
├── simulator/                      # Interactive simulator
│   └── brownian_unified_tracking_full_history.html
├── videos/                         # Recorded simulation playback (paper settings)
│   ├── brownian_sphere_player.html
│   ├── brownian_cylinder_player.html
│   └── brownian_torus_player.html
├── gifs/                           # Animated GIFs of simulations
│   ├── Dimension_reduction_on_the_sphere.gif
│   ├── Dimension_reduction_on_the_cylinder.gif
│   ├── Dimension_reduction_on_a_torus.gif
│   └── combined_dimension_reduction.gif
├── data/                           # Simulation output data (paper settings)
│   ├── sphere/                     # Sphere geometry data
│   ├── cylinder/                   # Cylinder geometry data
│   └── torus/                      # Torus geometry data
└── python/                         # Python simulation code
    ├── run_simulation.py           # Main simulation runner
    ├── brownian_s2_simulation.py   # Sphere-specific simulation
    ├── brownian_ui.py              # UI utilities
    └── html_to_gif.py              # Convert HTML videos to GIF
```

## Interactive Simulator

Open `simulator/brownian_unified_tracking_full_history.html` in a web browser to run interactive simulations. The simulator supports:

- **Multiple geometries**: Sphere (S²), Cylinder, and Torus
- **Adjustable parameters**:
  - Number of particles
  - Temperature
  - Coupling strength and type (Gaussian or discrete ±1)
  - Diffusion coefficient
- **Real-time visualization** with 3D rendering
- **Recording capability** to save simulations as playable HTML files
- **Data export** to CSV format

## Recorded Simulations

The `videos/` directory contains pre-recorded simulations **for the parameter settings used in the paper**. These can be played back by opening the HTML files in a browser:

- `brownian_sphere_player.html` - Simulation on the 2-sphere
- `brownian_cylinder_player.html` - Simulation on a cylinder
- `brownian_torus_player.html` - Simulation on a torus

Users can create their own videos with different parameter settings by running the interactive simulator and using its recording capability.

## Data Format

The `data/` directory contains simulation output data **for the parameter settings used in the paper**. The CSV files include:

- **Coords_*.csv**: Particle coordinates over time
- **Clusters_*.csv**: Cluster assignment data
- **L_series_*.csv**: Angular momentum time series
- **Rings_*.csv**: Ring structure data (torus only)

Users can generate their own datasets by running the simulator and using its data export functionality.

## Python Code

The `python/` directory contains standalone Python implementations:

- `run_simulation.py` - Run simulations and generate summary figures
- `brownian_s2_simulation.py` - Detailed sphere simulation with animation
- `html_to_gif.py` - Convert recorded HTML simulations to GIF format

### Dependencies

```bash
pip install numpy matplotlib
```

## The Model

Particles move on a curved manifold following overdamped Langevin dynamics:

$$d\mathbf{x}_i = \frac{1}{\gamma}\sum_j \phi_{ij} \nabla_i d(\mathbf{x}_i, \mathbf{x}_j) \, dt + \sqrt{2D} \cdot d\mathbf{W}_i$$

where:
- $d(\mathbf{x}_i, \mathbf{x}_j)$ is the geodesic distance between particles
- $\phi_{ij}$ are random couplings (positive = attractive, negative = repulsive)
- $D = T/\gamma$ is the diffusion coefficient
- $d\mathbf{W}_i$ is Brownian noise projected onto the tangent space

## Citation

If you use this code in your research, please cite:

```
@article{halperin2026order,
  title={Order Out of Noise and Disorder: Fate of the Frustrated Manifold},
  author={Halperin, Igor},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License

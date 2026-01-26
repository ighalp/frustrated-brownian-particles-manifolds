# Order Out of Noise and Disorder: Fate of the Frustrated Manifold

This repository contains simulation code, visualizations, and data for the paper "Order Out of Noise and Disorder: Fate of the Frustrated Manifold" by Igor Halperin.

The work studies frustrated Brownian particles on curved 2D manifolds (sphere, cylinder, and torus).

## Overview

The model explores the dynamics of interacting Brownian particles constrained to move on curved surfaces. Each pair of particles has a random coupling (either attractive or repulsive), leading to frustrated dynamics and interesting emergent collective behavior.

## Repository Structure

```
├── paper/                          # Research paper
│   └── Frustrated_Brownian_particles_on_manifolds.pdf
├── simulator/                      # Interactive simulator
│   └── brownian_unified_improved.html
├── videos/                         # Recorded simulation playback
│   ├── brownian_sphere_player.html
│   ├── brownian_cylinder_player.html
│   └── brownian_torus_player.html
├── data/                           # Simulation output data
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

Open `simulator/brownian_unified_improved.html` in a web browser to run interactive simulations. The simulator supports:

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

The `videos/` directory contains pre-recorded simulations that can be played back by opening the HTML files in a browser:

- `brownian_sphere_player.html` - Simulation on the 2-sphere
- `brownian_cylinder_player.html` - Simulation on a cylinder
- `brownian_torus_player.html` - Simulation on a torus

## Data Format

The CSV files in `data/` contain:

- **Coords_*.csv**: Particle coordinates over time
- **Clusters_*.csv**: Cluster assignment data
- **L_series_*.csv**: Angular momentum time series
- **Rings_*.csv**: Ring structure data (torus only)

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

$$d\mathbf{x}_i = \frac{1}{\gamma}\sum_j \phi_{ij} \nabla_i d(\mathbf{x}_i, \mathbf{x}_j) \, dt + \sqrt{2D} \, d\mathbf{W}_i$$

where:
- $d(\mathbf{x}_i, \mathbf{x}_j)$ is the geodesic distance between particles
- $\phi_{ij}$ are random couplings (positive = attractive, negative = repulsive)
- $D = T/\gamma$ is the diffusion coefficient
- $d\mathbf{W}_i$ is Brownian noise projected onto the tangent space

## Citation

If you use this code in your research, please cite:

```
@article{halperin2025order,
  title={Order Out of Noise and Disorder: Fate of the Frustrated Manifold},
  author={Halperin, Igor},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License

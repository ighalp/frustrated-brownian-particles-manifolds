"""
Brownian Particles on S² (2-Sphere) with Pairwise Interactions
==============================================================

Simulation of interacting Brownian particles diffusing on the surface of a 
2-sphere (S²) embedded in ℝ³ at temperature T, with animation output.

Mathematical Background:
------------------------
- S² = {x ∈ ℝ³ : |x| = R}
- Geodesic distance: d(x_i, x_j) = R * arccos((x_i · x_j) / R²)
- Pairwise potential: U(x_i, x_j) = φ_{ij} · d(x_i, x_j)
- Coupling constants φ_{ij}:
  1. Gaussian: φ_{ij} ~ N(0, σ²)
  2. Discrete: φ_{ij} = +1 with prob p, -1 with prob (1-p)

Overdamped Langevin dynamics on the manifold:
    dx = -γ⁻¹ P_TM(∇U) dt + √(2D) P_TM(dW_t)
where D = k_B T / γ (Einstein relation).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize
from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SimulationParams:
    """Container for simulation parameters."""
    N: int = 100                    # Number of particles
    R: float = 1.0                  # Sphere radius
    temperature: float = 1.0        # Temperature T
    gamma: float = 1.0              # Friction coefficient
    dt: float = 0.001               # Time step
    coupling_type: str = 'gaussian' # 'gaussian' or 'discrete'
    sigma: float = 1.0              # Std dev for Gaussian couplings
    p: float = 0.5                  # Probability of +1 for discrete
    coupling_seed: int = 42         # Seed for frozen couplings
    init_seed: int = 123            # Seed for initial positions


class BrownianParticlesS2:
    """
    Simulator for interacting Brownian particles on the 2-sphere S².
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.N = params.N
        self.R = params.R
        self.T = params.temperature
        self.gamma = params.gamma
        self.dt = params.dt
        self.D = params.temperature / params.gamma
        
        # Initialize positions
        np.random.seed(params.init_seed)
        self.positions = self._random_points_on_S2(self.N)
        self.initial_positions = self.positions.copy()
        
        # Generate frozen couplings
        np.random.seed(params.coupling_seed)
        self.phi = self._generate_couplings(params.coupling_type, 
                                            params.sigma, params.p)
        
        # History storage
        self.energy_history: List[float] = []
        self.time_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.current_time = 0.0
        
    def _random_points_on_S2(self, n: int) -> np.ndarray:
        """Generate n uniformly distributed random points on S²."""
        points = np.random.randn(n, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return self.R * points / norms
    
    def _generate_couplings(self, coupling_type: str, 
                           sigma: float, p: float) -> np.ndarray:
        """Generate the frozen symmetric coupling matrix φ_{ij}."""
        N = self.N
        phi = np.zeros((N, N))
        
        if coupling_type == 'gaussian':
            upper = np.triu(np.random.randn(N, N) * sigma, k=1)
            phi = upper + upper.T
        elif coupling_type == 'discrete':
            upper = np.triu(2 * (np.random.random((N, N)) < p).astype(float) - 1, k=1)
            phi = upper + upper.T
        
        return phi
    
    def geodesic_distance_matrix(self) -> np.ndarray:
        """Compute matrix of all pairwise geodesic distances."""
        dots = self.positions @ self.positions.T
        cos_angles = np.clip(dots / (self.R ** 2), -1.0, 1.0)
        return self.R * np.arccos(cos_angles)
    
    def total_energy(self) -> float:
        """Compute total potential energy."""
        dist_matrix = self.geodesic_distance_matrix()
        return np.sum(np.triu(self.phi * dist_matrix, k=1))
    
    def compute_forces(self) -> np.ndarray:
        """Compute forces on all particles using vectorized operations."""
        forces = np.zeros((self.N, 3))
        pos = self.positions
        R2 = self.R ** 2
        
        for i in range(self.N):
            xi = pos[i]
            
            # Vectorized computation for all j != i
            diffs = pos - xi  # (N, 3)
            dots = np.sum(pos * xi, axis=1)  # x_j · x_i for all j
            
            # Geodesic distances
            cos_angles = np.clip(dots / R2, -1.0, 1.0)
            distances = self.R * np.arccos(cos_angles)
            
            # Project x_j onto tangent space at x_i
            proj_factor = dots / R2
            proj_xj = pos - np.outer(proj_factor, xi)  # (N, 3)
            
            # Norms of projections
            proj_norms = np.linalg.norm(proj_xj, axis=1)
            
            # Avoid division by zero
            valid = (proj_norms > 1e-10) & (distances > 1e-10)
            
            # Unit tangent vectors toward each j
            tangent = np.zeros_like(proj_xj)
            tangent[valid] = proj_xj[valid] / proj_norms[valid, np.newaxis]
            
            # Gradient of d w.r.t. x_i is -tangent (moving toward j decreases d)
            # Force = -φ * grad(d) = -φ * (-tangent) = φ * tangent
            # Wait, let's be careful:
            # U = sum φ_{ij} d(x_i, x_j)
            # F_i = -∇_i U = -sum_j φ_{ij} ∇_i d(x_i, x_j)
            # ∇_i d = -tangent_toward_j
            # So F_i = -sum_j φ_{ij} * (-tangent) = sum_j φ_{ij} * tangent
            
            force_contrib = self.phi[i, :, np.newaxis] * tangent  # (N, 3)
            forces[i] = np.sum(force_contrib, axis=0)
        
        return forces
    
    def _project_to_tangent_space(self, vectors: np.ndarray, 
                                   points: np.ndarray) -> np.ndarray:
        """Project vectors onto tangent spaces at given points."""
        dots = np.sum(vectors * points, axis=1, keepdims=True)
        return vectors - (dots / (self.R ** 2)) * points
    
    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        """Project points back onto sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return self.R * points / norms
    
    def step(self) -> None:
        """Perform one Euler-Maruyama time step."""
        forces = self.compute_forces()
        noise = np.random.randn(self.N, 3)
        noise_tangent = self._project_to_tangent_space(noise, self.positions)
        
        drift = (forces / self.gamma) * self.dt
        diffusion = np.sqrt(2 * self.D * self.dt) * noise_tangent
        
        new_positions = self.positions + drift + diffusion
        self.positions = self._project_to_sphere(new_positions)
        self.current_time += self.dt
    
    def run(self, n_steps: int, record_every: int = 100,
            save_frames: bool = False, frame_every: int = 100,
            verbose: bool = True) -> None:
        """Run simulation."""
        if verbose:
            print(f"Running: {n_steps} steps, dt={self.dt}, N={self.N}")
            print(f"Coupling: {self.params.coupling_type}", end="")
            if self.params.coupling_type == 'gaussian':
                print(f", σ={self.params.sigma}")
            else:
                print(f", p={self.params.p}")
            print(f"T={self.T}, D={self.D}")
            print("-" * 50)
        
        for step in range(n_steps):
            self.step()
            
            if step % record_every == 0:
                self.energy_history.append(self.total_energy())
                self.time_history.append(self.current_time)
            
            if save_frames and step % frame_every == 0:
                self.position_history.append(self.positions.copy())
            
            if verbose and step % (n_steps // 10) == 0:
                print(f"Step {step:6d}/{n_steps}, t={self.current_time:.3f}, "
                      f"E={self.total_energy():.2f}")
        
        if verbose:
            print("Complete!")
    
    def get_spherical_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to spherical coordinates (θ, φ)."""
        x, y, z = self.positions.T
        r = np.linalg.norm(self.positions, axis=1)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x) % (2 * np.pi)
        return theta, phi


def create_sphere_mesh(R: float = 1.0, resolution: int = 40):
    """Create mesh for sphere visualization."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_animation(sim: BrownianParticlesS2, 
                     output_path: str = 'brownian_s2.mp4',
                     fps: int = 30,
                     duration: float = 10.0) -> None:
    """
    Create animation of particle evolution on the sphere.
    """
    if len(sim.position_history) < 2:
        print("Error: No position history. Run with save_frames=True")
        return
    
    n_frames = len(sim.position_history)
    
    # Setup figure
    fig = plt.figure(figsize=(14, 6))
    
    # 3D sphere view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    
    # Energy plot
    ax2 = fig.add_subplot(122)
    
    # Draw sphere surface once
    xs, ys, zs = create_sphere_mesh(sim.R, 30)
    
    # Initialize scatter plot
    pos0 = sim.position_history[0]
    scatter = ax1.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2],
                         c='red', s=25, alpha=0.8, edgecolors='darkred', 
                         linewidths=0.3)
    
    # Set axis limits
    lim = sim.R * 1.3
    ax1.set_xlim([-lim, lim])
    ax1.set_ylim([-lim, lim])
    ax1.set_zlim([-lim, lim])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Energy line
    energy_line, = ax2.plot([], [], 'b-', linewidth=1.5)
    time_marker, = ax2.plot([], [], 'ro', markersize=8)
    
    if len(sim.energy_history) > 0:
        ax2.set_xlim(0, sim.time_history[-1])
        e_min, e_max = min(sim.energy_history), max(sim.energy_history)
        margin = 0.1 * (e_max - e_min) if e_max > e_min else 1
        ax2.set_ylim(e_min - margin, e_max + margin)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Energy')
    ax2.grid(True, alpha=0.3)
    
    # Title
    coupling_info = (f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' 
                    else f"p={sim.params.p}")
    title_base = f'N={sim.N}, T={sim.T}, {sim.params.coupling_type} ({coupling_info})'
    
    def init():
        ax1.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue', 
                        linewidth=0, antialiased=True)
        return scatter, energy_line, time_marker
    
    def update(frame):
        # Update particle positions
        pos = sim.position_history[frame]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Rotate view
        ax1.view_init(elev=25, azim=frame * 0.5)
        
        # Update energy plot
        frame_ratio = frame / n_frames
        idx = int(frame_ratio * len(sim.energy_history))
        idx = min(idx, len(sim.energy_history) - 1)
        
        energy_line.set_data(sim.time_history[:idx+1], sim.energy_history[:idx+1])
        if idx < len(sim.time_history):
            time_marker.set_data([sim.time_history[idx]], [sim.energy_history[idx]])
        
        ax1.set_title(f'{title_base}\nFrame {frame}/{n_frames}')
        
        return scatter, energy_line, time_marker
    
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000/fps, blit=False)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    
    if output_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Claude'),
                                 bitrate=3000)
        except:
            print("FFMpeg not available, saving as GIF instead")
            output_path = output_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
    
    anim.save(output_path, writer=writer, dpi=120)
    print(f"Animation saved: {output_path}")
    plt.close()


def create_summary_figure(sim: BrownianParticlesS2,
                         output_path: str = 'summary.png') -> None:
    """Create a summary figure with multiple views."""
    fig = plt.figure(figsize=(16, 10))
    
    # Initial state (3D)
    ax1 = fig.add_subplot(231, projection='3d')
    xs, ys, zs = create_sphere_mesh(sim.R, 30)
    ax1.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue')
    pos = sim.initial_positions
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=20, alpha=0.7)
    ax1.set_title('Initial Configuration')
    ax1.set_box_aspect([1, 1, 1])
    lim = sim.R * 1.2
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim]); ax1.set_zlim([-lim, lim])
    
    # Final state (3D)
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue')
    pos = sim.positions
    ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=20, alpha=0.7)
    ax2.set_title('Final Configuration (Stationary)')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim([-lim, lim]); ax2.set_ylim([-lim, lim]); ax2.set_zlim([-lim, lim])
    
    # Another view angle
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue')
    ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=20, alpha=0.7)
    ax3.view_init(elev=90, azim=0)
    ax3.set_title('Top View')
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_xlim([-lim, lim]); ax3.set_ylim([-lim, lim]); ax3.set_zlim([-lim, lim])
    
    # Coupling matrix
    ax4 = fig.add_subplot(234)
    im = ax4.imshow(sim.phi, cmap='RdBu_r', aspect='equal')
    ax4.set_title(f'Coupling Matrix φ_{{ij}} ({sim.params.coupling_type})')
    ax4.set_xlabel('j'); ax4.set_ylabel('i')
    plt.colorbar(im, ax=ax4)
    
    # Coupling histogram
    ax5 = fig.add_subplot(235)
    upper_tri = sim.phi[np.triu_indices(sim.N, k=1)]
    ax5.hist(upper_tri, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('φ value'); ax5.set_ylabel('Density')
    ax5.set_title(f'Coupling Distribution (mean={np.mean(upper_tri):.3f})')
    
    # Energy evolution
    ax6 = fig.add_subplot(236)
    if len(sim.energy_history) > 0:
        ax6.plot(sim.time_history, sim.energy_history, 'b-', linewidth=1)
        ax6.axhline(y=sim.energy_history[-1], color='r', linestyle='--',
                   label=f'Final: {sim.energy_history[-1]:.1f}')
        ax6.legend()
    ax6.set_xlabel('Time'); ax6.set_ylabel('Energy')
    ax6.set_title('Energy Evolution')
    ax6.grid(True, alpha=0.3)
    
    coupling_info = (f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' 
                    else f"p={sim.params.p}")
    fig.suptitle(f'Brownian Particles on S²: N={sim.N}, T={sim.T}, '
                f'{sim.params.coupling_type} couplings ({coupling_info})', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary saved: {output_path}")
    plt.close()


def create_mollweide_animation(sim: BrownianParticlesS2,
                               output_path: str = 'mollweide.gif',
                               fps: int = 20) -> None:
    """Create Mollweide projection animation."""
    if len(sim.position_history) < 2:
        print("No position history available")
        return
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='mollweide')
    
    def get_mollweide_coords(positions):
        x, y, z = positions.T
        r = np.linalg.norm(positions, axis=1)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        longitude = phi
        latitude = np.pi/2 - theta
        return longitude, latitude
    
    lon0, lat0 = get_mollweide_coords(sim.position_history[0])
    scatter = ax.scatter(lon0, lat0, c='red', s=15, alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    coupling_info = (f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' 
                    else f"p={sim.params.p}")
    
    def update(frame):
        lon, lat = get_mollweide_coords(sim.position_history[frame])
        scatter.set_offsets(np.c_[lon, lat])
        ax.set_title(f'Mollweide Projection - Frame {frame}/{len(sim.position_history)}\n'
                    f'N={sim.N}, T={sim.T}, {sim.params.coupling_type} ({coupling_info})')
        return scatter,
    
    anim = FuncAnimation(fig, update, frames=len(sim.position_history),
                        interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"Mollweide animation saved: {output_path}")
    plt.close()


def main():
    """Main demonstration."""
    print("=" * 70)
    print("BROWNIAN PARTICLES ON S² - SIMULATION WITH ANIMATION")
    print("=" * 70)
    
    # =========================================================================
    # Simulation 1: Gaussian Couplings
    # =========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION 1: Gaussian Couplings φ_{ij} ~ N(0, σ²)")
    print("=" * 70)
    
    params_gaussian = SimulationParams(
        N=150,
        R=1.0,
        temperature=0.3,
        gamma=1.0,
        dt=0.002,
        coupling_type='gaussian',
        sigma=1.0,
        coupling_seed=42,
        init_seed=123
    )
    
    sim_gauss = BrownianParticlesS2(params_gaussian)
    sim_gauss.run(n_steps=25000, record_every=50, 
                  save_frames=True, frame_every=50, verbose=True)
    
    # Create outputs
    create_summary_figure(sim_gauss, '/home/claude/summary_gaussian.png')
    create_animation(sim_gauss, '/home/claude/brownian_gaussian.gif', fps=25)
    create_mollweide_animation(sim_gauss, '/home/claude/mollweide_gaussian.gif', fps=20)
    
    # =========================================================================
    # Simulation 2: Discrete Couplings (balanced)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION 2: Discrete Couplings φ_{ij} = ±1 (p=0.5)")
    print("=" * 70)
    
    params_discrete = SimulationParams(
        N=150,
        R=1.0,
        temperature=0.3,
        gamma=1.0,
        dt=0.002,
        coupling_type='discrete',
        p=0.5,
        coupling_seed=42,
        init_seed=123
    )
    
    sim_discrete = BrownianParticlesS2(params_discrete)
    sim_discrete.run(n_steps=25000, record_every=50,
                     save_frames=True, frame_every=50, verbose=True)
    
    create_summary_figure(sim_discrete, '/home/claude/summary_discrete.png')
    create_animation(sim_discrete, '/home/claude/brownian_discrete.gif', fps=25)
    create_mollweide_animation(sim_discrete, '/home/claude/mollweide_discrete.gif', fps=20)
    
    # =========================================================================
    # Simulation 3: Discrete Couplings (biased)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION 3: Discrete Couplings φ_{ij} = ±1 (p=0.2)")
    print("=" * 70)
    
    params_biased = SimulationParams(
        N=150,
        R=1.0,
        temperature=0.2,
        gamma=1.0,
        dt=0.002,
        coupling_type='discrete',
        p=0.2,
        coupling_seed=42,
        init_seed=123
    )
    
    sim_biased = BrownianParticlesS2(params_biased)
    sim_biased.run(n_steps=25000, record_every=50,
                   save_frames=True, frame_every=50, verbose=True)
    
    create_summary_figure(sim_biased, '/home/claude/summary_biased.png')
    create_animation(sim_biased, '/home/claude/brownian_biased.gif', fps=25)
    
    print("\n" + "=" * 70)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - summary_gaussian.png, brownian_gaussian.gif, mollweide_gaussian.gif")
    print("  - summary_discrete.png, brownian_discrete.gif, mollweide_discrete.gif")
    print("  - summary_biased.png, brownian_biased.gif")


if __name__ == "__main__":
    main()

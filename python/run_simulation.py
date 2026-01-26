"""
Brownian Particles on S² - Fast Demo Version
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


@dataclass
class SimulationParams:
    N: int = 100
    R: float = 1.0
    temperature: float = 1.0
    gamma: float = 1.0
    dt: float = 0.001
    coupling_type: str = 'gaussian'
    sigma: float = 1.0
    p: float = 0.5
    coupling_seed: int = 42
    init_seed: int = 123


class BrownianParticlesS2:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.N = params.N
        self.R = params.R
        self.T = params.temperature
        self.gamma = params.gamma
        self.dt = params.dt
        self.D = params.temperature / params.gamma
        
        np.random.seed(params.init_seed)
        self.positions = self._random_points_on_S2(self.N)
        self.initial_positions = self.positions.copy()
        
        np.random.seed(params.coupling_seed)
        self.phi = self._generate_couplings()
        
        self.energy_history: List[float] = []
        self.time_history: List[float] = []
        self.position_history: List[np.ndarray] = []
        self.current_time = 0.0
        
    def _random_points_on_S2(self, n: int) -> np.ndarray:
        points = np.random.randn(n, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return self.R * points / norms
    
    def _generate_couplings(self) -> np.ndarray:
        N = self.N
        phi = np.zeros((N, N))
        
        if self.params.coupling_type == 'gaussian':
            upper = np.triu(np.random.randn(N, N) * self.params.sigma, k=1)
        else:
            upper = np.triu(2 * (np.random.random((N, N)) < self.params.p).astype(float) - 1, k=1)
        
        return upper + upper.T
    
    def geodesic_distance_matrix(self) -> np.ndarray:
        dots = self.positions @ self.positions.T
        cos_angles = np.clip(dots / (self.R ** 2), -1.0, 1.0)
        return self.R * np.arccos(cos_angles)
    
    def total_energy(self) -> float:
        dist_matrix = self.geodesic_distance_matrix()
        return np.sum(np.triu(self.phi * dist_matrix, k=1))
    
    def compute_forces(self) -> np.ndarray:
        forces = np.zeros((self.N, 3))
        pos = self.positions
        R2 = self.R ** 2
        
        for i in range(self.N):
            xi = pos[i]
            dots = np.sum(pos * xi, axis=1)
            cos_angles = np.clip(dots / R2, -1.0, 1.0)
            distances = self.R * np.arccos(cos_angles)
            
            proj_factor = dots / R2
            proj_xj = pos - np.outer(proj_factor, xi)
            proj_norms = np.linalg.norm(proj_xj, axis=1)
            
            valid = (proj_norms > 1e-10) & (distances > 1e-10)
            tangent = np.zeros_like(proj_xj)
            tangent[valid] = proj_xj[valid] / proj_norms[valid, np.newaxis]
            
            force_contrib = self.phi[i, :, np.newaxis] * tangent
            forces[i] = np.sum(force_contrib, axis=0)
        
        return forces
    
    def _project_to_tangent_space(self, vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
        dots = np.sum(vectors * points, axis=1, keepdims=True)
        return vectors - (dots / (self.R ** 2)) * points
    
    def _project_to_sphere(self, points: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return self.R * points / norms
    
    def step(self) -> None:
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
        if verbose:
            print(f"Running: {n_steps} steps, N={self.N}, T={self.T}")
            print(f"Coupling: {self.params.coupling_type}")
        
        for step in range(n_steps):
            self.step()
            
            if step % record_every == 0:
                self.energy_history.append(self.total_energy())
                self.time_history.append(self.current_time)
            
            if save_frames and step % frame_every == 0:
                self.position_history.append(self.positions.copy())
            
            if verbose and step % (n_steps // 5) == 0:
                print(f"  Step {step}/{n_steps}, E={self.total_energy():.2f}")
        
        if verbose:
            print("  Done!")


def create_sphere_mesh(R: float = 1.0, resolution: int = 30):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_summary_figure(sim: BrownianParticlesS2, output_path: str) -> None:
    fig = plt.figure(figsize=(16, 10))
    xs, ys, zs = create_sphere_mesh(sim.R, 30)
    lim = sim.R * 1.2
    
    # Initial state
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue')
    pos = sim.initial_positions
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=20, alpha=0.7)
    ax1.set_title('Initial Configuration')
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim]); ax1.set_zlim([-lim, lim])
    
    # Final state
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue')
    pos = sim.positions
    ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=20, alpha=0.7)
    ax2.set_title('Final (Stationary) Configuration')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim([-lim, lim]); ax2.set_ylim([-lim, lim]); ax2.set_zlim([-lim, lim])
    
    # Top view
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
    ax5.hist(upper_tri, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax5.set_xlabel('φ value'); ax5.set_ylabel('Density')
    ax5.set_title(f'Coupling Distribution')
    
    # Energy evolution
    ax6 = fig.add_subplot(236)
    ax6.plot(sim.time_history, sim.energy_history, 'b-', linewidth=1)
    ax6.axhline(y=sim.energy_history[-1], color='r', linestyle='--',
               label=f'Final: {sim.energy_history[-1]:.1f}')
    ax6.legend()
    ax6.set_xlabel('Time'); ax6.set_ylabel('Energy')
    ax6.set_title('Energy Evolution')
    ax6.grid(True, alpha=0.3)
    
    coupling_info = f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' else f"p={sim.params.p}"
    fig.suptitle(f'Brownian Particles on S²: N={sim.N}, T={sim.T}, {sim.params.coupling_type} ({coupling_info})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Summary saved: {output_path}")
    plt.close()


def create_animation(sim: BrownianParticlesS2, output_path: str, fps: int = 20) -> None:
    if len(sim.position_history) < 2:
        print("  No position history for animation")
        return
    
    n_frames = len(sim.position_history)
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    xs, ys, zs = create_sphere_mesh(sim.R, 25)
    ax1.plot_surface(xs, ys, zs, alpha=0.12, color='lightblue', linewidth=0)
    
    pos0 = sim.position_history[0]
    scatter = ax1.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2],
                         c='red', s=25, alpha=0.8, edgecolors='darkred', linewidths=0.3)
    
    lim = sim.R * 1.3
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim]); ax1.set_zlim([-lim, lim])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])
    
    energy_line, = ax2.plot([], [], 'b-', linewidth=1.5)
    time_marker, = ax2.plot([], [], 'ro', markersize=8)
    
    ax2.set_xlim(0, sim.time_history[-1])
    e_min, e_max = min(sim.energy_history), max(sim.energy_history)
    margin = 0.1 * abs(e_max - e_min) + 1
    ax2.set_ylim(e_min - margin, e_max + margin)
    ax2.set_xlabel('Time'); ax2.set_ylabel('Total Energy')
    ax2.grid(True, alpha=0.3)
    
    coupling_info = f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' else f"p={sim.params.p}"
    title_base = f'N={sim.N}, T={sim.T}, {sim.params.coupling_type} ({coupling_info})'
    
    def update(frame):
        pos = sim.position_history[frame]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        ax1.view_init(elev=25, azim=frame * 1.2)
        
        idx = min(int(frame / n_frames * len(sim.energy_history)), len(sim.energy_history) - 1)
        energy_line.set_data(sim.time_history[:idx+1], sim.energy_history[:idx+1])
        time_marker.set_data([sim.time_history[idx]], [sim.energy_history[idx]])
        
        ax1.set_title(f'{title_base}\nFrame {frame}/{n_frames}')
        return scatter, energy_line, time_marker
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    print(f"  Saving animation to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"  Animation saved!")
    plt.close()


def main():
    print("=" * 60)
    print("BROWNIAN PARTICLES ON S² - SIMULATION")
    print("=" * 60)
    
    # Simulation 1: Gaussian couplings
    print("\n[1] Gaussian Couplings φ ~ N(0, σ²)")
    params_gauss = SimulationParams(
        N=100, temperature=0.3, dt=0.003,
        coupling_type='gaussian', sigma=1.0,
        coupling_seed=42, init_seed=123
    )
    sim_gauss = BrownianParticlesS2(params_gauss)
    sim_gauss.run(n_steps=12000, record_every=30, save_frames=True, frame_every=60)
    create_summary_figure(sim_gauss, '/home/claude/summary_gaussian.png')
    create_animation(sim_gauss, '/home/claude/brownian_gaussian.gif', fps=20)
    
    # Simulation 2: Discrete couplings (balanced)
    print("\n[2] Discrete Couplings φ = ±1 (p=0.5)")
    params_discrete = SimulationParams(
        N=100, temperature=0.3, dt=0.003,
        coupling_type='discrete', p=0.5,
        coupling_seed=42, init_seed=123
    )
    sim_discrete = BrownianParticlesS2(params_discrete)
    sim_discrete.run(n_steps=12000, record_every=30, save_frames=True, frame_every=60)
    create_summary_figure(sim_discrete, '/home/claude/summary_discrete.png')
    create_animation(sim_discrete, '/home/claude/brownian_discrete.gif', fps=20)
    
    # Simulation 3: Discrete couplings (biased)
    print("\n[3] Discrete Couplings φ = ±1 (p=0.2, mostly repulsive)")
    params_biased = SimulationParams(
        N=100, temperature=0.2, dt=0.003,
        coupling_type='discrete', p=0.2,
        coupling_seed=42, init_seed=123
    )
    sim_biased = BrownianParticlesS2(params_biased)
    sim_biased.run(n_steps=12000, record_every=30, save_frames=True, frame_every=60)
    create_summary_figure(sim_biased, '/home/claude/summary_biased.png')
    create_animation(sim_biased, '/home/claude/brownian_biased.gif', fps=20)
    
    print("\n" + "=" * 60)
    print("ALL COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

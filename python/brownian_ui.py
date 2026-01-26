"""
Brownian Particles on S² - Fast Interactive UI
===============================================

Optimized Streamlit interface with vectorized numpy operations.

Run with: streamlit run brownian_ui.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st
from dataclasses import dataclass
from typing import List
import tempfile
import os
import time

plt.switch_backend('Agg')


@dataclass
class SimulationParams:
    N: int = 100
    R: float = 1.0
    temperature: float = 0.3
    gamma: float = 1.0
    dt: float = 0.005
    coupling_type: str = 'gaussian'
    sigma: float = 1.0
    p: float = 0.5
    coupling_seed: int = 42
    init_seed: int = 123


class BrownianParticlesS2Fast:
    """Optimized simulator using vectorized numpy operations."""
    
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
        if self.params.coupling_type == 'gaussian':
            upper = np.triu(np.random.randn(N, N) * self.params.sigma, k=1)
        else:
            upper = np.triu(2 * (np.random.random((N, N)) < self.params.p).astype(float) - 1, k=1)
        return upper + upper.T
    
    def total_energy(self) -> float:
        """Vectorized energy calculation."""
        dots = self.positions @ self.positions.T
        cos_angles = np.clip(dots / (self.R ** 2), -1.0, 1.0)
        distances = self.R * np.arccos(cos_angles)
        return np.sum(np.triu(self.phi * distances, k=1))
    
    def compute_forces_vectorized(self) -> np.ndarray:
        """Fully vectorized force computation - MUCH faster."""
        N = self.N
        pos = self.positions
        R2 = self.R ** 2
        
        # Compute all dot products at once: (N, N)
        dots = pos @ pos.T
        
        # Geodesic distances
        cos_angles = np.clip(dots / R2, -1.0, 1.0)
        distances = self.R * np.arccos(cos_angles)
        
        forces = np.zeros((N, 3))
        
        for i in range(N):
            xi = pos[i]
            
            # All projections of x_j onto tangent space at x_i
            proj_factors = dots[i, :] / R2  # (N,)
            proj_xj = pos - np.outer(proj_factors, xi)  # (N, 3)
            
            # Norms of projections
            proj_norms = np.linalg.norm(proj_xj, axis=1)  # (N,)
            
            # Valid pairs (avoid division by zero)
            valid = (proj_norms > 1e-10) & (distances[i, :] > 1e-10)
            valid[i] = False  # No self-interaction
            
            # Unit tangent vectors
            tangent = np.zeros((N, 3))
            tangent[valid] = proj_xj[valid] / proj_norms[valid, np.newaxis]
            
            # Force on particle i from all j
            forces[i] = np.sum(self.phi[i, :, np.newaxis] * tangent, axis=0)
        
        return forces
    
    def step(self) -> None:
        """Single simulation step."""
        forces = self.compute_forces_vectorized()
        
        # Brownian noise
        noise = np.random.randn(self.N, 3)
        
        # Project noise onto tangent spaces
        dots = np.sum(noise * self.positions, axis=1, keepdims=True)
        noise_tangent = noise - (dots / (self.R ** 2)) * self.positions
        
        # Euler-Maruyama update
        drift = (forces / self.gamma) * self.dt
        diffusion = np.sqrt(2 * self.D * self.dt) * noise_tangent
        
        # Update and project back to sphere
        new_positions = self.positions + drift + diffusion
        norms = np.linalg.norm(new_positions, axis=1, keepdims=True)
        self.positions = self.R * new_positions / norms
        
        self.current_time += self.dt
    
    def run(self, n_steps: int, record_every: int = 50, 
            frame_every: int = 50, progress_callback=None) -> None:
        """Run simulation with progress updates."""
        for step in range(n_steps):
            self.step()
            
            if step % record_every == 0:
                self.energy_history.append(self.total_energy())
                self.time_history.append(self.current_time)
            
            if step % frame_every == 0:
                self.position_history.append(self.positions.copy())
            
            if progress_callback and step % max(1, n_steps // 100) == 0:
                progress_callback(step / n_steps)
        
        if progress_callback:
            progress_callback(1.0)
    
    def reset(self):
        self.positions = self.initial_positions.copy()
        self.energy_history = []
        self.time_history = []
        self.position_history = []
        self.current_time = 0.0


def create_sphere_mesh(R: float = 1.0, resolution: int = 30):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def create_snapshot_figure(sim: BrownianParticlesS2Fast, azim: float = 45) -> plt.Figure:
    """Create a snapshot of current state."""
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    xs, ys, zs = create_sphere_mesh(sim.R, 25)
    ax1.plot_surface(xs, ys, zs, alpha=0.15, color='lightblue', linewidth=0)
    
    pos = sim.positions
    # Adaptive size based on N
    size = max(5, 200 / sim.N)
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=size, alpha=0.8,
               edgecolors='darkred', linewidths=0.3)
    
    lim = sim.R * 1.2
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim]); ax1.set_zlim([-lim, lim])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_title(f'Particle Configuration (N={sim.N})')
    ax1.view_init(elev=25, azim=azim)
    
    ax2 = fig.add_subplot(122)
    if len(sim.energy_history) > 0:
        ax2.plot(sim.time_history, sim.energy_history, 'b-', linewidth=1)
        ax2.axhline(y=sim.energy_history[-1], color='r', linestyle='--', alpha=0.7,
                   label=f'Current: {sim.energy_history[-1]:.1f}')
        ax2.legend()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Energy')
    ax2.set_title('Energy Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_animation_file(sim: BrownianParticlesS2Fast, output_path: str, 
                          fps: int = 30, duration: float = 10.0,
                          progress_callback=None) -> str:
    """Create GIF animation."""
    if len(sim.position_history) < 2:
        return None
    
    n_frames = len(sim.position_history)
    target_frames = int(fps * duration)
    
    if n_frames > target_frames:
        frame_indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
    else:
        frame_indices = np.arange(n_frames)
    
    actual_frames = len(frame_indices)
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    xs, ys, zs = create_sphere_mesh(sim.R, 25)
    ax1.plot_surface(xs, ys, zs, alpha=0.12, color='lightblue', linewidth=0)
    
    pos0 = sim.position_history[0]
    size = max(5, 200 / sim.N)
    scatter = ax1.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2],
                         c='red', s=size, alpha=0.8, edgecolors='darkred', linewidths=0.3)
    
    lim = sim.R * 1.3
    ax1.set_xlim([-lim, lim]); ax1.set_ylim([-lim, lim]); ax1.set_zlim([-lim, lim])
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])
    
    energy_line, = ax2.plot([], [], 'b-', linewidth=1.5)
    time_marker, = ax2.plot([], [], 'ro', markersize=8)
    
    ax2.set_xlim(0, sim.time_history[-1] if sim.time_history else 1)
    if sim.energy_history:
        e_min, e_max = min(sim.energy_history), max(sim.energy_history)
        margin = 0.1 * abs(e_max - e_min) + 1
        ax2.set_ylim(e_min - margin, e_max + margin)
    ax2.set_xlabel('Time'); ax2.set_ylabel('Total Energy')
    ax2.grid(True, alpha=0.3)
    
    coupling_info = f"σ={sim.params.sigma}" if sim.params.coupling_type == 'gaussian' else f"p={sim.params.p}"
    title_base = f'N={sim.N}, T={sim.T}, {sim.params.coupling_type} ({coupling_info})'
    
    def update(frame_num):
        frame_idx = frame_indices[frame_num]
        pos = sim.position_history[frame_idx]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        ax1.view_init(elev=25, azim=frame_num * 0.8)
        
        energy_idx = min(int(frame_idx / n_frames * len(sim.energy_history)), len(sim.energy_history) - 1)
        energy_line.set_data(sim.time_history[:energy_idx+1], sim.energy_history[:energy_idx+1])
        if energy_idx < len(sim.time_history):
            time_marker.set_data([sim.time_history[energy_idx]], [sim.energy_history[energy_idx]])
        
        ax1.set_title(f'{title_base}\nt = {sim.time_history[energy_idx]:.2f}')
        
        if progress_callback and frame_num % 5 == 0:
            progress_callback(frame_num / actual_frames)
        
        return scatter, energy_line, time_marker
    
    anim = FuncAnimation(fig, update, frames=actual_frames, interval=1000/fps, blit=False)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    
    plt.close(fig)
    
    if progress_callback:
        progress_callback(1.0)
    
    return output_path


def main():
    st.set_page_config(
        page_title="Brownian Particles on S²",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Brownian Particles on S²")
    st.markdown("Interactive simulation of interacting Brownian particles on a sphere.")
    
    # Sidebar
    st.sidebar.header("🎛️ Parameters")
    
    st.sidebar.subheader("Particles & Temperature")
    N = st.sidebar.slider("N (particles)", 10, 300, 100, 5)
    T = st.sidebar.slider("T (temperature)", 0.01, 5.0, 0.3, 0.01)
    
    st.sidebar.subheader("Coupling")
    coupling_type = st.sidebar.radio("Type", ["gaussian", "discrete"])
    
    if coupling_type == "gaussian":
        sigma = st.sidebar.slider("σ (std dev)", 0.01, 10.0, 1.0, 0.01)
        p = 0.5
        st.sidebar.info(f"φᵢⱼ ~ N(0, {sigma:.2f}²)")
    else:
        p = st.sidebar.slider("p (prob +1)", 0.0, 1.0, 0.5, 0.001)
        sigma = 1.0
        st.sidebar.info(f"φᵢⱼ = +1 (p={p:.3f}), -1 (p={1-p:.3f})")
    
    st.sidebar.subheader("Simulation")
    n_steps = st.sidebar.slider("Steps", 1000, 50000, 10000, 500)
    dt = st.sidebar.select_slider("dt", options=[0.002, 0.005, 0.01, 0.02], value=0.005)
    
    st.sidebar.subheader("Seeds")
    col1, col2 = st.sidebar.columns(2)
    coupling_seed = col1.number_input("φ seed", 0, 9999, 42)
    init_seed = col2.number_input("Init seed", 0, 9999, 123)
    
    st.sidebar.subheader("Movie")
    movie_duration = st.sidebar.slider("Duration (s)", 5, 60, 15)
    movie_fps = st.sidebar.slider("FPS", 15, 60, 30)
    frame_every = st.sidebar.slider("Frame every", 10, 200, 30)
    
    # Create params
    params = SimulationParams(
        N=N, temperature=T, coupling_type=coupling_type,
        sigma=sigma, p=p, dt=dt,
        coupling_seed=int(coupling_seed), init_seed=int(init_seed)
    )
    
    # Session state
    if 'sim' not in st.session_state:
        st.session_state.sim = None
        st.session_state.sim_complete = False
        st.session_state.azim = 45
    
    # Main area
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            st.session_state.sim = BrownianParticlesS2Fast(params)
            st.session_state.sim_complete = False
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            def update_progress(p):
                progress_bar.progress(p)
                status.text(f"Simulating... {p*100:.0f}%")
            
            start = time.time()
            st.session_state.sim.run(n_steps, record_every=20, 
                                     frame_every=frame_every,
                                     progress_callback=update_progress)
            elapsed = time.time() - start
            
            st.session_state.sim_complete = True
            status.text(f"✅ Done in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")
        
        # Display
        if st.session_state.sim is not None and st.session_state.sim_complete:
            sim = st.session_state.sim
            
            # Rotation control
            st.session_state.azim = st.slider("Rotate view", 0, 360, int(st.session_state.azim), 5)
            
            fig = create_snapshot_figure(sim, st.session_state.azim)
            st.pyplot(fig)
            plt.close(fig)
            
            # Stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("N", sim.N)
            c2.metric("T", f"{sim.T:.2f}")
            c3.metric("Energy", f"{sim.energy_history[-1]:.1f}")
            c4.metric("Frames", len(sim.position_history))
    
    with col_side:
        st.subheader("🎬 Export")
        
        if st.session_state.sim is not None and st.session_state.sim_complete:
            sim = st.session_state.sim
            
            if len(sim.position_history) > 1:
                if st.button("Generate GIF", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                        output_path = tmp.name
                    
                    start = time.time()
                    create_animation_file(sim, output_path, fps=movie_fps, 
                                         duration=movie_duration,
                                         progress_callback=lambda p: progress.progress(p))
                    elapsed = time.time() - start
                    
                    status.text(f"✅ Created in {elapsed:.1f}s")
                    
                    with open(output_path, 'rb') as f:
                        gif_data = f.read()
                    
                    st.download_button(
                        "⬇️ Download GIF",
                        data=gif_data,
                        file_name=f"brownian_N{N}_T{T}.gif",
                        mime="image/gif",
                        use_container_width=True
                    )
                    
                    st.image(gif_data, caption="Preview")
                    os.unlink(output_path)
        else:
            st.info("Run simulation first")
    
    # Math details
    with st.expander("📐 Physics"):
        st.markdown(r"""
**Model:** Particles on S² with pairwise geodesic interactions

$$U = \sum_{i<j} \phi_{ij} \cdot d(x_i, x_j)$$

where $d$ is geodesic distance: $d(x_i, x_j) = \arccos(x_i \cdot x_j)$

**Couplings:**
- Gaussian: $\phi_{ij} \sim \mathcal{N}(0, \sigma^2)$
- Discrete: $\phi_{ij} = \pm 1$

**Dynamics:** Overdamped Langevin on manifold
$$dx = -\gamma^{-1} \nabla U \, dt + \sqrt{2D} \, dW$$
        """)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TPU L2 Simulation: Excitations and Fields (updated)
- Stable mode selection
- Correct gradient spacing
- float32 arrays for performance
- deterministic RNG
- precomputed coords
- NDJSON diagnostics (diagnostics.ndjson)
- efficient seeding of top excitations
- checkpointed diagnostics and final summary
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import os

class TPU_L2_Simulation:
    def __init__(self, l0_l1_data_dir="tpu_output_V56c_auto", output_dir="L2_excitations"):
        self.l0_l1_dir = Path(l0_l1_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("🌌 TPU L2 Simulation: Excitations and Fields")
        print(f"Loading foundation data from: {self.l0_l1_dir}")
        print(f"Output directory: {self.output_dir}")

        # L2-specific parameters
        self.field_modes = 50
        self.excitation_threshold = 0.1
        self.propagation_steps = 10000
        self.Nseed = 100  # number of top excitations to seed
        self.checkpoint_interval = 100
        self.progress_interval = 1000

        # Load foundation and prepare
        self.load_foundation_data()

    def load_foundation_data(self):
        """Load L0/L1 foundation as boundary conditions"""
        try:
            self.substrate_config = np.load(self.l0_l1_dir / "V56c_auto_final_sigma.npy")
            self.foundation_times = np.load(self.l0_l1_dir / "V56c_auto_times.npy")
            self.foundation_masses = np.load(self.l0_l1_dir / "V56c_auto_masses.npy")
            self.foundation_temps = np.load(self.l0_l1_dir / "V56c_auto_temps.npy")

            # Grid and spacing
            self.Nx, self.Ny, self.Nz = self.substrate_config.shape
            self.L = 30.0
            self.dx = 2.0 * self.L / (self.Nx - 1)

            # Reproducibility and memory
            self.rng = np.random.default_rng(42)
            self.substrate_config = self.substrate_config.astype(np.float32)

            # Precompute coordinate grids (float32)
            self.x_coords, self.y_coords, self.z_coords = np.meshgrid(
                np.arange(self.Nx, dtype=np.float32),
                np.arange(self.Ny, dtype=np.float32),
                np.arange(self.Nz, dtype=np.float32),
                indexing='ij'
            )

            print("✅ Foundation data loaded successfully")
            print(f"   Grid: {self.Nx}x{self.Ny}x{self.Nz}")
            print(f"   Final foundation mass: {self.foundation_masses[-1]:.3f}")
            print(f"   Foundation σ range: [{self.substrate_config.min():.3f}, {self.substrate_config.max():.3f}]")

        except Exception as e:
            print(f"❌ Error loading foundation data: {e}")
            raise

    # --- Diagnostics helper (NDJSON) ---
    def _diag_path(self):
        return self.output_dir / "diagnostics.ndjson"

    def emit_diag(self, tag, payload):
        rec = {"ts": time.time(), "tag": tag, "payload": payload}
        with open(self._diag_path(), "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def compute_field_modes(self):
        """Extract field modes from foundation σ patterns"""
        print("🔬 Computing L2 field modes from foundation...")

        fft_substrate = np.fft.fftn(self.substrate_config.astype(np.complex64))
        power_spectrum = np.abs(fft_substrate) ** 2

        flat_power = power_spectrum.flatten()
        mode_indices = np.argsort(flat_power)[-self.field_modes:][::-1]  # descending
        mode_coords = np.unravel_index(mode_indices, power_spectrum.shape)
        mode_amplitudes = flat_power[mode_indices]

        self.field_modes_data = np.zeros((self.field_modes, 4), dtype=np.float32)  # kx,ky,kz,amplitude
        for i in range(self.field_modes):
            kx = mode_coords[0][i] if mode_coords[0][i] < self.Nx // 2 else mode_coords[0][i] - self.Nx
            ky = mode_coords[1][i] if mode_coords[1][i] < self.Ny // 2 else mode_coords[1][i] - self.Ny
            kz = mode_coords[2][i] if mode_coords[2][i] < self.Nz // 2 else mode_coords[2][i] - self.Nz
            self.field_modes_data[i, 0] = float(kx)
            self.field_modes_data[i, 1] = float(ky)
            self.field_modes_data[i, 2] = float(kz)
            self.field_modes_data[i, 3] = float(np.sqrt(mode_amplitudes[i]))

        # Emit diagnostics for modes
        top_modes = [{"kx": int(self.field_modes_data[i,0]),
                      "ky": int(self.field_modes_data[i,1]),
                      "kz": int(self.field_modes_data[i,2]),
                      "amp": float(self.field_modes_data[i,3])}
                     for i in range(min(20, self.field_modes))]
        self.emit_diag("l2_mode_energies_top20", {"modes": top_modes})

        print(f"✅ Extracted {self.field_modes} dominant field modes")
        return self.field_modes_data

    def compute_excitation_spectrum(self):
        """Compute excitation spectrum from foundation interactions"""
        print("⚡ Computing L2 excitation spectrum...")

        grad_x = np.gradient(self.substrate_config, self.dx, axis=0)
        grad_y = np.gradient(self.substrate_config, self.dx, axis=1)
        grad_z = np.gradient(self.substrate_config, self.dx, axis=2)

        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        excitation_mask = grad_magnitude > self.excitation_threshold
        excitation_locations = np.argwhere(excitation_mask)
        excitation_energies = grad_magnitude[excitation_mask].astype(np.float32)

        total_ex = int(len(excitation_locations))
        mean_e = float(np.mean(excitation_energies)) if total_ex > 0 else 0.0
        max_e = float(np.max(excitation_energies)) if total_ex > 0 else 0.0

        self.excitation_spectrum = {
            'locations': excitation_locations.astype(np.int32),
            'energies': excitation_energies,
            'total_excitations': total_ex,
            'mean_energy': mean_e,
            'max_energy': max_e
        }

        # Emit excitation histogram summary
        bins = [0.0, 0.1, 1.0, 10.0, 100.0]
        hist, edges = np.histogram(excitation_energies, bins=bins)
        hist_summary = {f"{edges[i]:.3f}-{edges[i+1]:.3f}": int(hist[i]) for i in range(len(hist))}
        self.emit_diag("l2_excitation_summary", {
            "total_excitations": total_ex,
            "mean_energy": mean_e,
            "max_energy": max_e,
            "histogram": hist_summary
        })

        print(f"✅ Found {self.excitation_spectrum['total_excitations']} excitation sites")
        print(f"   Mean energy: {self.excitation_spectrum['mean_energy']:.3f}")
        print(f"   Max energy: {self.excitation_spectrum['max_energy']:.3f}")

        return self.excitation_spectrum

    def simulate_field_propagation(self):
        """Model how L2 excitations propagate through universe"""
        print("🌊 Simulating L2 field propagation...")

        propagation_data = {
            'times': [],
            'field_energies': [],
            'propagation_distances': [],
            'interaction_events': []
        }

        current_field = np.zeros_like(self.substrate_config, dtype=np.float32)

        energies = np.asarray(self.excitation_spectrum['energies'], dtype=float)
        locs = np.asarray(self.excitation_spectrum['locations'], dtype=int)
        Nseed = min(self.Nseed, len(energies))
        if Nseed > 0:
            top_idx = np.argsort(energies)[-Nseed:][::-1]
            for i in top_idx:
                x, y, z = locs[i]
                current_field[int(x), int(y), int(z)] = float(energies[i])

        dt = 0.01
        diffusion_coeff = 0.1
        interaction_prob = 1e-6
        rng = self.rng

        # Emit unit mapping once
        self.emit_diag("unit_mapping", {"Nx": int(self.Nx), "Ny": int(self.Ny), "Nz": int(self.Nz), "L": float(self.L), "dx": float(self.dx)})

        x_coords = self.x_coords  # float32
        y_coords = self.y_coords
        z_coords = self.z_coords

        for step in range(self.propagation_steps):
            laplacian = (np.roll(current_field, 1, 0) + np.roll(current_field, -1, 0) +
                         np.roll(current_field, 1, 1) + np.roll(current_field, -1, 1) +
                         np.roll(current_field, 1, 2) + np.roll(current_field, -1, 2) -
                         6.0 * current_field) / (self.dx**2)

            current_field += dt * diffusion_coeff * laplacian

            if rng.random() < interaction_prob:
                ix, iy, iz = rng.integers(0, self.Nx, size=3)
                interaction_strength = float(rng.exponential(0.1))
                current_field[ix, iy, iz] *= (1.0 - interaction_strength)
                propagation_data['interaction_events'].append((step * dt, int(ix), int(iy), int(iz), interaction_strength))
                self.emit_diag("l2_interaction_event", {
                    "step": int(step), "time": float(step * dt),
                    "ix": int(ix), "iy": int(iy), "iz": int(iz), "strength": float(interaction_strength)
                })

            if (step % self.checkpoint_interval) == 0:
                total_energy = float(np.sum(current_field**2))
                propagation_data['times'].append(float(step * dt))
                propagation_data['field_energies'].append(total_energy)

                if total_energy > 0.0:
                    center_x = float(np.sum(current_field**2 * x_coords) / total_energy)
                    center_y = float(np.sum(current_field**2 * y_coords) / total_energy)
                    center_z = float(np.sum(current_field**2 * z_coords) / total_energy)

                    rms_distance = float(np.sqrt(np.sum(current_field**2 *
                                                        ((x_coords - center_x)**2 +
                                                         (y_coords - center_y)**2 +
                                                         (z_coords - center_z)**2)) / total_energy) * self.dx)
                else:
                    rms_distance = 0.0

                propagation_data['propagation_distances'].append(rms_distance)
                self.emit_diag("checkpoint_summary", {
                    "step": int(step), "time": float(step * dt),
                    "field_energy": total_energy, "rms_distance": rms_distance,
                    "interaction_count": len(propagation_data['interaction_events'])
                })

            if (step % self.progress_interval) == 0:
                print(f"   Step {step}/{self.propagation_steps}, Field energy: {np.sum(current_field**2):.3e}")

        # Final diagnostics
        final_energy = float(propagation_data['field_energies'][-1]) if propagation_data['field_energies'] else 0.0
        max_prop = float(max(propagation_data['propagation_distances'])) if propagation_data['propagation_distances'] else 0.0
        total_interactions = len(propagation_data['interaction_events'])
        self.emit_diag("final_summary", {
            "final_field_energy": final_energy,
            "max_propagation_distance": max_prop,
            "total_interactions": total_interactions
        })

        self.propagation_data = propagation_data
        self.final_field = current_field

        print("✅ Propagation simulation completed")
        print(f"   Final field energy: {final_energy:.3e}")
        print(f"   Max propagation distance: {max_prop:.3f}")
        print(f"   Total interaction events: {total_interactions}")

        return propagation_data

    def save_l2_data(self):
        """Save all L2 simulation results"""
        print("💾 Saving L2 simulation data...")
        try:
            np.save(self.output_dir / "field_modes.npy", self.field_modes_data.astype(np.float32))
            np.save(self.output_dir / "excitation_locations.npy", self.excitation_spectrum['locations'])
            np.save(self.output_dir / "excitation_energies.npy", self.excitation_spectrum['energies'].astype(np.float32))

            np.save(self.output_dir / "propagation_times.npy", np.array(self.propagation_data['times'], dtype=np.float32))
            np.save(self.output_dir / "field_energies.npy", np.array(self.propagation_data['field_energies'], dtype=np.float32))
            np.save(self.output_dir / "propagation_distances.npy", np.array(self.propagation_data['propagation_distances'], dtype=np.float32))

            np.save(self.output_dir / "final_field.npy", self.final_field.astype(np.float32))

            if self.propagation_data['interaction_events']:
                interaction_array = np.array(self.propagation_data['interaction_events'], dtype=object)
                np.save(self.output_dir / "interaction_events.npy", interaction_array)

            print("✅ All L2 data saved successfully")
        except Exception as e:
            print(f"❌ Error saving L2 data: {e}")

    def create_l2_plots(self):
        """Generate L2 analysis plots"""
        print("📊 Creating L2 analysis plots...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].plot(self.propagation_data['times'], self.propagation_data['field_energies'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Total Field Energy')
            axes[0, 0].set_title('L2 Field Energy Evolution')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')

            axes[0, 1].plot(self.propagation_data['times'], self.propagation_data['propagation_distances'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('RMS Propagation Distance')
            axes[0, 1].set_title('L2 Field Propagation Distance')
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].hist(self.excitation_spectrum['energies'], bins=50, alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Excitation Energy')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('L2 Excitation Energy Distribution')
            axes[1, 0].grid(True, alpha=0.3)

            mode_magnitudes = self.field_modes_data[:, 3]
            axes[1, 1].plot(range(len(mode_magnitudes)), mode_magnitudes, 'mo-', markersize=4)
            axes[1, 1].set_xlabel('Mode Index')
            axes[1, 1].set_ylabel('Mode Amplitude')
            axes[1, 1].set_title('L2 Field Mode Spectrum')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')

            plt.tight_layout()
            plt.savefig(self.output_dir / "L2_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()

            mid = self.Nz // 2
            plt.figure(figsize=(10, 8))
            plt.imshow(self.final_field[:, :, mid], extent=[-self.L, self.L, -self.L, self.L],
                       origin='lower', cmap='plasma')
            plt.colorbar(label='Field Amplitude')
            plt.title('L2 Final Field Configuration (z=0 slice)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(self.output_dir / "L2_final_field.png", dpi=150, bbox_inches='tight')
            plt.close()

            print("✅ L2 plots created successfully")
        except Exception as e:
            print(f"❌ Error creating L2 plots: {e}")

    def run_l2_simulation(self):
        """Run complete L2 simulation"""
        print("🚀 Starting L2 Excitations and Fields Simulation")
        print("=" * 60)

        self.compute_field_modes()
        self.compute_excitation_spectrum()
        self.simulate_field_propagation()
        self.save_l2_data()
        self.create_l2_plots()

        print("=" * 60)
        print("🎉 L2 Simulation completed successfully!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"🔬 Field modes: {self.field_modes}")
        print(f"⚡ Excitation sites: {self.excitation_spectrum['total_excitations']}")
        print(f"🌊 Propagation steps: {self.propagation_steps}")
        print(f"🔗 Interaction events: {len(self.propagation_data['interaction_events'])}")
        print(f"📁 Diagnostics: {self._diag_path()}")

if __name__ == "__main__":
    sim = TPU_L2_Simulation()
    sim.run_l2_simulation()
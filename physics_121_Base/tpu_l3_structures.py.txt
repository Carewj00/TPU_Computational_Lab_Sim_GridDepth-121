"""
TPU Simulation L3: Persistent Structures from L2 Fields
Models long-lived excitations that rarely interact with L0/L1 foundation
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class TPU_L3_Simulation:
    def __init__(self, l0_l1_data_dir="tpu_output_V56c_auto", l2_data_dir="L2_excitations", output_dir="L3_structures"):
        self.l0_l1_dir = Path(l0_l1_data_dir)
        self.l2_dir = Path(l2_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🌌 TPU L3 Simulation: Persistent Structures")
        print(f"Loading L0/L1 foundation from: {self.l0_l1_dir}")
        print(f"Loading L2 field data from: {self.l2_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Load dependency data
        self.load_dependency_data()
        
        # L3-specific parameters
        self.structure_lifetime_threshold = 1000  # Minimum lifetime for persistence
        self.interaction_probability = 1e-8  # Very rare L0/L1 interactions
        self.propagation_distance_limit = 1e6  # Billion-year scale propagation
        self.stability_analysis_steps = 20000
        
    def load_dependency_data(self):
        """Load L0/L1 foundation and L2 field data"""
        try:
            # Load L0/L1 foundation
            self.substrate_config = np.load(self.l0_l1_dir / "V56c_auto_final_sigma.npy")
            self.foundation_times = np.load(self.l0_l1_dir / "V56c_auto_times.npy")
            self.foundation_masses = np.load(self.l0_l1_dir / "V56c_auto_masses.npy")
            
            # Load L2 field data
            self.field_modes = np.load(self.l2_dir / "field_modes.npy")
            self.final_field = np.load(self.l2_dir / "final_field.npy")
            self.propagation_times = np.load(self.l2_dir / "propagation_times.npy")
            self.field_energies = np.load(self.l2_dir / "field_energies.npy")
            
            # Extract grid parameters
            self.Nx, self.Ny, self.Nz = self.substrate_config.shape
            self.L = 30.0
            self.dx = 2 * self.L / (self.Nx - 1)
            
            print(f"✅ Dependency data loaded successfully")
            print(f"   Foundation σ range: [{self.substrate_config.min():.3f}, {self.substrate_config.max():.3f}]")
            print(f"   L2 field modes: {len(self.field_modes)}")
            print(f"   Final L2 field energy: {np.sum(self.final_field**2):.3e}")
            
        except Exception as e:
            print(f"❌ Error loading dependency data: {e}")
            raise
    
    def identify_persistent_structures(self):
        """Identify stable, long-lived structures from L2 fields"""
        print("🔍 Identifying persistent structures from L2 fields...")
        
        # Find local maxima in L2 field (potential structure seeds)
        from scipy import ndimage
        
        # Smooth the field to identify coherent structures
        smoothed_field = ndimage.gaussian_filter(self.final_field, sigma=2.0)
        
        # Find local maxima
        local_maxima = ndimage.maximum_filter(smoothed_field, size=5) == smoothed_field
        structure_mask = local_maxima & (smoothed_field > np.percentile(smoothed_field, 90))
        
        # Get structure locations and energies
        structure_locations = np.argwhere(structure_mask)
        structure_energies = smoothed_field[structure_mask]
        
        # Filter by energy threshold for persistence
        energy_threshold = np.percentile(structure_energies, 70)
        persistent_mask = structure_energies > energy_threshold
        
        self.persistent_locations = structure_locations[persistent_mask]
        self.persistent_energies = structure_energies[persistent_mask]
        
        print(f"✅ Found {len(self.persistent_locations)} persistent structures")
        print(f"   Energy range: [{self.persistent_energies.min():.3f}, {self.persistent_energies.max():.3f}]")
        print(f"   Mean energy: {self.persistent_energies.mean():.3f}")
        
        return self.persistent_locations, self.persistent_energies
    
    def simulate_structure_evolution(self):
        """Simulate long-term evolution of persistent structures"""
        print("⏳ Simulating long-term structure evolution...")
        
        # Initialize structure tracking
        num_structures = len(self.persistent_locations)
        structure_lifetimes = np.zeros(num_structures)
        structure_distances = np.zeros((self.stability_analysis_steps, num_structures))
        structure_energies_evolution = np.zeros((self.stability_analysis_steps, num_structures))
        interaction_events = []
        
        # Current structure positions (start at persistent locations)
        current_positions = self.persistent_locations.astype(float)
        current_energies = self.persistent_energies.copy()
        
        dt = 0.1  # Time step for evolution
        diffusion_coeff = 0.001  # Very slow diffusion for persistent structures
        
        for step in range(self.stability_analysis_steps):
            time = step * dt
            
            # Very slow random walk (structures are nearly stationary)
            position_noise = np.random.randn(*current_positions.shape) * np.sqrt(2 * diffusion_coeff * dt)
            current_positions += position_noise
            
            # Keep structures within bounds
            current_positions = np.clip(current_positions, 0, self.Nx - 1)
            
            # Energy decay (very slow for persistent structures)
            energy_decay_rate = 1e-5
            current_energies *= (1 - energy_decay_rate * dt)
            
            # Rare interactions with L0/L1 substrate
            for i in range(num_structures):
                if np.random.rand() < self.interaction_probability:
                    # Interaction with substrate
                    pos = current_positions[i].astype(int)
                    substrate_strength = self.substrate_config[tuple(pos)]
                    
                    # Interaction can either boost or drain energy
                    interaction_strength = np.random.exponential(0.1)
                    if np.random.rand() < 0.3:  # 30% chance of energy boost
                        current_energies[i] *= (1 + interaction_strength)
                    else:  # 70% chance of energy drain
                        current_energies[i] *= (1 - interaction_strength)
                    
                    interaction_events.append((time, i, pos, interaction_strength, substrate_strength))
            
            # Calculate propagation distances from origin
            origin_distances = np.sqrt(np.sum((current_positions - self.persistent_locations)**2, axis=1)) * self.dx
            structure_distances[step] = origin_distances
            structure_energies_evolution[step] = current_energies
            
            # Update lifetimes (structures are "alive" if energy > threshold)
            alive_mask = current_energies > 0.01 * self.persistent_energies
            structure_lifetimes[alive_mask] = time
            
            if step % 2000 == 0:
                alive_count = np.sum(alive_mask)
                mean_distance = np.mean(origin_distances[alive_mask]) if alive_count > 0 else 0
                print(f"   Step {step}/{self.stability_analysis_steps}, Alive: {alive_count}/{num_structures}, Mean distance: {mean_distance:.3f}")
        
        self.structure_lifetimes = structure_lifetimes
        self.structure_distances = structure_distances
        self.structure_energies_evolution = structure_energies_evolution
        self.interaction_events = interaction_events
        
        print(f"✅ Structure evolution completed")
        print(f"   Mean lifetime: {np.mean(structure_lifetimes):.1f}")
        print(f"   Max lifetime: {np.max(structure_lifetimes):.1f}")
        print(f"   Total interactions: {len(interaction_events)}")
        print(f"   Max propagation distance: {np.max(structure_distances):.3f}")
        
        return structure_lifetimes, structure_distances, interaction_events
    
    def analyze_billion_year_propagation(self):
        """Analyze structures that could propagate for billions of years"""
        print("🌌 Analyzing billion-year propagation potential...")
        
        # Identify ultra-stable structures
        long_lived_mask = self.structure_lifetimes > 0.8 * np.max(self.structure_lifetimes)
        long_lived_indices = np.where(long_lived_mask)[0]
        
        # Calculate propagation characteristics
        propagation_analysis = {
            'ultra_stable_count': len(long_lived_indices),
            'ultra_stable_energies': self.persistent_energies[long_lived_indices],
            'max_distances': np.max(self.structure_distances, axis=0)[long_lived_indices],
            'interaction_rates': [],
            'stability_factors': []
        }
        
        # Analyze interaction rates for ultra-stable structures
        for idx in long_lived_indices:
            structure_interactions = [event for event in self.interaction_events if event[1] == idx]
            interaction_rate = len(structure_interactions) / np.max(self.structure_lifetimes)
            propagation_analysis['interaction_rates'].append(interaction_rate)
            
            # Stability factor (energy retention over time)
            initial_energy = self.persistent_energies[idx]
            final_energy = self.structure_energies_evolution[-1, idx]
            stability_factor = final_energy / initial_energy if initial_energy > 0 else 0
            propagation_analysis['stability_factors'].append(stability_factor)
        
        self.propagation_analysis = propagation_analysis
        
        print(f"✅ Billion-year propagation analysis completed")
        print(f"   Ultra-stable structures: {propagation_analysis['ultra_stable_count']}")
        print(f"   Mean interaction rate: {np.mean(propagation_analysis['interaction_rates']):.2e} per time unit")
        print(f"   Mean stability factor: {np.mean(propagation_analysis['stability_factors']):.3f}")
        
        return propagation_analysis
    
    def save_l3_data(self):
        """Save all L3 simulation results"""
        print("💾 Saving L3 simulation data...")
        
        try:
            # Save persistent structure data
            np.save(self.output_dir / "persistent_locations.npy", self.persistent_locations)
            np.save(self.output_dir / "persistent_energies.npy", self.persistent_energies)
            
            # Save evolution data
            np.save(self.output_dir / "structure_lifetimes.npy", self.structure_lifetimes)
            np.save(self.output_dir / "structure_distances.npy", self.structure_distances)
            np.save(self.output_dir / "structure_energies_evolution.npy", self.structure_energies_evolution)
            
            # Save interaction events
            if self.interaction_events:
                interaction_array = np.array(self.interaction_events, dtype=object)
                np.save(self.output_dir / "interaction_events.npy", interaction_array)
            
            # Save propagation analysis
            np.save(self.output_dir / "ultra_stable_indices.npy", 
                   np.where(self.structure_lifetimes > 0.8 * np.max(self.structure_lifetimes))[0])
            np.save(self.output_dir / "interaction_rates.npy", 
                   np.array(self.propagation_analysis['interaction_rates']))
            np.save(self.output_dir / "stability_factors.npy", 
                   np.array(self.propagation_analysis['stability_factors']))
            
            print("✅ All L3 data saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving L3 data: {e}")
    
    def create_l3_plots(self):
        """Generate L3 analysis plots"""
        print("📊 Creating L3 analysis plots...")
        
        try:
            # Structure evolution plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Lifetime distribution
            axes[0,0].hist(self.structure_lifetimes, bins=30, alpha=0.7, color='blue')
            axes[0,0].set_xlabel('Structure Lifetime')
            axes[0,0].set_ylabel('Count')
            axes[0,0].set_title('L3 Structure Lifetime Distribution')
            axes[0,0].grid(True, alpha=0.3)
            
            # Energy evolution for top structures
            time_axis = np.arange(self.stability_analysis_steps) * 0.1
            top_indices = np.argsort(self.persistent_energies)[-5:]  # Top 5 structures
            for i, idx in enumerate(top_indices):
                axes[0,1].plot(time_axis, self.structure_energies_evolution[:, idx], 
                              label=f'Structure {idx}', alpha=0.7)
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('Energy')
            axes[0,1].set_title('L3 Energy Evolution (Top Structures)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_yscale('log')
            
            # Propagation distances
            mean_distances = np.mean(self.structure_distances, axis=1)
            max_distances = np.max(self.structure_distances, axis=1)
            axes[1,0].plot(time_axis, mean_distances, 'b-', label='Mean Distance', linewidth=2)
            axes[1,0].plot(time_axis, max_distances, 'r--', label='Max Distance', linewidth=2)
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Propagation Distance')
            axes[1,0].set_title('L3 Structure Propagation')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Interaction events over time
            if self.interaction_events:
                interaction_times = [event[0] for event in self.interaction_events]
                axes[1,1].hist(interaction_times, bins=20, alpha=0.7, color='green')
                axes[1,1].set_xlabel('Time')
                axes[1,1].set_ylabel('Interaction Count')
                axes[1,1].set_title('L3-L0/L1 Interaction Events')
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, 'No Interactions\n(As Expected)', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('L3-L0/L1 Interaction Events')
            
            plt.suptitle('TPU L3: Persistent Structures Analysis', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / "L3_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Billion-year propagation analysis
            if self.propagation_analysis['ultra_stable_count'] > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Stability factors
                axes[0].hist(self.propagation_analysis['stability_factors'], bins=15, alpha=0.7, color='purple')
                axes[0].set_xlabel('Stability Factor (Final/Initial Energy)')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Ultra-Stable Structure Stability')
                axes[0].grid(True, alpha=0.3)
                
                # Max propagation distances
                axes[1].hist(self.propagation_analysis['max_distances'], bins=15, alpha=0.7, color='orange')
                axes[1].set_xlabel('Maximum Propagation Distance')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Ultra-Stable Structure Propagation')
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle('TPU L3: Billion-Year Propagation Analysis', fontsize=16)
                plt.tight_layout()
                plt.savefig(self.output_dir / "L3_billion_year_analysis.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            print("✅ L3 plots created successfully")
            
        except Exception as e:
            print(f"❌ Error creating L3 plots: {e}")
    
    def run_l3_simulation(self):
        """Run complete L3 simulation"""
        print("🚀 Starting L3 Persistent Structures Simulation")
        print("="*60)
        
        # Step 1: Identify persistent structures from L2 fields
        self.identify_persistent_structures()
        
        # Step 2: Simulate long-term structure evolution
        self.simulate_structure_evolution()
        
        # Step 3: Analyze billion-year propagation potential
        self.analyze_billion_year_propagation()
        
        # Step 4: Save all data
        self.save_l3_data()
        
        # Step 5: Create analysis plots
        self.create_l3_plots()
        
        print("="*60)
        print("🎉 L3 Simulation completed successfully!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"🔍 Persistent structures: {len(self.persistent_locations)}")
        print(f"⏳ Mean lifetime: {np.mean(self.structure_lifetimes):.1f}")
        print(f"🌌 Ultra-stable structures: {self.propagation_analysis['ultra_stable_count']}")
        print(f"🔗 Total L0/L1 interactions: {len(self.interaction_events)}")
        print(f"📏 Max propagation: {np.max(self.structure_distances):.3f}")

if __name__ == "__main__":
    # Run L3 simulation
    l3_sim = TPU_L3_Simulation()
    l3_sim.run_l3_simulation()

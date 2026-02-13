"""
TPU Simulation L2 DEMO: Excitations and Fields from L0/L1 Foundation
This demonstrates how L2 would work with your actual foundation data
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class TPU_L2_Demo:
    def __init__(self, output_dir="L2_demo_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🌌 TPU L2 DEMO: Excitations and Fields")
        print(f"This demonstrates how L2 works with foundation data")
        print(f"Output directory: {self.output_dir}")
        
        # Create mock foundation data (similar to your actual results)
        self.create_mock_foundation_data()
        
        # L2-specific parameters
        self.field_modes = 50
        self.excitation_threshold = 0.1
        self.propagation_steps = 5000  # Reduced for demo
        
    def create_mock_foundation_data(self):
        """Create mock foundation data similar to your actual run"""
        print("📊 Creating mock foundation data (similar to your actual results)...")
        
        # Grid parameters (matching your run)
        self.Nx = self.Ny = self.Nz = 61
        self.L = 30.0
        self.dx = 2 * self.L / (self.Nx - 1)
        
        # Create mock sigma field with realistic structure
        x = y = z = np.linspace(-self.L, self.L, self.Nx)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Mock final sigma (low values like your actual run: mean σ: 0.053)
        self.substrate_config = 0.1 * np.exp(-r**2 / (2 * 15**2)) + 0.01 * np.random.randn(self.Nx, self.Ny, self.Nz)
        self.substrate_config = np.maximum(self.substrate_config, 0)
        
        # Mock thermodynamic data (similar to your actual evolution)
        self.foundation_times = np.array([0.0, 4.0, 8.0, 14.0, 20.0, 28.0])
        self.foundation_masses = np.array([32268.78, 25046.42, 16621.10, 12595.05, 12002.94, 12000.0])
        self.foundation_temps = np.array([3.411, 0.622, 0.661, 0.597, 2.804, 2.8])
        
        print(f"✅ Mock foundation data created")
        print(f"   Grid: {self.Nx}x{self.Ny}x{self.Nz}")
        print(f"   Final foundation mass: {self.foundation_masses[-1]:.1f}")
        print(f"   Foundation σ range: [{self.substrate_config.min():.3f}, {self.substrate_config.max():.3f}]")
        print(f"   Mean σ: {self.substrate_config.mean():.3f} (similar to your 0.053)")
    
    def compute_field_modes(self):
        """Extract field modes from foundation σ patterns"""
        print("🔬 Computing L2 field modes from foundation...")
        
        # 3D FFT of foundation sigma to get mode structure
        fft_substrate = np.fft.fftn(self.substrate_config)
        power_spectrum = np.abs(fft_substrate)**2
        
        # Extract dominant modes
        flat_power = power_spectrum.flatten()
        mode_indices = np.argsort(flat_power)[-self.field_modes:]
        
        # Convert back to 3D indices
        mode_coords = np.unravel_index(mode_indices, power_spectrum.shape)
        mode_amplitudes = flat_power[mode_indices]
        
        # Create field mode array
        self.field_modes_data = np.zeros((self.field_modes, 4))  # [kx, ky, kz, amplitude]
        
        for i in range(self.field_modes):
            kx = mode_coords[0][i] if mode_coords[0][i] < self.Nx//2 else mode_coords[0][i] - self.Nx
            ky = mode_coords[1][i] if mode_coords[1][i] < self.Ny//2 else mode_coords[1][i] - self.Ny  
            kz = mode_coords[2][i] if mode_coords[2][i] < self.Nz//2 else mode_coords[2][i] - self.Nz
            
            self.field_modes_data[i] = [kx, ky, kz, np.sqrt(mode_amplitudes[i])]
        
        print(f"✅ Extracted {self.field_modes} dominant field modes")
        print(f"   Strongest mode amplitude: {self.field_modes_data[-1, 3]:.3e}")
        return self.field_modes_data
    
    def compute_excitation_spectrum(self):
        """Compute excitation spectrum from foundation interactions"""
        print("⚡ Computing L2 excitation spectrum...")
        
        # Compute gradients of foundation sigma (excitation sources)
        grad_x = np.gradient(self.substrate_config, axis=0)
        grad_y = np.gradient(self.substrate_config, axis=1) 
        grad_z = np.gradient(self.substrate_config, axis=2)
        
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Identify excitation regions (high gradient areas)
        excitation_mask = grad_magnitude > self.excitation_threshold
        excitation_locations = np.argwhere(excitation_mask)
        
        # Compute excitation energies
        excitation_energies = grad_magnitude[excitation_mask]
        
        self.excitation_spectrum = {
            'locations': excitation_locations,
            'energies': excitation_energies,
            'total_excitations': len(excitation_locations),
            'mean_energy': np.mean(excitation_energies) if len(excitation_energies) > 0 else 0,
            'max_energy': np.max(excitation_energies) if len(excitation_energies) > 0 else 0
        }
        
        print(f"✅ Found {self.excitation_spectrum['total_excitations']} excitation sites")
        print(f"   Mean energy: {self.excitation_spectrum['mean_energy']:.3f}")
        print(f"   Max energy: {self.excitation_spectrum['max_energy']:.3f}")
        
        return self.excitation_spectrum
    
    def simulate_field_propagation(self):
        """Model how L2 excitations propagate through universe"""
        print("🌊 Simulating L2 field propagation...")
        
        # Initialize propagation data
        propagation_data = {
            'times': [],
            'field_energies': [],
            'propagation_distances': [],
            'interaction_events': []
        }
        
        # Simple propagation model: fields spread from excitation sites
        current_field = np.zeros_like(self.substrate_config)
        
        # Seed with excitation locations (limit for performance)
        num_seeds = min(50, len(self.excitation_spectrum['locations']))
        for i in range(num_seeds):
            loc = self.excitation_spectrum['locations'][i]
            x, y, z = loc
            energy = self.excitation_spectrum['energies'][i]
            current_field[x, y, z] = energy
        
        # Propagate over time
        dt = 0.01
        diffusion_coeff = 0.1
        
        for step in range(self.propagation_steps):
            # Diffusion-like propagation (simplified)
            laplacian = (np.roll(current_field, 1, 0) + np.roll(current_field, -1, 0) +
                        np.roll(current_field, 1, 1) + np.roll(current_field, -1, 1) +
                        np.roll(current_field, 1, 2) + np.roll(current_field, -1, 2) - 
                        6 * current_field) / self.dx**2
            
            current_field += dt * diffusion_coeff * laplacian
            
            # Rare interactions with L0/L1 (very small probability)
            interaction_prob = 1e-5  # Higher for demo
            if np.random.rand() < interaction_prob:
                # Random interaction event
                ix, iy, iz = np.random.randint(0, self.Nx, 3)
                interaction_strength = np.random.exponential(0.1)
                current_field[ix, iy, iz] *= (1 - interaction_strength)
                propagation_data['interaction_events'].append((step * dt, ix, iy, iz, interaction_strength))
            
            # Record data every 50 steps
            if step % 50 == 0:
                propagation_data['times'].append(step * dt)
                propagation_data['field_energies'].append(np.sum(current_field**2))
                
                # Compute propagation distance (RMS spread)
                x_coords, y_coords, z_coords = np.meshgrid(range(self.Nx), range(self.Ny), range(self.Nz), indexing='ij')
                total_energy = np.sum(current_field**2)
                if total_energy > 1e-10:
                    center_x = np.sum(current_field**2 * x_coords) / total_energy
                    center_y = np.sum(current_field**2 * y_coords) / total_energy
                    center_z = np.sum(current_field**2 * z_coords) / total_energy
                    
                    rms_distance = np.sqrt(np.sum(current_field**2 * 
                                                ((x_coords - center_x)**2 + 
                                                 (y_coords - center_y)**2 + 
                                                 (z_coords - center_z)**2)) / total_energy)
                    propagation_data['propagation_distances'].append(rms_distance * self.dx)
                else:
                    propagation_data['propagation_distances'].append(0)
            
            if step % 500 == 0:
                print(f"   Step {step}/{self.propagation_steps}, Field energy: {np.sum(current_field**2):.3e}")
        
        self.propagation_data = propagation_data
        self.final_field = current_field
        
        print(f"✅ Propagation simulation completed")
        print(f"   Final field energy: {propagation_data['field_energies'][-1]:.3e}")
        print(f"   Max propagation distance: {max(propagation_data['propagation_distances']):.3f}")
        print(f"   Total interaction events: {len(propagation_data['interaction_events'])}")
        
        return propagation_data
    
    def create_l2_plots(self):
        """Generate L2 analysis plots"""
        print("📊 Creating L2 analysis plots...")
        
        try:
            # Field propagation plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Field energy evolution
            axes[0,0].plot(self.propagation_data['times'], self.propagation_data['field_energies'], 'b-', linewidth=2)
            axes[0,0].set_xlabel('Time')
            axes[0,0].set_ylabel('Total Field Energy')
            axes[0,0].set_title('L2 Field Energy Evolution')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_yscale('log')
            
            # Propagation distance
            axes[0,1].plot(self.propagation_data['times'], self.propagation_data['propagation_distances'], 'r-', linewidth=2)
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('RMS Propagation Distance')
            axes[0,1].set_title('L2 Field Propagation Distance')
            axes[0,1].grid(True, alpha=0.3)
            
            # Excitation energy histogram
            if len(self.excitation_spectrum['energies']) > 0:
                axes[1,0].hist(self.excitation_spectrum['energies'], bins=20, alpha=0.7, color='green')
            axes[1,0].set_xlabel('Excitation Energy')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title('L2 Excitation Energy Distribution')
            axes[1,0].grid(True, alpha=0.3)
            
            # Field mode spectrum
            mode_magnitudes = self.field_modes_data[:, 3]
            axes[1,1].plot(range(len(mode_magnitudes)), mode_magnitudes, 'mo-', markersize=4)
            axes[1,1].set_xlabel('Mode Index')
            axes[1,1].set_ylabel('Mode Amplitude')
            axes[1,1].set_title('L2 Field Mode Spectrum')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_yscale('log')
            
            plt.suptitle('TPU L2 Demo: Field Emergence from Foundation', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / "L2_demo_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Foundation vs Final field comparison
            mid = self.Nz // 2
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Foundation sigma field
            im1 = axes[0].imshow(self.substrate_config[:, :, mid], extent=[-self.L, self.L, -self.L, self.L], 
                               origin='lower', cmap='viridis')
            axes[0].set_title('L0/L1 Foundation σ Field (z=0 slice)')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0])
            
            # Final L2 field
            im2 = axes[1].imshow(self.final_field[:, :, mid], extent=[-self.L, self.L, -self.L, self.L], 
                               origin='lower', cmap='plasma')
            axes[1].set_title('L2 Final Field Configuration (z=0 slice)')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1])
            
            plt.suptitle('TPU L2 Demo: Foundation → Field Evolution', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / "L2_demo_fields.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ L2 demo plots created successfully")
            
        except Exception as e:
            print(f"❌ Error creating L2 plots: {e}")
    
    def run_l2_demo(self):
        """Run complete L2 demonstration"""
        print("🚀 Starting L2 Excitations and Fields DEMONSTRATION")
        print("="*60)
        
        # Step 1: Compute field modes from foundation
        self.compute_field_modes()
        
        # Step 2: Compute excitation spectrum
        self.compute_excitation_spectrum()
        
        # Step 3: Simulate field propagation
        self.simulate_field_propagation()
        
        # Step 4: Create analysis plots
        self.create_l2_plots()
        
        print("="*60)
        print("🎉 L2 DEMO completed successfully!")
        print(f"📁 Results saved in: {self.output_dir}")
        print(f"🔬 Field modes: {self.field_modes}")
        print(f"⚡ Excitation sites: {self.excitation_spectrum['total_excitations']}")
        print(f"🌊 Propagation steps: {self.propagation_steps}")
        print(f"🔗 Interaction events: {len(self.propagation_data['interaction_events'])}")
        print()
        print("📋 **How to run with YOUR actual data:**")
        print("1. Copy your 'tpu_output_V56c_auto' folder to this directory")
        print("2. Run: python tpu_l2_excitations.py")
        print("3. It will automatically load your foundation data and run L2")

if __name__ == "__main__":
    # Run L2 demonstration
    l2_demo = TPU_L2_Demo()
    l2_demo.run_l2_demo()

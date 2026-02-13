import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import psutil  # For RAM monitoring
import os

# ==================== Warm-Phase V5.6c: Feedback Replication Mode ====================
# Auto-save version - saves all data automatically without prompts
# Changes from original:
# - Automatic directory creation
# - Comprehensive data saving at regular intervals
# - Progress checkpoints with state preservation
# - Error handling for file operations
# - Memory monitoring and cleanup

version = "V56c_auto"

# Create output directory
output_dir = f"tpu_output_{version}"
os.makedirs(output_dir, exist_ok=True)

# File paths
anim_filename        = os.path.join(output_dir, f"{version}_animation.mp4")
thermo_filename      = os.path.join(output_dir, f"{version}_thermo_plot.png")
slices_filename      = os.path.join(output_dir, f"{version}_slices_plot.png")
power_spec_filename  = os.path.join(output_dir, f"{version}_power_spectrum.png")
carew_w_filename     = os.path.join(output_dir, f"{version}_carew_w_plot.png")

final_sigma_filename = os.path.join(output_dir, f"{version}_final_sigma.npy")
times_npy            = os.path.join(output_dir, f"{version}_times.npy")
masses_npy           = os.path.join(output_dir, f"{version}_masses.npy")
hilberts_npy         = os.path.join(output_dir, f"{version}_hilberts.npy")
radii_npy            = os.path.join(output_dir, f"{version}_radii.npy")
temps_npy            = os.path.join(output_dir, f"{version}_temps.npy")
evap_rates_npy       = os.path.join(output_dir, f"{version}_evap_rates.npy")
k_npy                = os.path.join(output_dir, f"{version}_k.npy")
Pk_npy               = os.path.join(output_dir, f"{version}_Pk.npy")

# Checkpoint files for resuming
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Parameters
Nx = Ny = Nz = 121
L = 30.0
dx = dy = dz = 2 * L / (Nx - 1)
x = y = z = np.linspace(-L, L, Nx)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

t_max = 28.0
Nt = int(1e6)  # Safety cap

D = 0.015
alpha_coll = 0.45
alpha_evap = 1.1
epsilon4 = 0.18
beta = 0.0015
eps = 1e-3

D_c = 0.7
gamma_c = 0.20
k_c = 9.0
D_rho = 1.5

c_thresh = 0.48
sigma_sat = 35.0
max_sigma = 110.0

initial_spark_amp = 22.0
initial_spark_width = 5.0
spark_rate = 0.12  # Base for global
evap_spark_thresh = 0.01  # Evap burst trigger for local sparks
local_spark_prob = 0.05  # Higher prob for feedback

NUM_PART = 12
r_start = 26.0

sigma_0 = 50.0
T_c = 1.0
rho_c = 5.0
P_c = 10.0
alpha_t = 0.001
beta_nu = 0.4
lambda_val = 0.05
base_dt = 0.004
base_spark_width = 4.5

# Auto-save intervals
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every N steps
DATA_SAVE_INTERVAL = 1000   # Save intermediate data every N steps
PLOT_SAVE_INTERVAL = 10000  # Save plots every N steps

def save_checkpoint(step, time, sigma, C, rho, pos, vel, traj, times, masses, hilberts, radii, temps, evap_rates):
    """Save simulation state for resuming"""
    try:
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.npz")
        np.savez_compressed(checkpoint_file,
                          step=step, time=time, sigma=sigma, C=C, rho=rho,
                          pos=pos, vel=vel, traj=np.array(traj),
                          times=np.array(times), masses=np.array(masses),
                          hilberts=np.array(hilberts), radii=np.array(radii),
                          temps=np.array(temps), evap_rates=np.array(evap_rates))
        print(f"Checkpoint saved at step {step}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint at step {step}: {e}")

def save_intermediate_data(times, masses, hilberts, radii, temps, evap_rates, suffix=""):
    """Save intermediate thermodynamic data"""
    try:
        np.save(times_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(times))
        np.save(masses_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(masses))
        np.save(hilberts_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(hilberts))
        np.save(radii_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(radii))
        np.save(temps_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(temps))
        np.save(evap_rates_npy.replace('.npy', f'_intermediate{suffix}.npy'), np.array(evap_rates))
    except Exception as e:
        print(f"Warning: Failed to save intermediate data: {e}")

def save_intermediate_plots(times, masses, hilberts, radii, temps, evap_rates, suffix=""):
    """Save intermediate plots"""
    try:
        fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        ax[0].plot(times, masses, 'b.-')
        ax[0].set_ylabel('Mass / Entropy')
        ax[0].grid(True)
        ax[1].plot(times, hilberts, 'g.-')
        ax[1].set_ylabel('Hilbert Capacity')
        ax[1].grid(True)
        ax[2].plot(times, radii, 'r.-')
        ax[2].set_ylabel('Horizon Radius')
        ax[2].grid(True)
        ax[3].plot(times, temps, 'm.-', label='T')
        if len(temps) == len(radii):
            ax[3].plot(times, np.array(temps) * np.array(radii), 'c--', label='T × r')
        ax[3].set_ylabel('Temperature')
        ax[3].set_xlabel('Time')
        ax[3].legend()
        ax[3].grid(True)
        plt.tight_layout()
        plt.savefig(thermo_filename.replace('.png', f'_intermediate{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to save intermediate plots: {e}")

np.random.seed(42)
theta = np.arccos(2*np.random.rand(NUM_PART) - 1)
phi = 2*np.pi * np.random.rand(NUM_PART)
pos = np.zeros((NUM_PART, 3))
pos[:,0] = r_start * np.sin(theta) * np.cos(phi)
pos[:,1] = r_start * np.sin(theta) * np.sin(phi)
pos[:,2] = r_start * np.cos(theta)
vel = np.zeros_like(pos)
traj = [pos.copy()]

def get_field_coords(p):
    ix = (p[:,0] + L) / dx
    iy = (p[:,1] + L) / dy
    iz = (p[:,2] + L) / dz
    return np.array([ix, iy, iz])  # Shape (3, NUM_PART)

# Fields
sigma = np.zeros((Nx, Ny, Nz))
rho = np.zeros((Nx, Ny, Nz))
C = np.ones((Nx, Ny, Nz))

# Initial spark
r2 = X**2 + Y**2 + Z**2
sigma += initial_spark_amp * np.exp(-r2 / (2 * initial_spark_width**2))

# Operators
def lap3d(f):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1) +
            np.roll(f, 1, 2) + np.roll(f, -1, 2) - 6*f) / dx**2

def gradx(f): return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2*dx)
def grady(f): return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2*dy)
def gradz(f): return (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2*dz)

# Smoothed radial σ(r) for Carew-W
def compute_smoothed_sigma_r(sigma, lambda_val):
    mid = Nx // 2
    r_max = L
    num_r = 100
    r_vals = np.linspace(1e-8, r_max, num_r)
    sigma_r = np.zeros(num_r)
    for i, r in enumerate(r_vals):
        dist = np.sqrt(X**2 + Y**2 + Z**2) - r
        weights = np.exp(-dist**2 / (2 * lambda_val**2))  # Gaussian shell
        sigma_r[i] = np.sum(sigma * weights) / (np.sum(weights) + 1e-10)
    return r_vals, sigma_r

def compute_carew_w(r_vals, sigma_r, M, lambda_val):
    f_r = 1.0 - (2 * M * r_vals**2 * sigma_r) / (r_vals**3 + 2 * M * lambda_val**2)
    dr = r_vals[1] - r_vals[0]  # Assume uniform
    f_prime = np.gradient(f_r, dr)
    f_double_prime = np.gradient(f_prime, dr)
    R = -f_double_prime - (4 / r_vals) * f_prime + (2 / r_vals**2) * (1 - f_r)
    K = f_double_prime**2 + (4 / r_vals**2) * f_prime**2 + (4 / r_vals**4) * (1 - f_r)**2
    return f_r, R, K

# Power Spectrum Computation
def compute_power_spectrum(field, dx):
    # 3D FFT
    fft_field = np.fft.fftn(field)
    ps = np.abs(fft_field)**2
    
    # Frequencies
    kx = np.fft.fftfreq(Nx, d=dx)
    ky = np.fft.fftfreq(Ny, d=dx)
    kz = np.fft.fftfreq(Nz, d=dx)
    KX, KY, KZ = np.meshgrid(kx, ky, kz)
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    # Radial average
    k_bins = np.linspace(k.min(), k.max(), 50)
    dk = k_bins[1] - k_bins[0]
    Pk = np.zeros(len(k_bins) - 1)
    for i in range(len(Pk)):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        Pk[i] = np.mean(ps[mask]) if np.any(mask) else 0
    
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    return k_centers, Pk

# Diagnostics
times = [0.0]
masses = []
hilberts = []
radii = []
temps = []
evap_rates = []

snapshot_times = [0.0, 4.0, 8.0, 14.0, 20.0, 28.0]
snapshots = []
anim_frames = []

# Initial diagnostics
gx = gradx(sigma)
gy = grady(sigma)
gz = gradz(sigma)
grad2 = gx**2 + gy**2 + gz**2
R_surf = grad2 / (sigma + eps)
R_bulk = -lap3d(sigma)
C_target = np.exp(-k_c * R_surf)

mass = np.sum(sigma) * dx**3
hilbert = np.sum(sigma * C) * dx**3
area = np.sum(C > c_thresh) * dx**3
radius = (area * 3/(4*np.pi))**(1/3) if area > 0 else 0.0

horizon_mask = (C < c_thresh + 0.15) & (C > c_thresh - 0.15)
T = np.mean(np.sqrt(R_surf[horizon_mask])) if np.any(horizon_mask) else 0.0

masses.append(mass)
hilberts.append(hilbert)
radii.append(radius)
temps.append(T)
evap_rates.append(0.0)
snapshots.append((0.0, sigma.copy(), C.copy(), rho.copy(), pos.copy()))
anim_frames.append((sigma.copy(), C.copy(), rho.copy(), pos.copy()))

prev_mass = mass
prev_time = 0.0

print(f"Starting TPU simulation {version}")
print(f"Output directory: {output_dir}")
print(f"Grid size: {Nx}x{Ny}x{Nz}, Domain: [{-L}, {L}]^3")
print(f"Target time: {t_max}, Max steps: {Nt}")
print(f"Auto-save intervals - Checkpoint: {CHECKPOINT_INTERVAL}, Data: {DATA_SAVE_INTERVAL}, Plots: {PLOT_SAVE_INTERVAL}")

# Evolution
time = 0.0
step = 0
while time < t_max and step < Nt:
    step += 1
    
    lap_sigma = lap3d(sigma)
    hyperlap = lap3d(lap_sigma)
    gx = gradx(sigma)
    gy = grady(sigma)
    gz = gradz(sigma)
    grad2 = gx**2 + gy**2 + gz**2
    R_surf = grad2 / (sigma + eps)
    R_bulk = -lap_sigma
    C_target = np.exp(-k_c * R_surf)
    
    noise = np.random.randn(*sigma.shape)
    
    diff_factor = 1.0 - 0.75 * C
    collapse_factor = (C ** 2.0) / (1.0 + (sigma / sigma_sat)**2)
    evap_factor = (1.0 - C) ** 2.0
    
    # Proxies for σ(T,ρ,P,t)
    mean_T = np.mean(R_surf)  # Curvature proxy for T
    mean_rho = np.mean(rho)
    mean_P = np.mean(np.sqrt(grad2))  # Gradient proxy for P
    mean_sigma = np.mean(sigma)
    
    # σ multiplier (for collapse/evap only)
    sigma_multiplier = sigma_0 * np.exp(-mean_T / T_c) * (1 - mean_rho / rho_c) * (1 - mean_P / P_c) * (1 + alpha_t * time)
    
    # ν_s for variable dt and spark modulation
    nu_s = mean_sigma ** beta_nu + 1e-6
    current_dt = base_dt / nu_s
    
    # Modulated spark_prob (base * current_dt * nu_s)
    current_spark_prob = spark_rate * current_dt * nu_s
    
    # Modulated spark_width (~ λ)
    current_spark_width = base_spark_width * lambda_val
    
    evap_contribution = sigma_multiplier * alpha_evap * evap_factor * R_surf * sigma
    collapse_contribution = sigma_multiplier * alpha_coll * collapse_factor * R_bulk * sigma
    
    dsigma_dt = (D * diff_factor * lap_sigma
                 + collapse_contribution
                 - evap_contribution
                 - epsilon4 * hyperlap
                 + beta * noise)
    
    sigma += current_dt * dsigma_dt
    sigma = np.minimum(sigma, max_sigma)
    sigma = np.maximum(sigma, 0.0)
    sigma[np.isnan(sigma)] = 0.0
    
    drho_dt = D_rho * lap3d(rho) + evap_contribution
    rho += current_dt * drho_dt
    rho = np.maximum(rho, 0.0)
    
    dC_dt = D_c * lap3d(C) + gamma_c * (C_target - C)
    C += current_dt * dC_dt
    C = np.clip(C, 0.0, 1.0)
    
    if np.random.rand() < current_spark_prob:
        cx = np.random.uniform(-L*0.7, L*0.7)
        cy = np.random.uniform(-L*0.7, L*0.7)
        cz = np.random.uniform(-L*0.7, L*0.7)
        dist2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
        this_amp = np.random.uniform(10.0, 18.0)
        sigma += this_amp * np.exp(-dist2 / (2 * current_spark_width**2))
        sigma = np.minimum(sigma, max_sigma)
    
    # Local evap-triggered sparks for feedback
    high_evap_mask = evap_contribution > evap_spark_thresh
    if np.any(high_evap_mask):
        indices = np.argwhere(high_evap_mask)
        for idx in indices[:5]:  # Limit to 5 per step
            if np.random.rand() < local_spark_prob:
                cx, cy, cz = X[tuple(idx)], Y[tuple(idx)], Z[tuple(idx)]
                dist2 = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2
                this_amp = np.random.uniform(5.0, 12.0)  # Smaller for local
                sigma += this_amp * np.exp(-dist2 / (2 * current_spark_width**2))
                sigma = np.minimum(sigma, max_sigma)
    
    # Geodesics
    grad_Rx = gradx(R_bulk)
    grad_Ry = grady(R_bulk)
    grad_Rz = gradz(R_bulk)
    coords = get_field_coords(pos)
    acc_x = -ndimage.map_coordinates(grad_Rx, coords, order=1, mode='constant', cval=0.0)
    acc_y = -ndimage.map_coordinates(grad_Ry, coords, order=1, mode='constant', cval=0.0)
    acc_z = -ndimage.map_coordinates(grad_Rz, coords, order=1, mode='constant', cval=0.0)
    acc = np.stack((acc_x, acc_y, acc_z), axis=1)
    vel += current_dt * acc
    pos += current_dt * vel
    if step % 10 == 0:
        traj.append(pos.copy())
    
    if step % 30 == 0 and len(anim_frames) < 500:
        anim_frames.append((sigma.copy(), C.copy(), rho.copy(), pos.copy()))
    
    # Auto-save checkpoints
    if step % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(step, time, sigma, C, rho, pos, vel, traj, times, masses, hilberts, radii, temps, evap_rates)
    
    # Auto-save intermediate data
    if step % DATA_SAVE_INTERVAL == 0:
        save_intermediate_data(times, masses, hilberts, radii, temps, evap_rates, f"_step_{step}")
    
    # Auto-save intermediate plots
    if step % PLOT_SAVE_INTERVAL == 0 and len(times) > 1:
        save_intermediate_plots(times, masses, hilberts, radii, temps, evap_rates, f"_step_{step}")
    
    if step % 1000 == 0:
        mem_gb = psutil.Process().memory_info().rss / 1e9
        print(f"Step {step:6d}, time {time:6.2f}, σ̄ {mean_sigma:5.1f}, dt {current_dt:.6f}, frames {len(anim_frames):3d}, RAM {mem_gb:.1f} GB")
    
    if (len(snapshots) < len(snapshot_times) and
        abs(time - snapshot_times[len(snapshots)]) < current_dt/2) or time >= t_max:
        mass = np.sum(sigma) * dx**3
        hilbert = np.sum(sigma * C) * dx**3
        area = np.sum(C > c_thresh) * dx**3
        radius = (area * 3/(4*np.pi))**(1/3) if area > 0 else 0.0
        
        horizon_mask = (C < c_thresh + 0.15) & (C > c_thresh - 0.15)
        T = np.mean(np.sqrt(R_surf[horizon_mask])) if np.any(horizon_mask) else temps[-1]
        
        dmass_dt = (mass - prev_mass) / (time - prev_time + 1e-12)
        evap_rates.append(-dmass_dt)
        
        masses.append(mass)
        hilberts.append(hilbert)
        radii.append(radius)
        temps.append(T)
        times.append(time)
        snapshots.append((time, sigma.copy(), C.copy(), rho.copy(), pos.copy()))
        
        # Add Carew-W
        r_vals, sigma_r = compute_smoothed_sigma_r(sigma, lambda_val)
        f_r, R_curv, K_curv = compute_carew_w(r_vals, sigma_r, mass, lambda_val)
        
        prev_mass = mass
        prev_time = time
        
        print(f"Snapshot saved at time {time:.2f}: M={mass:.2f}, R={radius:.2f}, T={T:.3f}")
    
    time += current_dt

print(f"\nSimulation completed!")
print(f"Final time: {time:.2f}, Total steps: {step}")
print(f"Final mean σ: {np.mean(sigma):.2f}")

# Compute Power Spectrum of final sigma
print("Computing power spectrum...")
final_sigma = snapshots[-1][1]  # Last snapshot's sigma
k, Pk = compute_power_spectrum(final_sigma - final_sigma.mean(), dx)  # Subtract mean for fluctuations

# Save all final data
print("Saving final data...")
try:
    np.save(final_sigma_filename, final_sigma)
    np.save(times_npy, np.array(times))
    np.save(masses_npy, np.array(masses))
    np.save(hilberts_npy, np.array(hilberts))
    np.save(radii_npy, np.array(radii))
    np.save(temps_npy, np.array(temps))
    np.save(evap_rates_npy, np.array(evap_rates))
    np.save(k_npy, k)
    np.save(Pk_npy, Pk)
    print("✓ All numpy data saved successfully")
except Exception as e:
    print(f"✗ Error saving numpy data: {e}")

# Generate final plots
print("Generating final plots...")

# Thermodynamics plot
try:
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    ax[0].plot(times, masses, 'b.-', linewidth=1.5, markersize=4)
    ax[0].set_ylabel('Mass / Entropy', fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f'TPU Simulation {version} - Thermodynamic Evolution', fontsize=14)
    
    ax[1].plot(times, hilberts, 'g.-', linewidth=1.5, markersize=4)
    ax[1].set_ylabel('Hilbert Capacity', fontsize=12)
    ax[1].grid(True, alpha=0.3)
    
    ax[2].plot(times, radii, 'r.-', linewidth=1.5, markersize=4)
    ax[2].set_ylabel('Horizon Radius', fontsize=12)
    ax[2].grid(True, alpha=0.3)
    
    ax[3].plot(times, temps, 'm.-', linewidth=1.5, markersize=4, label='Temperature')
    if len(temps) == len(radii):
        ax[3].plot(times, np.array(temps) * np.array(radii), 'c--', linewidth=1.5, label='T × r')
    ax[3].set_ylabel('Temperature', fontsize=12)
    ax[3].set_xlabel('Time', fontsize=12)
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(thermo_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Thermodynamics plot saved")
except Exception as e:
    print(f"✗ Error saving thermodynamics plot: {e}")

# Power spectrum plot
try:
    plt.figure(figsize=(10, 6))
    plt.loglog(k[k>0], Pk[k>0], 'b-', linewidth=2, label='P(k)')
    plt.xlabel('Wavenumber k', fontsize=12)
    plt.ylabel('Power P(k)', fontsize=12)
    plt.title(f'Power Spectrum of Final σ Field - {version}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(power_spec_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Power spectrum plot saved")
except Exception as e:
    print(f"✗ Error saving power spectrum plot: {e}")

# Carew-W geometry plot
try:
    r_vals, sigma_r = compute_smoothed_sigma_r(final_sigma, lambda_val)
    final_mass = masses[-1]
    f_r, R_curv, K_curv = compute_carew_w(r_vals, sigma_r, final_mass, lambda_val)
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    ax[0,0].plot(r_vals, sigma_r, 'b-', linewidth=2)
    ax[0,0].set_xlabel('Radius r')
    ax[0,0].set_ylabel('σ(r)')
    ax[0,0].set_title('Radial Sampling Density')
    ax[0,0].grid(True, alpha=0.3)
    
    ax[0,1].plot(r_vals, f_r, 'r-', linewidth=2)
    ax[0,1].set_xlabel('Radius r')
    ax[0,1].set_ylabel('f(r)')
    ax[0,1].set_title('Lapse Function')
    ax[0,1].grid(True, alpha=0.3)
    ax[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax[1,0].plot(r_vals, R_curv, 'g-', linewidth=2)
    ax[1,0].set_xlabel('Radius r')
    ax[1,0].set_ylabel('R(r)')
    ax[1,0].set_title('Ricci Scalar')
    ax[1,0].grid(True, alpha=0.3)
    
    ax[1,1].plot(r_vals, K_curv, 'm-', linewidth=2)
    ax[1,1].set_xlabel('Radius r')
    ax[1,1].set_ylabel('K(r)')
    ax[1,1].set_title('Kretschmann Scalar')
    ax[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Carew W-Geometry Analysis - {version}', fontsize=16)
    plt.tight_layout()
    plt.savefig(carew_w_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Carew W-geometry plot saved")
except Exception as e:
    print(f"✗ Error saving Carew W-geometry plot: {e}")

# Field slices plot
try:
    mid = Nx // 2
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # σ field
    im1 = ax[0,0].imshow(final_sigma[mid,:,:], extent=[-L,L,-L,L], origin='lower', cmap='viridis')
    ax[0,0].set_title('σ Field (z=0 slice)')
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=ax[0,0])
    
    # C field
    final_C = snapshots[-1][2]
    im2 = ax[0,1].imshow(final_C[mid,:,:], extent=[-L,L,-L,L], origin='lower', cmap='plasma')
    ax[0,1].set_title('Coherence Field C (z=0 slice)')
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=ax[0,1])
    
    # ρ field
    final_rho = snapshots[-1][3]
    im3 = ax[1,0].imshow(final_rho[mid,:,:], extent=[-L,L,-L,L], origin='lower', cmap='hot')
    ax[1,0].set_title('ρ Field (z=0 slice)')
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=ax[1,0])
    
    # Particle trajectories
    traj_array = np.array(traj)
    for i in range(NUM_PART):
        ax[1,1].plot(traj_array[:,i,0], traj_array[:,i,1], alpha=0.7, linewidth=1)
    ax[1,1].set_title('Particle Trajectories (x-y projection)')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].grid(True, alpha=0.3)
    ax[1,1].set_xlim(-L, L)
    ax[1,1].set_ylim(-L, L)
    
    plt.suptitle(f'Final Field Configuration - {version}', fontsize=16)
    plt.tight_layout()
    plt.savefig(slices_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Field slices plot saved")
except Exception as e:
    print(f"✗ Error saving field slices plot: {e}")

# Save simulation summary
try:

    summary_file = os.path.join(output_dir, f"{version}_summary.txt") 
    with open(summary_file, 'w', encoding='utf-8') as f:
    #summary_file = os.path.join(output_dir, f"{version}_summary.txt")
    #with open(summary_file, 'w') as f:
        f.write(f"TPU Simulation Summary - {version}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Grid size: {Nx}x{Ny}x{Nz}\n")
        f.write(f"Domain: [{-L}, {L}]^3\n")
        f.write(f"Total simulation time: {time:.2f}\n")
        f.write(f"Total steps: {step}\n")
        f.write(f"Final mean σ: {np.mean(final_sigma):.3f}\n")
        f.write(f"Final mass: {masses[-1]:.3f}\n")
        f.write(f"Final radius: {radii[-1]:.3f}\n")
        f.write(f"Final temperature: {temps[-1]:.3f}\n")
        f.write(f"Number of snapshots: {len(snapshots)}\n")
        f.write(f"Number of animation frames: {len(anim_frames)}\n")
        f.write(f"Number of particle trajectories: {len(traj)}\n")
        f.write(f"\nFiles generated:\n")
        f.write(f"- Final σ field: {os.path.basename(final_sigma_filename)}\n")
        f.write(f"- Thermodynamic data: times, masses, hilberts, radii, temps, evap_rates\n")
        f.write(f"- Power spectrum: k, Pk\n")
        f.write(f"- Plots: thermodynamics, power spectrum, Carew W-geometry, field slices\n")
        f.write(f"- Checkpoints: {len(os.listdir(checkpoint_dir))} files\n")
    print("✓ Simulation summary saved")
except Exception as e:
    print(f"✗ Error saving summary: {e}")

print(f"\n" + "="*60)
print(f"TPU SIMULATION {version} COMPLETED SUCCESSFULLY")
print(f"="*60)
print(f"Output directory: {output_dir}")
print(f"Total files generated: {len(os.listdir(output_dir))} + {len(os.listdir(checkpoint_dir))} checkpoints")
print(f"Final simulation time: {time:.2f}")
print(f"Final mean σ: {np.mean(final_sigma):.3f}")
print(f"Memory usage: {psutil.Process().memory_info().rss / 1e9:.1f} GB")
print("="*60)
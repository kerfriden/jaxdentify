"""
Test MALA on 2D Banana distribution.

Banana distribution:
  x ~ Normal(0, 1)
  y | x ~ Normal(b * x^2, sigma^2)
"""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import new modular API
from optimization.inference import run_mala


def make_banana_logpi(b=2.0, sigma=0.12):
    """
    Banana distribution (2D):
      x ~ Normal(0, 1)
      y | x ~ Normal(b * x^2, sigma^2)
    
    Args:
        b: Curvature parameter
        sigma: Conditional standard deviation
    
    Returns:
        logpi: Callable theta -> log-density
    """
    LOG2PI = jnp.log(2.0 * jnp.pi)
    inv_sigma2 = 1.0 / (sigma * sigma)

    def logpi(theta):
        x, y = theta[0], theta[1]
        lp_x = -0.5 * (LOG2PI + x * x)  # log N(x; 0, 1)
        r = y - b * x * x
        lp_y = -0.5 * (LOG2PI + jnp.log(sigma * sigma) + r * r * inv_sigma2)  # log N(y; b*x^2, sigma^2)
        return lp_x + lp_y

    return logpi


def plot_banana_with_samples(samples, b=2.0, sigma=0.12, filename="plots/mala_banana.png", 
                            n_func_evals=0, n_grad_evals=0):
    """
    Plot banana distribution contours with MALA samples.
    
    Args:
        samples: [N, 2] array of samples
        b: Banana curvature
        sigma: Banana conditional std
        filename: Output path
        n_func_evals: Number of function evaluations
        n_grad_evals: Number of gradient evaluations
    """
    # Create output directory
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Setup grid for contours with better aspect ratio
    x_range = np.linspace(-3.5, 3.5, 200)
    y_range = np.linspace(-0.5, 6.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute log-density on grid
    logpi = make_banana_logpi(b, sigma)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = float(logpi(theta))
    
    # Convert to density (exp) for better visualization
    Z_density = np.exp(Z)
    
    # Create plot with proper aspect ratio
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot contours
    levels = 20
    contour = ax.contour(X, Y, Z_density, levels=levels, colors='gray', alpha=0.5, linewidths=0.8)
    contourf = ax.contourf(X, Y, Z_density, levels=levels, cmap='Blues', alpha=0.4)
    
    # Plot samples
    samples_np = np.array(samples)
    label_text = f'MALA samples (N={len(samples)})'
    if n_func_evals > 0 or n_grad_evals > 0:
        label_text += f'\nf-evals: {n_func_evals:,}, ∇f-evals: {n_grad_evals:,}'
    ax.scatter(samples_np[:, 0], samples_np[:, 1], 
               c='red', s=15, alpha=0.6, edgecolors='darkred', linewidths=0.3,
               label=label_text)
    
    # Plot theoretical mean curve: y = b * x^2
    x_curve = np.linspace(-3, 3, 100)
    y_curve = b * x_curve**2
    ax.plot(x_curve, y_curve, 'k--', linewidth=2.5, label=f'Mean: y = {b}x²')
    
    # Styling
    ax.set_xlabel('x', fontsize=13, fontweight='bold')
    ax.set_ylabel('y', fontsize=13, fontweight='bold')
    ax.set_title(f'MALA on Banana Distribution (b={b}, σ={sigma})', fontsize=15, fontweight='bold')
    ax.legend(loc='upper center', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.5, 6.5)
    
    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {filename}")
    plt.close()


def compute_diagnostics(samples, b=2.0, sigma=0.12):
    """Compute and print MALA diagnostics."""
    samples_np = np.array(samples)
    
    print("\n" + "="*60)
    print("MALA DIAGNOSTICS")
    print("="*60)
    
    # Sample statistics
    mean_x = np.mean(samples_np[:, 0])
    mean_y = np.mean(samples_np[:, 1])
    std_x = np.std(samples_np[:, 0])
    std_y = np.std(samples_np[:, 1])
    
    print(f"\nSample statistics:")
    print(f"  E[x] = {mean_x:.4f}  (true: 0.0)")
    print(f"  E[y] = {mean_y:.4f}  (true: {b:.4f} for x~N(0,1))")
    print(f"  std[x] = {std_x:.4f}  (true: 1.0)")
    print(f"  std[y] = {std_y:.4f}")
    
    # Effective sample size (rough estimate from autocorrelation)
    def autocorr_lag1(x):
        x_centered = x - np.mean(x)
        c0 = np.dot(x_centered, x_centered) / len(x)
        c1 = np.dot(x_centered[:-1], x_centered[1:]) / (len(x) - 1)
        return c1 / c0 if c0 > 1e-10 else 0.0
    
    rho_x = autocorr_lag1(samples_np[:, 0])
    rho_y = autocorr_lag1(samples_np[:, 1])
    
    ess_x = len(samples) * (1 - rho_x) / (1 + rho_x) if rho_x < 0.99 else len(samples) / 100
    ess_y = len(samples) * (1 - rho_y) / (1 + rho_y) if rho_y < 0.99 else len(samples) / 100
    
    print(f"\nAutocorrelation (lag-1):")
    print(f"  ρ[x] = {rho_x:.4f}")
    print(f"  ρ[y] = {rho_y:.4f}")
    print(f"\nEffective sample size (rough):")
    print(f"  ESS[x] ≈ {ess_x:.0f} / {len(samples)}")
    print(f"  ESS[y] ≈ {ess_y:.0f} / {len(samples)}")
    
    print("="*60 + "\n")


def main():
    """Run MALA on banana distribution and visualize results."""
    
    print("="*60)
    print("TESTING MALA ON BANANA DISTRIBUTION")
    print("="*60)
    
    # Configuration
    b = 2.0
    sigma = 0.12
    
    # MALA parameters
    eps = 0.05           # Step size
    n_steps = 5000       # Total steps
    burn = 1000          # Burn-in
    thin = 2             # Thinning
    
    # Random seed
    key = random.PRNGKey(42)
    
    # Initial state (start near origin)
    theta_init = jnp.array([0.0, 0.0])
    
    print(f"\nBanana parameters: b={b}, σ={sigma}")
    print(f"MALA settings: eps={eps}, n_steps={n_steps}, burn={burn}, thin={thin}")
    print(f"Initial state: {theta_init}")
    
    # Create log-density
    logpi = make_banana_logpi(b, sigma)
    
    print("\n" + "-"*60)
    print("Running MALA...")
    print("-"*60)
    
    # Run MALA using new API
    samples, acc_rate = run_mala(
        key=key,
        theta_init=theta_init,
        eps=eps,
        n_steps=n_steps,
        burn=burn,
        thin=thin,
        logpi=logpi,
        use_fast=True
    )
    
    n_kept = len(samples)
    print(f"\n✓ MALA completed!")
    print(f"  Acceptance rate: {acc_rate:.2%}")
    print(f"  Samples kept: {n_kept} (after burn={burn}, thin={thin})")
    
    # Diagnostics
    compute_diagnostics(samples, b, sigma)
    
    # Plot
    print("Generating plot...")
    # MALA: ~2 gradient evals per step (current + proposed), 1 func eval per step
    n_grad_evals = n_steps * 2
    n_func_evals = n_steps * 2  # logpi evaluated at current and proposed
    plot_banana_with_samples(samples, b, sigma, filename="plots/mala_banana.png",
                            n_func_evals=n_func_evals, n_grad_evals=n_grad_evals)
    
    print("\n" + "="*60)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()

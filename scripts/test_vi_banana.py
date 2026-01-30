"""
Test Variational Inference (Gaussian and Normalizing Flow) on 2D Banana distribution.
"""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from optimization.vi_flow import fit_gaussian_vi, sample_gaussian_vi, make_flow_vi


def make_banana_logpi(b=2.0, sigma=0.12):
    """Banana distribution."""
    LOG2PI = jnp.log(2.0 * jnp.pi)
    sigma_sq = sigma * sigma
    inv_sigma2 = 1.0 / sigma_sq

    def logpi(theta):
        x = theta[..., 0]  # Works with both [2] and [N, 2]
        y = theta[..., 1]
        lp_x = -0.5 * (LOG2PI + x * x)
        r = y - b * x * x
        lp_y = -0.5 * (LOG2PI + jnp.log(sigma_sq) + r * r * inv_sigma2)
        return lp_x + lp_y

    return logpi


def plot_banana_with_samples(samples, b=2.0, sigma=0.12, filename="plots/vi_banana.png", 
                            title_prefix="VI", n_func_evals=0, n_grad_evals=0):
    """Plot banana distribution with VI samples."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    x_range = np.linspace(-3.5, 3.5, 200)
    y_range = np.linspace(-0.5, 6.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    logpi = make_banana_logpi(b, sigma)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = jnp.array([X[i, j], Y[i, j]])
            Z[i, j] = float(logpi(theta))
    
    Z_density = np.exp(Z)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    levels = 20
    ax.contour(X, Y, Z_density, levels=levels, colors='gray', alpha=0.5, linewidths=0.8)
    contourf = ax.contourf(X, Y, Z_density, levels=levels, cmap='Blues', alpha=0.4)
    
    samples_np = np.array(samples)
    label_text = f'{title_prefix} samples (N={len(samples)})'
    if n_func_evals > 0 or n_grad_evals > 0:
        label_text += f'\nf-evals: {n_func_evals:,}, ∇f-evals: {n_grad_evals:,}'
    ax.scatter(samples_np[:, 0], samples_np[:, 1], 
               c='purple', s=15, alpha=0.6, edgecolors='darkviolet', linewidths=0.3,
               label=label_text)
    
    x_curve = np.linspace(-3, 3, 100)
    y_curve = b * x_curve**2
    ax.plot(x_curve, y_curve, 'k--', linewidth=2.5, label=f'Mean: y = {b}x²')
    
    ax.set_xlabel('x', fontsize=13, fontweight='bold')
    ax.set_ylabel('y', fontsize=13, fontweight='bold')
    ax.set_title(f'{title_prefix} on Banana Distribution (b={b}, sigma={sigma})', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper center', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.5, 6.5)
    
    cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {filename}")
    plt.close()


def compute_diagnostics(samples, b=2.0, sigma=0.12, method="VI"):
    """Compute diagnostics for VI samples."""
    samples_np = np.array(samples)
    
    print("\n" + "="*60)
    print(f"{method} DIAGNOSTICS")
    print("="*60)
    
    mean_x = np.mean(samples_np[:, 0])
    mean_y = np.mean(samples_np[:, 1])
    std_x = np.std(samples_np[:, 0])
    std_y = np.std(samples_np[:, 1])
    
    print(f"\nSample statistics:")
    print(f"  E[x] = {mean_x:.4f}  (true: 0.0)")
    print(f"  E[y] = {mean_y:.4f}  (true: {b:.4f} for x~N(0,1))")
    print(f"  std[x] = {std_x:.4f}  (true: 1.0)")
    print(f"  std[y] = {std_y:.4f}")
    
    print("="*60 + "\n")


def test_gaussian_vi():
    """Test Gaussian VI on banana."""
    print("="*60)
    print("TESTING GAUSSIAN VI ON BANANA DISTRIBUTION")
    print("="*60)
    
    b = 2.0
    sigma = 0.12
    d = 2
    
    # VI parameters
    n_iters = 2000
    n_samples_elbo = 20
    lr = 0.001  # Reduced learning rate for stability
    n_samples_final = 2000
    
    key = random.PRNGKey(42)
    
    print(f"\nBanana parameters: b={b}, sigma={sigma}")
    print(f"Gaussian VI settings: n_iters={n_iters}, n_samples={n_samples_elbo}, lr={lr}")
    
    logpi = make_banana_logpi(b, sigma)
    
    print("\n" + "-"*60)
    print("Fitting Gaussian VI...")
    print("-"*60 + "\n")
    
    key, k_fit = random.split(key)
    mu, log_sigma, elbo_history = fit_gaussian_vi(
        logpi,
        d,
        k_fit,
        cov="mean-field",
        n_iters=n_iters,
        n_samples=n_samples_elbo,
        lr=lr,
        verbose=True,
    )
    
    print(f"\n✓ Gaussian VI fitted!")
    print(f"  Final ELBO: {elbo_history[-1]:.4f}")
    print(f"  Fitted mu: {mu}")
    print(f"  Fitted sigma: {np.exp(log_sigma)}")
    
    # Sample from fitted distribution
    key, k_sample = random.split(key)
    samples = sample_gaussian_vi(mu, log_sigma, k_sample, n_samples_final)
    
    compute_diagnostics(samples, b, sigma, method="Gaussian VI")
    
    # Plot
    print("Generating plot...")
    # Gaussian VI: gradient of ELBO computed n_iters times with n_samples_elbo samples each
    n_grad_evals = n_iters * n_samples_elbo
    n_func_evals = n_iters * n_samples_elbo
    plot_banana_with_samples(samples, b, sigma, filename="plots/gaussian_vi_banana.png",
                            title_prefix="Gaussian VI",
                            n_func_evals=n_func_evals, n_grad_evals=n_grad_evals)
    
    print("\n" + "="*60)
    print("✓ GAUSSIAN VI TEST COMPLETED")
    print("="*60)


def test_flow_vi():
    """Test Normalizing Flow VI on banana."""
    print("\n" + "="*60)
    print("TESTING NORMALIZING FLOW VI ON BANANA DISTRIBUTION")
    print("="*60)
    
    b = 2.0
    sigma = 0.12
    d = 2
    
    # Flow parameters (RealNVP robust architecture)
    n_layers = 12  # Reduced from 24 for 2D (24 is for d=10)
    hidden_dim = 64  # Moderate for 2D
    s_cap = 1.5
    n_iters = 3000
    n_samples_elbo = 128
    lr = 0.002
    n_samples_final = 2000
    
    key = random.PRNGKey(43)
    
    print(f"RealNVP Flow settings: n_layers={n_layers}, hidden_dim={hidden_dim}, s_cap={s_cap}")
    print(f"  n_iters={n_iters}, n_samples={n_samples_elbo}, lr={lr}")
    
    logpi = make_banana_logpi(b, sigma)
    
    print("\n" + "-"*60)
    print("Creating and fitting RealNVP Normalizing Flow...")
    print("-"*60 + "\n")
    
    flow_forward, flow_inverse, fit_flow, sample_flow = make_flow_vi(
        logpi, d, n_layers=n_layers, hidden_dim=hidden_dim, s_cap=s_cap, use_random_perm=True
    )
    
    key, k_fit = random.split(key)
    flow_params, elbo_history = fit_flow(
        k_fit, n_iters=n_iters, n_samples=n_samples_elbo, lr=lr, verbose=True
    )
    
    print(f"\n✓ Flow VI fitted!")
    print(f"  Final ELBO: {elbo_history[-1]:.4f}")
    
    # Sample from fitted flow
    key, k_sample = random.split(key)
    samples = sample_flow(flow_params, k_sample, n_samples_final)
    
    compute_diagnostics(samples, b, sigma, method="Flow VI")
    
    # Plot
    print("Generating plot...")
    # Flow VI: gradient computed n_iters times with n_samples_elbo samples each
    # Each sample requires d*n_layers gradient evaluations through the flow
    n_grad_evals = n_iters * n_samples_elbo
    n_func_evals = n_iters * n_samples_elbo
    plot_banana_with_samples(samples, b, sigma, filename="plots/flow_vi_banana.png",
                            title_prefix="Flow VI",
                            n_func_evals=n_func_evals, n_grad_evals=n_grad_evals)
    
    print("\n" + "="*60)
    print("✓ FLOW VI TEST COMPLETED")
    print("="*60)


def main():
    """Run both VI tests."""
    test_gaussian_vi()
    print("\n" * 2)
    test_flow_vi()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Both Gaussian VI and Normalizing Flow VI tested on banana distribution.")
    print("Normalizing Flow should better capture the curved banana shape.")
    print("Check plots in plots/ directory for visual comparison.")
    print("="*60)


if __name__ == "__main__":
    main()

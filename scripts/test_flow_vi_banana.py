"""
Test Normalizing Flow Variational Inference on 2D Banana distribution.
"""

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from optimization.vi_flow import make_flow_vi


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


def plot_banana_with_samples(samples, b=2.0, sigma=0.12, filename="plots/flow_vi_banana.png", 
                            title_prefix="Flow VI", n_func_evals=0, n_grad_evals=0):
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


def compute_diagnostics(samples, b=2.0, sigma=0.12):
    """Compute diagnostics for VI samples."""
    samples_np = np.array(samples)
    
    print("\n" + "="*60)
    print("FLOW VI DIAGNOSTICS")
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


def main():
    """Test Normalizing Flow VI on banana."""
    print("="*60)
    print("TESTING NORMALIZING FLOW VI ON BANANA DISTRIBUTION")
    print("="*60)
    
    b = 2.0
    sigma = 0.12
    d = 2
    
    # Flow parameters (batched RealNVP architecture)
    n_layers = 12  # 12 coupling layers for 2D
    hidden_dim = 64  # Hidden dimension for MLP conditioners
    s_cap = 2.2  # Scale clipping (higher allows more expressive transformations)
    n_iters = 3000
    n_samples_elbo = 256  # More samples for stable ELBO estimation
    lr = 0.002
    n_samples_final = 2000
    
    key = random.PRNGKey(43)
    
    print(f"\nBanana parameters: b={b}, sigma={sigma}")
    print(f"RealNVP Flow settings: n_layers={n_layers}, hidden_dim={hidden_dim}, s_cap={s_cap}")
    print(f"  n_iters={n_iters}, n_samples={n_samples_elbo}, lr={lr}")
    
    logpi = make_banana_logpi(b, sigma)
    
    print("\n" + "-"*60)
    print("Creating and fitting RealNVP Normalizing Flow...")
    print("-"*60 + "\n")
    
    flow_forward, flow_inverse, fit_flow, sample_flow = make_flow_vi(
        logpi, d, n_layers=n_layers, hidden_dim=hidden_dim, s_cap=s_cap, use_random_perm=False
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
    
    compute_diagnostics(samples, b, sigma)
    
    # Plot
    print("Generating plot...")
    n_grad_evals = n_iters * n_samples_elbo
    n_func_evals = n_iters * n_samples_elbo
    plot_banana_with_samples(samples, b, sigma, filename="plots/flow_vi_banana.png",
                            title_prefix="Flow VI",
                            n_func_evals=n_func_evals, n_grad_evals=n_grad_evals)
    
    print("\n" + "="*60)
    print("✓ FLOW VI TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

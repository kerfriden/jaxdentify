"""Test preconditioned MALA on 2D Banana distribution.

Banana distribution:
  x ~ Normal(0, 1)
  y | x ~ Normal(b * x^2, sigma^2)

This script computes a MAP-Hessian preconditioner and runs MALA with
constant covariance proposals. It also generates a contour + samples plot.

Run:
  python scripts/test_mala_precond_banana.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimization.inference import run_mala
from optimization.preconditioning import map_hessian_preconditioner, sample_preconditioner_gaussian


def make_banana_logpi(b: float = 2.0, sigma: float = 0.12):
    """2D banana distribution log-density."""
    log2pi = jnp.log(2.0 * jnp.pi)
    inv_sigma2 = 1.0 / (sigma * sigma)

    def logpi(theta: jnp.ndarray) -> jnp.ndarray:
        x, y = theta[0], theta[1]
        lp_x = -0.5 * (log2pi + x * x)
        r = y - b * x * x
        lp_y = -0.5 * (log2pi + jnp.log(sigma * sigma) + r * r * inv_sigma2)
        return lp_x + lp_y

    return logpi


def plot_banana_with_samples(
    samples: np.ndarray,
    *,
    b: float = 2.0,
    sigma: float = 0.12,
    filename: str = "plots/mala_precond_banana.png",
):
    """Plot banana distribution contours with samples and save to filename."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    x_range = np.linspace(-3.5, 3.5, 250)
    y_range = np.linspace(-0.5, 6.5, 250)
    X, Y = np.meshgrid(x_range, y_range)

    logpi = make_banana_logpi(b=b, sigma=sigma)
    grid = jnp.stack([jnp.asarray(X).ravel(), jnp.asarray(Y).ravel()], axis=1)
    Z = jax.vmap(logpi)(grid).reshape(X.shape)
    Z = np.asarray(jax.device_get(Z))

    # Stable density for visualization
    Z_density = np.exp(Z - np.max(Z))

    fig, ax = plt.subplots(figsize=(9, 7))
    levels = 25
    ax.contour(X, Y, Z_density, levels=levels, colors="gray", alpha=0.45, linewidths=0.8)
    cf = ax.contourf(X, Y, Z_density, levels=levels, cmap="Blues", alpha=0.45)

    ax.scatter(samples[:, 0], samples[:, 1], c="red", s=14, alpha=0.6, linewidths=0.0)

    x_curve = np.linspace(-3.0, 3.0, 200)
    ax.plot(x_curve, b * x_curve**2, "k--", linewidth=2.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Preconditioned MALA on Banana (b={b}, sigma={sigma})")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-0.5, 6.5)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {filename}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("TESTING PRECONDITIONED MALA ON BANANA DISTRIBUTION")
    print("=" * 60)

    b = 2.0
    sigma = 0.12

    # MALA parameters
    eps = 0.25
    n_steps = 5000
    burn = 1000
    thin = 2

    key = random.PRNGKey(42)
    theta_init = jnp.array([0.0, 0.0])

    logpi = make_banana_logpi(b=b, sigma=sigma)

    print(f"\nBanana parameters: b={b}, sigma={sigma}")
    print(f"MALA settings: eps={eps}, n_steps={n_steps}, burn={burn}, thin={thin}")

    print("\nComputing MAP + Hessian preconditioner...")
    theta_map, precond, info = map_hessian_preconditioner(
        logpi,
        theta_init,
        mode="full",
        jitter=1e-6,
        map_max_iter=50,
        map_gtol=1e-8,
        map_print_every=10,
    )
    print("MAP theta:", np.asarray(theta_map))
    print("Hessian eig range:", info.get("hessian_min_eig"), info.get("hessian_max_eig"))

    # Draw from the MAP-Hessian Gaussian approximation for sanity-checking
    key, k_gauss = random.split(key)
    gauss_samples = sample_preconditioner_gaussian(k_gauss, theta_map, precond, n=2000)
    gauss_np = np.asarray(jax.device_get(gauss_samples))
    print(f"Gaussian(precond) samples: mean={gauss_np.mean(axis=0)}, std={gauss_np.std(axis=0)}")
    plot_banana_with_samples(
        gauss_np,
        b=b,
        sigma=sigma,
        filename="plots/gaussian_from_precond_banana.png",
    )

    print("\nRunning preconditioned MALA...")
    samples, acc_rate = run_mala(
        key=key,
        theta_init=theta_init,
        eps=eps,
        n_steps=n_steps,
        burn=burn,
        thin=thin,
        logpi=logpi,
        precond=precond,
        use_fast=False,
    )

    samples_np = np.asarray(jax.device_get(samples))
    print(f"\nCompleted. Acceptance rate: {float(acc_rate):.2%}")
    print(f"Samples kept: {len(samples_np)}")

    print("\nGenerating plot...")
    plot_banana_with_samples(samples_np, b=b, sigma=sigma, filename="plots/mala_precond_banana.png")

    print("=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()


def test_preconditioned_mala_banana_saves_plot(tmp_path):
    """Pytest smoke test: preconditioned MALA runs and saves a banana plot."""
    b = 2.0
    sigma = 0.12
    logpi = make_banana_logpi(b=b, sigma=sigma)

    key = random.PRNGKey(0)
    theta0 = jnp.array([0.0, 0.0])

    theta_map, precond, info = map_hessian_preconditioner(
        logpi,
        theta0,
        mode="full",
        jitter=1e-6,
        map_max_iter=25,
        map_gtol=1e-8,
        map_print_every=None,
    )

    assert jnp.all(jnp.isfinite(theta_map))
    assert float(jnp.linalg.norm(theta_map)) < 1e-2
    assert "cov" in precond and "chol" in precond and "prec" in precond and "logdet" in precond
    assert np.isfinite(float(info["logpi_map"]))

    # Also ensure we can draw from the Gaussian approximation
    key, k_gauss = random.split(key)
    gauss_samples = sample_preconditioner_gaussian(k_gauss, theta_map, precond, n=256)
    gauss_np = np.asarray(jax.device_get(gauss_samples))
    assert gauss_np.shape == (256, 2)
    assert np.all(np.isfinite(gauss_np))

    samples, acc_rate = run_mala(
        key=key,
        theta_init=theta0,
        eps=0.25,
        n_steps=5000,
        burn=300,
        thin=2,
        logpi=logpi,
        precond=precond,
        use_fast=False,
    )

    samples_np = np.asarray(jax.device_get(samples))
    assert samples_np.ndim == 2 and samples_np.shape[1] == 2
    assert np.all(np.isfinite(samples_np))
    assert 0.0 <= float(acc_rate) <= 1.0

    out = tmp_path / "mala_precond_banana.png"
    plot_banana_with_samples(samples_np, b=b, sigma=sigma, filename=str(out))
    assert out.exists() and out.stat().st_size > 0

    repo_plot = Path("plots") / "mala_precond_banana.png"
    repo_plot.parent.mkdir(parents=True, exist_ok=True)
    repo_plot.write_bytes(out.read_bytes())
    assert repo_plot.exists() and repo_plot.stat().st_size > 0

    mean_x = float(np.mean(samples_np[:, 0]))
    assert abs(mean_x) < 0.5

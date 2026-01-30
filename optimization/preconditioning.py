"""Preconditioning utilities (MAP + Hessian-based).

Provides:
- MAP estimation via BFGS on U(theta) = -logpi(theta)
- Hessian-at-MAP Gaussian preconditioner for MALA/HMC etc.

The preconditioner is represented as a dict:
    {
        "cov":   covariance matrix C (d,d) (typically approx posterior cov),
        "chol":  Cholesky L of C (lower-triangular),
        "prec":  precision matrix C^{-1} (d,d),
        "logdet": log(det(C)),
        "mean":  optional mean vector (d,) (for Laplace-style Gaussian approx)
    }

All functions operate in unconstrained theta space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Literal, Mapping, Tuple

import jax
import jax.numpy as jnp

from optimization.optimizers import adamw, bfgs
from optimization.vi_flow import fit_gaussian_vi, fit_gaussian_vi_full
from jax.scipy.linalg import solve_triangular


@dataclass(frozen=True)
class LaplaceGaussian(Mapping[str, jnp.ndarray]):
    """Gaussian posterior approximation in theta-space.

    Behaves like a read-only dict with keys {mean, cov, chol, prec, logdet}, and
    also provides `.sample(key, n)` convenience.

    Notes:
      - Returned by `laplace_gaussian(...)` (MAP+Hessian / Laplace)
            - Also used by `gaussian_vi_gaussian(...)` (Gaussian VI; diag or full-cov)
    """

    mean: jnp.ndarray
    cov: jnp.ndarray
    chol: jnp.ndarray
    prec: jnp.ndarray
    logdet: jnp.ndarray

    def sample(self, key: jax.Array, *, n: int) -> jnp.ndarray:
        return sample_preconditioner_gaussian(key, None, self.as_dict(), n=n)

    def as_dict(self) -> Dict[str, jnp.ndarray]:
        return {
            "mean": self.mean,
            "cov": self.cov,
            "chol": self.chol,
            "prec": self.prec,
            "logdet": self.logdet,
        }

    # Mapping interface (allows passing to code expecting precond["cov"], etc.)
    def __getitem__(self, k: str) -> jnp.ndarray:
        if k == "mean":
            return self.mean
        if k == "cov":
            return self.cov
        if k == "chol":
            return self.chol
        if k == "prec":
            return self.prec
        if k == "logdet":
            return self.logdet
        raise KeyError(k)

    def __iter__(self) -> Iterator[str]:
        return iter(("mean", "cov", "chol", "prec", "logdet"))

    def __len__(self) -> int:
        return 5


def sample_preconditioner_gaussian(
    key: jax.Array,
    mean: jnp.ndarray | None,
    precond: Dict[str, jnp.ndarray],
    *,
    n: int,
) -> jnp.ndarray:
    """Draw samples from N(mean, precond['cov']).

    If mean is None, uses precond['mean'].

    Args:
        key: JAX PRNG key
        mean: [d] mean vector (typically theta_map). If None, uses precond['mean'].
        precond: dict with keys {cov, chol, prec, logdet}
        n: number of samples

    Returns:
        samples: [n, d]
    """
    if mean is None:
        if "mean" not in precond:
            raise ValueError("mean is None but precond has no 'mean'")
        mean = precond["mean"]
    mean = jnp.asarray(mean)
    L = precond["chol"]
    eps = jax.random.normal(key, shape=(int(n), mean.shape[0]))
    return mean[None, :] + eps @ L.T


def find_map(
    logpi: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    method: Literal["bfgs", "adamw"] = "bfgs",
    max_iter: int | None = None,
    gtol: float = 1e-7,
    n_display: int | None = None,
    # AdamW knobs (only used if method="adamw")
    adamw_lr: float = 5e-2,
    adamw_weight_decay: float = 0.0,
    adamw_clip_norm: float | None = None,
    adamw_lr_schedule: Literal["constant", "cosine"] = "cosine",
    adamw_lr_min: float = 1e-4,
    adamw_grad_noise: float = 0.0,
    adamw_seed: int = 0,
) -> Tuple[jnp.ndarray, float, dict]:
    """Find MAP estimate by minimizing U(theta) = -logpi(theta)."""

    def U(theta):
        return -logpi(theta)

    theta0 = jnp.asarray(theta0)
    if method == "bfgs":
        theta_map, U_map, info = bfgs(U, theta0, max_iter=max_iter, gtol=gtol, n_display=n_display)
        info = {**info, "method": "bfgs"}
    elif method == "adamw":
        theta_map, U_map, info = adamw(
            U,
            theta0,
            max_iter=int(max_iter) if max_iter is not None else 5000,
            lr=float(adamw_lr),
            gtol=float(gtol),
            n_display=n_display,
            weight_decay=float(adamw_weight_decay),
            clip_norm=adamw_clip_norm,
            lr_schedule=str(adamw_lr_schedule),
            lr_min=float(adamw_lr_min),
            grad_noise=float(adamw_grad_noise),
            seed=int(adamw_seed),
        )
        info = {**info, "method": "adamw"}
    else:
        raise ValueError("method must be 'bfgs' or 'adamw'")

    logpi_map = float(-U_map)
    return theta_map, logpi_map, info


def preconditioner_from_hessian(
    H: jnp.ndarray,
    *,
    jitter: float = 1e-6,
    mode: Literal["full", "diag"] = "full",
) -> Dict[str, jnp.ndarray]:
    """Build a Gaussian preconditioner from a Hessian of U=-logpi.

    Uses C \approx (H + jitter I)^{-1}.

    If mode="diag", uses only the diagonal of H.
    """
    H = jnp.asarray(H)
    d = int(H.shape[0])

    if mode == "diag":
        diag = jnp.clip(jnp.diag(H), a_min=jitter)
        C = jnp.diag(1.0 / diag)
    elif mode == "full":
        H_reg = H + jitter * jnp.eye(d, dtype=H.dtype)
        C = jnp.linalg.inv(H_reg)
    else:
        raise ValueError("mode must be 'full' or 'diag'")

    # Symmetrize numerically (important before Cholesky)
    C = 0.5 * (C + C.T)
    L = jnp.linalg.cholesky(C + jitter * jnp.eye(d, dtype=C.dtype))
    # Precision from Cholesky for stability
    # prec = C^{-1} = (L L^T)^{-1}
    I = jnp.eye(d, dtype=C.dtype)
    prec = jax.scipy.linalg.cho_solve((L, True), I)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return {"cov": C, "chol": L, "prec": prec, "logdet": logdet}


def map_hessian_preconditioner(
    logpi: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    map_method: Literal["bfgs", "adamw"] = "bfgs",
    jitter: float = 1e-6,
    mode: Literal["full", "diag"] = "full",
    map_max_iter: int | None = None,
    map_gtol: float = 1e-7,
    map_print_every: int | None = None,
    map_display: int | None = None,
    # AdamW knobs (only used if map_method="adamw")
    map_adamw_lr: float = 5e-2,
    map_adamw_weight_decay: float = 0.0,
    map_adamw_clip_norm: float | None = None,
    map_adamw_lr_schedule: Literal["constant", "cosine"] = "cosine",
    map_adamw_lr_min: float = 1e-4,
    map_adamw_grad_noise: float = 0.0,
    map_adamw_seed: int = 0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], dict]:
    """Compute MAP and Hessian-based preconditioner.

    Returns:
      theta_map, precond, info
    where info contains MAP optimizer info plus Hessian stats.
    """
    if map_display is not None:
        # Backward compatibility: `map_display` was the old name.
        # Keep it working, but prefer the clearer `map_print_every`.
        import warnings

        if map_print_every is not None:
            raise ValueError("Provide only one of map_print_every or map_display")
        warnings.warn(
            "map_display is deprecated; use map_print_every instead",
            DeprecationWarning,
            stacklevel=2,
        )
        map_print_every = map_display

    theta_map, logpi_map, info_map = find_map(
        logpi,
        theta0,
        method=map_method,
        max_iter=map_max_iter,
        gtol=map_gtol,
        n_display=map_print_every,
        adamw_lr=map_adamw_lr,
        adamw_weight_decay=map_adamw_weight_decay,
        adamw_clip_norm=map_adamw_clip_norm,
        adamw_lr_schedule=map_adamw_lr_schedule,
        adamw_lr_min=map_adamw_lr_min,
        adamw_grad_noise=map_adamw_grad_noise,
        adamw_seed=map_adamw_seed,
    )

    def U(theta):
        return -logpi(theta)

    H = jax.hessian(U)(theta_map)
    precond = preconditioner_from_hessian(H, jitter=jitter, mode=mode)
    # Storing the mean makes this dict usable as a Laplace Gaussian object.
    precond = {**precond, "mean": theta_map}

    info = {
        "map": info_map,
        "logpi_map": logpi_map,
        "hessian_min_eig": float(jnp.linalg.eigvalsh(0.5 * (H + H.T))[0]),
        "hessian_max_eig": float(jnp.linalg.eigvalsh(0.5 * (H + H.T))[-1]),
    }
    return theta_map, precond, info


def gaussian_vi_preconditioner(
    logpi: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    covariance: Literal["full", "diag"] = "full",
    n_iters: int = 500,
    n_samples: int = 4,
    lr: float = 1e-2,
    verbose: bool = True,
    jitter: float = 1e-12,
    mu0: jnp.ndarray | None = None,
    log_sigma0: jnp.ndarray | None = None,
    chol0: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], dict]:
    """Gaussian VI as a Gaussian preconditioner.

    Fits q(theta) by maximizing the ELBO and returns a preconditioner dict
    compatible with MALA/HMC preconditioning.

    - covariance="full": q(theta)=N(mu, L L^T) (captures correlations)
    - covariance="diag": q(theta)=N(mu, diag(sigma^2)) (mean-field, faster)

    This provides an alternative to MAP+Hessian (Laplace) when you prefer not to
    compute Hessians or when MAP optimization is difficult.
    """
    theta0 = jnp.asarray(theta0)
    d = int(theta0.shape[0])

    key = jax.random.PRNGKey(0)
    if mu0 is None:
        mu0 = theta0

    cov_mode = str(covariance).lower()
    if cov_mode not in {"full", "diag"}:
        raise ValueError(f"covariance must be 'full' or 'diag', got {covariance!r}")

    if cov_mode == "diag":
        mu, log_sigma, elbo_hist = fit_gaussian_vi(
            logpi,
            d,
            key,
            n_iters=int(n_iters),
            n_samples=int(n_samples),
            lr=float(lr),
            verbose=bool(verbose),
            mu0=mu0,
            log_sigma0=log_sigma0,
        )

        sigma2 = jnp.exp(2.0 * log_sigma)
        sigma2 = jnp.clip(sigma2, a_min=float(jitter))
        sigma = jnp.sqrt(sigma2)

        C = jnp.diag(sigma2)
        L = jnp.diag(sigma)
        prec = jnp.diag(1.0 / sigma2)
        logdet = jnp.sum(jnp.log(sigma2))
        precond = {"cov": C, "chol": L, "prec": prec, "logdet": logdet, "mean": mu}
    else:
        mu, L, elbo_hist = fit_gaussian_vi_full(
            logpi,
            d,
            key,
            n_iters=int(n_iters),
            n_samples=int(n_samples),
            lr=float(lr),
            verbose=bool(verbose),
            mu0=mu0,
            chol0=chol0,
            jitter=float(max(float(jitter), 1e-12)),
        )

        C = L @ L.T
        Linv = solve_triangular(L, jnp.eye(d, dtype=L.dtype), lower=True)
        prec = Linv.T @ Linv
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L) + 1e-32))
        precond = {"cov": C, "chol": L, "prec": prec, "logdet": logdet, "mean": mu}

    info = {
        "method": f"gaussian_vi_{cov_mode}",
        "n_iters": int(n_iters),
        "n_samples": int(n_samples),
        "lr": float(lr),
        "elbo_last": (float(elbo_hist[-1]) if getattr(elbo_hist, "size", 0) else None),
    }
    return mu, precond, info


def gaussian_vi_gaussian(
    logpi: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    covariance: Literal["full", "diag"] = "full",
    n_iters: int = 500,
    n_samples: int = 4,
    lr: float = 1e-2,
    verbose: bool = True,
    jitter: float = 1e-12,
    mu0: jnp.ndarray | None = None,
    log_sigma0: jnp.ndarray | None = None,
    chol0: jnp.ndarray | None = None,
) -> Tuple[LaplaceGaussian, dict]:
    """Gaussian posterior approximation via Gaussian VI.

    Returns (g, info) where g behaves like a dict with keys
    {mean, cov, chol, prec, logdet} and supports g.sample(key, n=...).
    """
    _, precond, info = gaussian_vi_preconditioner(
        logpi,
        theta0,
        covariance=covariance,
        n_iters=n_iters,
        n_samples=n_samples,
        lr=lr,
        verbose=verbose,
        jitter=jitter,
        mu0=mu0,
        log_sigma0=log_sigma0,
        chol0=chol0,
    )
    g = LaplaceGaussian(
        mean=precond["mean"],
        cov=precond["cov"],
        chol=precond["chol"],
        prec=precond["prec"],
        logdet=precond["logdet"],
    )
    return g, info


# Example (Laplace / Gaussian posterior approx):
#
#   import jax
#   import jax.numpy as jnp
#   from optimization.preconditioning import laplace_gaussian
#
#   def logpi(theta):
#       return -0.5 * jnp.sum(theta**2)   # toy standard normal
#
#   theta0 = jnp.zeros(3)
#   g, info = laplace_gaussian(logpi, theta0, map_method="bfgs", map_print_every=10)
#   theta_samples = g.sample(jax.random.key(0), n=1000)
#   # g behaves like a dict too: g["mean"], g["cov"], g["prec"], ...

def laplace_gaussian(
    logpi: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    map_method: Literal["bfgs", "adamw"] = "bfgs",
    jitter: float = 1e-6,
    mode: Literal["full", "diag"] = "full",
    map_max_iter: int | None = None,
    map_gtol: float = 1e-7,
    map_print_every: int | None = None,
    map_display: int | None = None,
    # AdamW knobs (only used if map_method="adamw")
    map_adamw_lr: float = 5e-2,
    map_adamw_weight_decay: float = 0.0,
    map_adamw_clip_norm: float | None = None,
    map_adamw_lr_schedule: Literal["constant", "cosine"] = "cosine",
    map_adamw_lr_min: float = 1e-4,
    map_adamw_grad_noise: float = 0.0,
    map_adamw_seed: int = 0,
) -> Tuple[LaplaceGaussian, dict]:
    """Compute Laplace (Gaussian) posterior approximation in theta-space.

    Returns:
      precond, info

    Where the returned object behaves like a dict containing
    {mean, cov, chol, prec, logdet} and also provides `.sample(key, n=...)`.

    Note: this is equivalent to map_hessian_preconditioner(...), but returns
    a single Gaussian object (precond) instead of (theta_map, precond).
    """
    theta_map, precond, info = map_hessian_preconditioner(
        logpi,
        theta0,
        map_method=map_method,
        jitter=jitter,
        mode=mode,
        map_max_iter=map_max_iter,
        map_gtol=map_gtol,
        map_print_every=map_print_every,
        map_display=map_display,
        map_adamw_lr=map_adamw_lr,
        map_adamw_weight_decay=map_adamw_weight_decay,
        map_adamw_clip_norm=map_adamw_clip_norm,
        map_adamw_lr_schedule=map_adamw_lr_schedule,
        map_adamw_lr_min=map_adamw_lr_min,
        map_adamw_grad_noise=map_adamw_grad_noise,
        map_adamw_seed=map_adamw_seed,
    )
    # theta_map is already stored in precond['mean'] for convenience.
    _ = theta_map
    g = LaplaceGaussian(
        mean=precond["mean"],
        cov=precond["cov"],
        chol=precond["chol"],
        prec=precond["prec"],
        logdet=precond["logdet"],
    )
    return g, info

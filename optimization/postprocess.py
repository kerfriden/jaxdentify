"""
Post-processing utilities for MCMC/VI samples.

Handles conversion from theta samples to:
- Physical parameter space via projection
- Summary statistics (mean, percentiles)
- Posterior predictive distributions
"""

import jax
import jax.numpy as jnp
from optimization.parameter_mappings import project_theta, to_params


def theta_to_params_samples(space, thetas):
    """Convert theta samples to physical-parameter samples.

    This is a pure *mapping* (not “MAP”): it maps each unconstrained theta sample
    to a parameter dict, by projecting into bounds and applying inverse scaling.

    Args:
        space: Parameter space from build_param_space
        thetas: [N, d] array of theta samples

    Returns:
        params_samples: PyTree of parameter arrays with leading dimension N
    """
    proj = jax.vmap(lambda th: project_theta(space, th))(thetas)
    params_samples = jax.vmap(lambda th: to_params(space, th))(proj)
    return params_samples


def map_to_params(space, thetas):
    """Backwards-compatible alias for theta_to_params_samples(...)."""
    return theta_to_params_samples(space, thetas)


def posterior_param_summary(thetas, space):
    """
    Compute posterior mean and credible intervals in parameter space.
    
    Args:
        thetas: [N, d] array of theta samples
        space: Parameter space from build_param_space
    
    Returns:
        summary: Dict {param_name: (mean, p05, p95)}
                Each entry can be scalar or array matching parameter shape
    """
    params_list = theta_to_params_samples(space, thetas)  # PyTree with shape [N, ...]
    
    def _summarize(x):
        mean = jnp.mean(x, axis=0)
        p05 = jnp.percentile(x, 5.0, axis=0, method="linear")
        p95 = jnp.percentile(x, 95.0, axis=0, method="linear")
        return (mean, p05, p95)
    
    return jax.tree.map(_summarize, params_list)


def posterior_predictive(thetas, space, pred_func):
    """
    Generate posterior predictive distribution using forward model.
    
    For each sample:
        1. Convert theta to params
        2. Evaluate pred_func(params) -> [T]
        3. Aggregate across samples
    
    Args:
        thetas: [N, d] array of theta samples
        space: Parameter space from build_param_space
        pred_func: Callable params_dict -> [T] (forward model)
    
    Returns:
        mean: [T] predictive mean
        p05: [T] 5th percentile
        p95: [T] 95th percentile
        preds: [N, T] all predictions
    """
    params_list = theta_to_params_samples(space, thetas)
    
    # Vectorize forward model over parameter samples
    preds = jax.vmap(lambda p: pred_func(p))(params_list)  # [N, T]
    
    mean = jnp.mean(preds, axis=0)
    p05 = jnp.percentile(preds, 5.0, axis=0, method="linear")
    p95 = jnp.percentile(preds, 95.0, axis=0, method="linear")
    
    return mean, p05, p95, preds

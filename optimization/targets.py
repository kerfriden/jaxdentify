"""
Module for constructing log-density targets logpi(theta).

Handles 3 cases:
1. User provides logpi(theta) directly -> return as-is
2. User provides loglik_theta(theta) -> add Gaussian prior
3. User provides loglik_params(params) + space -> compose with projection

All downstream code works only with logpi(theta).
"""

import jax.numpy as jnp
from optimization.parameter_mappings import project_theta, to_params


def unit_gaussian_logprior(theta):
    """Standard Gaussian prior on raw unconstrained theta: N(0, I)."""
    return -0.5 * jnp.dot(theta, theta)


def add_prior(loglik_theta, logprior=None):
    """
    Combine log-likelihood with log-prior to create posterior.
    
    Args:
        loglik_theta: Callable theta -> log-likelihood
        logprior: Callable theta -> log-prior (default: unit Gaussian)
    
    Returns:
        logpost: Callable theta -> log-posterior
    """
    if logprior is None:
        logprior = unit_gaussian_logprior
    
    def logpost(theta):
        return loglik_theta(theta) + logprior(theta)
    
    return logpost


def make_loglik_theta(space, loglik_params):
    """
    Convert loglik_params(params) to loglik_theta(theta) using space projection.
    
    Args:
        space: Parameter space from build_param_space
        loglik_params: Callable params_dict -> log-likelihood
    
    Returns:
        loglik_theta: Callable theta -> log-likelihood
    """
    def loglik_theta(theta):
        theta_proj = project_theta(space, theta)
        params = to_params(space, theta_proj)
        return loglik_params(params)
    
    return loglik_theta


def as_logpi(logpi=None, loglik_theta=None, space=None, loglik_params=None, logprior=None):
    """
    Unified interface to construct logpi(theta) from various inputs.
    
    Priority order:
    1. If logpi is provided -> return it directly
    2. If loglik_theta is provided -> add prior and return
    3. If loglik_params + space are provided -> compose projection + add prior
    
    Args:
        logpi: Direct log-density function theta -> R (highest priority)
        loglik_theta: Log-likelihood function theta -> R
        space: Parameter space (required if loglik_params is used)
        loglik_params: Log-likelihood function params_dict -> R
        logprior: Prior function theta -> R (default: unit Gaussian)
    
    Returns:
        logpi: Callable theta -> R (unnormalized log-density)
    
    Raises:
        ValueError: If insufficient arguments provided
    """
    # Case 1: Direct logpi provided
    if logpi is not None:
        return logpi
    
    # Case 2: loglik_theta provided
    if loglik_theta is not None:
        return add_prior(loglik_theta, logprior)
    
    # Case 3: loglik_params + space provided
    if loglik_params is not None:
        if space is None:
            raise ValueError("space is required when loglik_params is provided")
        loglik_theta = make_loglik_theta(space, loglik_params)
        return add_prior(loglik_theta, logprior)
    
    raise ValueError(
        "Must provide one of: logpi, loglik_theta, or (loglik_params + space)"
    )

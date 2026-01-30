"""
DEPRECATED: This file is kept for backwards compatibility only.

The code has been reorganized into modular components:
- targets.py: Construction of log-density targets
- mcmc_mala.py: Pure MALA implementations
- postprocess.py: Mapping and summary statistics
- inference.py: High-level API

Please update imports:
    from optimization.postprocess import theta_to_params_samples, posterior_param_summary, posterior_predictive
  from optimization.mcmc_mala import make_mala, make_mala_fast
  from optimization.targets import as_logpi
  from optimization.inference import run_mala
"""

# Re-export for backwards compatibility
from optimization.postprocess import (
        theta_to_params_samples,
    map_to_params,
    posterior_param_summary,
    posterior_predictive
)

from optimization.mcmc_mala import (
    make_mala,
    make_mala_fast
)

from optimization.targets import (
    unit_gaussian_logprior,
    add_prior,
    make_loglik_theta,
    as_logpi
)

from optimization.inference import run_mala


def make_mala_unit_gaussian_prior(loglik):
    """
    DEPRECATED: Use targets.as_logpi(loglik_theta=loglik) instead.
    
    For backwards compatibility, wraps loglik with unit Gaussian prior.
    """
    import warnings
    warnings.warn(
        "make_mala_unit_gaussian_prior is deprecated. "
        "Use: run_mala(..., loglik_theta=loglik) or "
        "targets.as_logpi(loglik_theta=loglik)",
        DeprecationWarning,
        stacklevel=2
    )
    
    logpost = add_prior(loglik, unit_gaussian_logprior)
    _, run_chain = make_mala_fast(logpost)
    return run_chain
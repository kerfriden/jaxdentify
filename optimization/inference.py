"""
High-level inference API (facade).

Provides unified interface run_mala(...) that accepts:
- Direct logpi(theta)
- loglik_theta(theta) + prior
- loglik_params(params) + space + prior

Automatically dispatches to targets.as_logpi() then runs appropriate algorithm.
"""

from optimization.targets import as_logpi


def run_mala(
    key,
    theta_init,
    eps=1e-2,
    n_steps=1000,
    burn=100,
    thin=1,
    *,
    logpi=None,
    loglik_theta=None,
    space=None,
    loglik_params=None,
    logprior=None,
    use_fast=True,
    precond=None,
    use_precond=None,
):
    """
    Run MALA sampling with unified interface.
    
    Automatically constructs logpi(theta) from provided inputs, then runs MALA.
    
    Args:
        key: JAX PRNG key
        theta_init: Initial state [d]
        eps: MALA step size
        n_steps: Total number of steps
        burn: Burn-in steps to discard
        thin: Thinning interval
        
        # Target specification (provide ONE of):
        logpi: Direct log-density theta -> R
        loglik_theta: Log-likelihood theta -> R (will add prior)
        loglik_params: Log-likelihood params_dict -> R (requires space)
        
        # Optional:
        space: Parameter space (required if loglik_params used)
        logprior: Prior theta -> R (default: unit Gaussian)
        use_fast: Use cached-gradient version (default: True)
    
    Returns:
        samples: [N_keep, d] post burn/thin
        acc_rate: Acceptance rate over kept samples
    
    Examples:
        # Direct logpi
        samples, acc = run_mala(key, theta0, eps=0.01, logpi=my_logpi)
        
        # Log-likelihood + default prior
        samples, acc = run_mala(key, theta0, eps=0.01, loglik_theta=my_loglik)
        
        # Params-based + projection
        samples, acc = run_mala(
            key, theta0, eps=0.01,
            space=param_space,
            loglik_params=my_loglik_params
        )
    """
    # Construct unified logpi
    target = as_logpi(
        logpi=logpi,
        loglik_theta=loglik_theta,
        space=space,
        loglik_params=loglik_params,
        logprior=logprior
    )
    
    # Select algorithm implementation
    # If precond is provided, use it by default (no need to also set use_precond).
    if use_precond is None:
        use_precond = precond is not None
    elif use_precond is False and precond is not None:
        raise ValueError("Got precond but use_precond=False; either set use_precond=None/True or pass precond=None")

    if use_precond:
        if precond is None:
            raise ValueError("use_precond=True requires precond (dict with cov/chol/prec/logdet)")
        from optimization.mcmc_mala import make_mala_precond
        _, run_chain = make_mala_precond(target, precond)
    else:
        if use_fast:
            from optimization.mcmc_mala import make_mala_fast
            _, run_chain = make_mala_fast(target)
        else:
            from optimization.mcmc_mala import make_mala
            _, run_chain = make_mala(target)
    
    # Run chain
    samples, acc_rate = run_chain(key, theta_init, eps, n_steps, burn, thin)
    
    return samples, acc_rate


def run_hmc(
    key,
    theta_init,
    eps=1e-2,
    L=10,
    n_steps=1000,
    burn=100,
    thin=1,
    *,
    logpi=None,
    loglik_theta=None,
    space=None,
    loglik_params=None,
    logprior=None,
    precond=None,
    use_precond=None,
):
    """Run HMC sampling with unified interface.

    Automatically constructs logpi(theta) from provided inputs, then runs HMC.

    Args:
        key: JAX PRNG key
        theta_init: Initial state [d]
        eps: Leapfrog step size
        L: Number of leapfrog steps per HMC step
        n_steps: Total number of HMC steps
        burn: Burn-in steps to discard
        thin: Thinning interval

        # Target specification (provide ONE of):
        logpi: Direct log-density theta -> R
        loglik_theta: Log-likelihood theta -> R (will add prior)
        loglik_params: Log-likelihood params_dict -> R (requires space)

        # Optional:
        space: Parameter space (required if loglik_params used)
        logprior: Prior theta -> R (default: unit Gaussian)
        use_precond: Use constant mass matrix from precond
        precond: dict with keys {cov, chol, prec, logdet} (required if use_precond)

    Returns:
        samples: [N_keep, d] post burn/thin
        acc_rate: Acceptance rate over kept samples
    """
    target = as_logpi(
        logpi=logpi,
        loglik_theta=loglik_theta,
        space=space,
        loglik_params=loglik_params,
        logprior=logprior,
    )

    if use_precond is None:
        use_precond = precond is not None
    elif use_precond is False and precond is not None:
        raise ValueError("Got precond but use_precond=False; either set use_precond=None/True or pass precond=None")

    if use_precond:
        if precond is None:
            raise ValueError("use_precond=True requires precond (dict with cov/chol/prec/logdet)")
        from optimization.mcmc_hmc import make_hmc_mass
        _, run_chain = make_hmc_mass(target, precond)
    else:
        from optimization.mcmc_hmc import make_hmc
        _, run_chain = make_hmc(target)

    samples, acc_rate = run_chain(
        key,
        theta_init,
        eps=eps,
        L=L,
        n_steps=n_steps,
        burn=burn,
        thin=thin,
    )

    return samples, acc_rate

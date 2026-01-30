"""
Pure MALA (Metropolis-Adjusted Langevin Algorithm) implementations.

All functions work only with logpi(theta), no dependency on space/projections.
This makes it easy to add HMC, whitening, VI, etc. later.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial


def _log_q(x, y, g_y, eps):
    """
    Log-density of Langevin proposal q(x | y) = N(y + 0.5*eps^2*g_y, eps^2*I).
    
    Includes full normalization constant (even though it often cancels in MH ratio).
    """
    mean = y + 0.5 * (eps**2) * g_y
    diff = x - mean
    d = diff.shape[0]
    return -0.5 * (d * jnp.log(2.0 * jnp.pi * (eps**2)) + jnp.dot(diff, diff) / (eps**2))


def make_mala(logpi):
    """
    Standard MALA implementation (recomputes gradient each step).
    
    Args:
        logpi: Callable theta -> R (unnormalized log-density)
    
    Returns:
        mala_step: Callable (key, theta, eps) -> (theta_new, accept_bool)
        run_chain: Callable (key, theta_init, eps, n_steps, burn, thin) -> (samples, acc_rate)
    """
    grad_logpi = jax.grad(logpi)

    @jax.jit
    def mala_step(key, theta, eps):
        # Propose
        k1, k2 = random.split(key)
        g = grad_logpi(theta)
        mean = theta + 0.5 * (eps**2) * g
        prop = mean + eps * random.normal(k1, shape=theta.shape)

        # MH accept
        g_prop = grad_logpi(prop)
        log_alpha = (logpi(prop) - logpi(theta)
                     + _log_q(theta, prop, g_prop, eps)
                     - _log_q(prop, theta, g, eps))
        accept = jnp.log(random.uniform(k2)) < log_alpha
        theta_new = jnp.where(accept, prop, theta)
        return theta_new, accept

    @partial(jax.jit, static_argnames=("n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=1e-2, n_steps=1000, burn=100, thin=1):
        """
        Run MALA chain.
        
        Args:
            key: JAX PRNG key
            theta_init: Initial state [d]
            eps: Step size
            n_steps: Total number of steps
            burn: Burn-in steps to discard
            thin: Thinning interval
        
        Returns:
            samples: [N_keep, d] post burn/thin
            acc_rate: Acceptance rate over kept samples
        """
        keys = random.split(key, n_steps)

        def body(theta, k):
            theta_new, acc = mala_step(k, theta, eps)
            return theta_new, (theta_new, acc)

        _, (thetas, accepts) = lax.scan(body, theta_init, keys)

        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)
        return thetas_kept, acc_rate

    return mala_step, run_chain


def make_mala_fast(logpi):
    """
    Optimized MALA with cached gradients (avoids redundant computations).
    
    Caches logpi(theta) and grad(theta) between iterations to avoid recomputing
    the gradient of the current state when it becomes the previous state.
    
    Args:
        logpi: Callable theta -> R (unnormalized log-density)
    
    Returns:
        mala_step_cached: Callable (key, theta, logpi_theta, grad_theta, eps) -> (theta_new, logpi_new, grad_new, accept)
        run_chain: Callable (key, theta_init, eps, n_steps, burn, thin) -> (samples, acc_rate)
    """
    vgrad_logpi = jax.value_and_grad(logpi)

    @jax.jit
    def mala_step_cached(key, theta, logpi_theta, grad_theta, eps):
        """
        Single MALA step with cached values.
        
        Args:
            key: JAX PRNG key
            theta: Current state [d]
            logpi_theta: Cached logpi(theta)
            grad_theta: Cached gradient at theta
            eps: Step size
        
        Returns:
            theta_new: New state
            logpi_new: logpi(theta_new)
            grad_new: gradient at theta_new
            accept: Boolean acceptance flag
        """
        k1, k2 = random.split(key)

        mean = theta + 0.5 * (eps**2) * grad_theta
        prop = mean + eps * random.normal(k1, shape=theta.shape)

        logpi_prop, grad_prop = vgrad_logpi(prop)

        log_alpha = (logpi_prop - logpi_theta
                     + _log_q(theta, prop, grad_prop, eps)
                     - _log_q(prop, theta, grad_theta, eps))

        accept = jnp.log(random.uniform(k2)) < log_alpha
        theta_new = jnp.where(accept, prop, theta)
        logpi_new = jnp.where(accept, logpi_prop, logpi_theta)
        grad_new = jnp.where(accept, grad_prop, grad_theta)

        return theta_new, logpi_new, grad_new, accept

    @partial(jax.jit, static_argnames=("n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=1e-2, n_steps=1000, burn=100, thin=1):
        """
        Run optimized MALA chain with cached gradients.
        
        Args:
            key: JAX PRNG key
            theta_init: Initial state [d]
            eps: Step size
            n_steps: Total number of steps
            burn: Burn-in steps to discard
            thin: Thinning interval
        
        Returns:
            samples: [N_keep, d] post burn/thin
            acc_rate: Acceptance rate over kept samples
        """
        logpi0, grad0 = vgrad_logpi(theta_init)

        def body(carry, _):
            key, theta, lp, g, acc_count = carry
            key, k = random.split(key)
            theta, lp, g, a = mala_step_cached(k, theta, lp, g, eps)
            return (key, theta, lp, g, acc_count + a.astype(jnp.int32)), (theta, a)

        init = (key, theta_init, logpi0, grad0, jnp.array(0, jnp.int32))
        (keyT, thetaT, lpT, gT, acc_count), (thetas, accepts) = lax.scan(
            body, init, xs=None, length=n_steps
        )

        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)

        return thetas_kept, acc_rate

    return mala_step_cached, run_chain


def _log_q_precond(x, y, g_y, eps, precond):
    """Log-density of preconditioned Langevin proposal.

    q(x | y) = N(y + 0.5*eps^2*C*g_y, eps^2*C)
    where C is a constant covariance.
    """
    C = precond["cov"]
    prec = precond["prec"]
    logdetC = precond["logdet"]

    mean = y + 0.5 * (eps**2) * (C @ g_y)
    diff = x - mean
    d = diff.shape[0]

    # Σ = eps^2 * C
    logdet_Sigma = d * jnp.log(eps**2) + logdetC
    quad = (diff @ (prec @ diff)) / (eps**2)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet_Sigma + quad)


def make_mala_precond(logpi, precond):
    """MALA with constant Gaussian preconditioner.

    This uses a fixed covariance C (often approx posterior covariance).
    It is compatible with any logpi(theta).

    Args:
        logpi: Callable theta -> R
        precond: dict with keys {cov, chol, prec, logdet}

    Returns:
        mala_step, run_chain
    """
    vgrad_logpi = jax.value_and_grad(logpi)
    C = precond["cov"]
    Lc = precond["chol"]

    @jax.jit
    def mala_step(key, theta, eps):
        k1, k2 = random.split(key)

        lp, g = vgrad_logpi(theta)
        mean = theta + 0.5 * (eps**2) * (C @ g)
        prop = mean + eps * (Lc @ random.normal(k1, shape=theta.shape))

        lp_prop, g_prop = vgrad_logpi(prop)

        log_alpha = (lp_prop - lp
                     + _log_q_precond(theta, prop, g_prop, eps, precond)
                     - _log_q_precond(prop, theta, g, eps, precond))
        accept = jnp.log(random.uniform(k2)) < log_alpha
        theta_new = jnp.where(accept, prop, theta)
        return theta_new, accept

    @partial(jax.jit, static_argnames=("n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=1e-2, n_steps=1000, burn=100, thin=1):
        keys = random.split(key, n_steps)

        def body(theta, k):
            theta_new, acc = mala_step(k, theta, eps)
            return theta_new, (theta_new, acc)

        _, (thetas, accepts) = lax.scan(body, theta_init, keys)

        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)
        return thetas_kept, acc_rate

    return mala_step, run_chain

"""
Hamiltonian Monte Carlo (HMC) implementation.

All functions work only with logpi(theta), no dependency on space/projections.
Follows the same interface pattern as mcmc_mala.py for consistency.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial


def leapfrog_step(grad_U, q, p, eps):
    """
    Single leapfrog integration step for Hamiltonian dynamics.
    
    Hamilton's equations with potential U = -logpi:
      dq/dt = p
      dp/dt = -grad_U(q)
    
    Args:
        grad_U: Callable q -> gradient of potential energy
        q: Position [d]
        p: Momentum [d]
        eps: Step size
    
    Returns:
        q_new: New position
        p_new: New momentum
    """
    # Half step for momentum
    p_half = p - 0.5 * eps * grad_U(q)
    
    # Full step for position
    q_new = q + eps * p_half
    
    # Half step for momentum
    p_new = p_half - 0.5 * eps * grad_U(q_new)
    
    return q_new, p_new


def leapfrog_integrate(grad_U, q0, p0, eps, L):
    """
    Perform L leapfrog steps.
    
    Args:
        grad_U: Callable q -> gradient of potential energy
        q0: Initial position [d]
        p0: Initial momentum [d]
        eps: Step size
        L: Number of leapfrog steps
    
    Returns:
        q_final: Final position
        p_final: Final momentum
    """
    def body(carry, _):
        q, p = carry
        q_new, p_new = leapfrog_step(grad_U, q, p, eps)
        return (q_new, p_new), None
    
    (q_final, p_final), _ = lax.scan(body, (q0, p0), xs=None, length=L)
    return q_final, p_final


def make_hmc(logpi):
    """
    Standard HMC implementation with leapfrog integration.
    
    Uses Hamiltonian H(q,p) = U(q) + K(p) where:
      U(q) = -logpi(q)  (potential energy)
      K(p) = 0.5 * p^T p  (kinetic energy with unit mass)
    
    Args:
        logpi: Callable theta -> R (unnormalized log-density)
    
    Returns:
        hmc_step: Callable (key, theta, eps, L) -> (theta_new, accept_bool)
        run_chain: Callable (key, theta_init, eps, L, n_steps, burn, thin) -> (samples, acc_rate)
    """
    # Potential U = -logpi, so grad_U = -grad(logpi)
    grad_logpi = jax.grad(logpi)
    
    def grad_U(q):
        return -grad_logpi(q)
    
    @partial(jax.jit, static_argnames=("L",))
    def hmc_step(key, q, eps, L):
        """
        Single HMC step.
        
        Args:
            key: JAX PRNG key
            q: Current position [d]
            eps: Leapfrog step size
            L: Number of leapfrog steps
        
        Returns:
            q_new: New position
            accept: Boolean acceptance flag
        """
        k1, k2 = random.split(key)
        
        # Sample momentum from standard normal
        p = random.normal(k1, shape=q.shape)
        
        # Current Hamiltonian
        current_U = -logpi(q)
        current_K = 0.5 * jnp.dot(p, p)
        current_H = current_U + current_K
        
        # Leapfrog integration
        q_prop, p_prop = leapfrog_integrate(grad_U, q, p, eps, L)
        
        # Negate momentum for reversibility (optional, doesn't affect acceptance)
        p_prop = -p_prop
        
        # Proposed Hamiltonian
        proposed_U = -logpi(q_prop)
        proposed_K = 0.5 * jnp.dot(p_prop, p_prop)
        proposed_H = proposed_U + proposed_K
        
        # Metropolis acceptance
        log_accept_prob = current_H - proposed_H
        accept = jnp.log(random.uniform(k2)) < log_accept_prob
        
        q_new = jnp.where(accept, q_prop, q)
        return q_new, accept
    
    @partial(jax.jit, static_argnames=("L", "n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=0.01, L=10, n_steps=1000, burn=100, thin=1):
        """
        Run HMC chain.
        
        Args:
            key: JAX PRNG key
            theta_init: Initial state [d]
            eps: Leapfrog step size
            L: Number of leapfrog steps per HMC step
            n_steps: Total number of HMC steps
            burn: Burn-in steps to discard
            thin: Thinning interval
        
        Returns:
            samples: [N_keep, d] post burn/thin
            acc_rate: Acceptance rate over kept samples
        """
        keys = random.split(key, n_steps)
        
        def body(theta, k):
            theta_new, acc = hmc_step(k, theta, eps, L)
            return theta_new, (theta_new, acc)
        
        _, (thetas, accepts) = lax.scan(body, theta_init, keys)
        
        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)
        return thetas_kept, acc_rate
    
    return hmc_step, run_chain


def make_hmc_mass(logpi, precond):
    """HMC with constant mass matrix.

    Uses mass M = precond["cov"]. Momentum p ~ N(0, M).
    K(p) = 0.5 * p^T M^{-1} p.

    Args:
        logpi: Callable theta -> R
        precond: dict with keys {cov, chol, prec, logdet}

    Returns:
        hmc_step, run_chain
    """
    M = precond["cov"]
    Minv = precond["prec"]
    Lm = precond["chol"]

    grad_logpi = jax.grad(logpi)

    def grad_U(q):
        return -grad_logpi(q)

    def kinetic(p):
        return 0.5 * (p @ (Minv @ p))

    def leapfrog_step_mass(q, p, eps):
        p_half = p - 0.5 * eps * grad_U(q)
        q_new = q + eps * (Minv @ p_half)
        p_new = p_half - 0.5 * eps * grad_U(q_new)
        return q_new, p_new

    def leapfrog_integrate_mass(q0, p0, eps, L):
        def body(carry, _):
            q, p = carry
            q, p = leapfrog_step_mass(q, p, eps)
            return (q, p), None

        (q_final, p_final), _ = lax.scan(body, (q0, p0), xs=None, length=L)
        return q_final, p_final

    @partial(jax.jit, static_argnames=("L",))
    def hmc_step(key, q, eps, L):
        k1, k2 = random.split(key)

        # p ~ N(0, M)
        p = Lm @ random.normal(k1, shape=q.shape)

        current_U = -logpi(q)
        current_K = kinetic(p)
        current_H = current_U + current_K

        q_prop, p_prop = leapfrog_integrate_mass(q, p, eps, L)
        p_prop = -p_prop

        proposed_U = -logpi(q_prop)
        proposed_K = kinetic(p_prop)
        proposed_H = proposed_U + proposed_K

        log_accept_prob = current_H - proposed_H
        accept = jnp.log(random.uniform(k2)) < log_accept_prob
        q_new = jnp.where(accept, q_prop, q)
        return q_new, accept

    @partial(jax.jit, static_argnames=("L", "n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=0.01, L=10, n_steps=1000, burn=100, thin=1):
        keys = random.split(key, n_steps)

        def body(theta, k):
            theta_new, acc = hmc_step(k, theta, eps, L)
            return theta_new, (theta_new, acc)

        _, (thetas, accepts) = lax.scan(body, theta_init, keys)

        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)
        return thetas_kept, acc_rate

    return hmc_step, run_chain


def make_hmc_fast(logpi):
    """
    Optimized HMC with cached gradient evaluations.
    
    Caches logpi and gradient to avoid redundant computation when
    proposal is rejected.
    
    Args:
        logpi: Callable theta -> R (unnormalized log-density)
    
    Returns:
        hmc_step_cached: Callable (key, theta, logpi_theta, grad_theta, eps, L) -> (...)
        run_chain: Callable (key, theta_init, eps, L, n_steps, burn, thin) -> (samples, acc_rate)
    """
    vgrad_logpi = jax.value_and_grad(logpi)
    grad_logpi = jax.grad(logpi)
    
    def grad_U(q):
        return -grad_logpi(q)
    
    @partial(jax.jit, static_argnames=("L",))
    def hmc_step_cached(key, q, logpi_q, grad_q, eps, L):
        """
        Single HMC step with cached values.
        
        Args:
            key: JAX PRNG key
            q: Current position [d]
            logpi_q: Cached logpi(q)
            grad_q: Cached gradient at q
            eps: Leapfrog step size
            L: Number of leapfrog steps
        
        Returns:
            q_new: New position
            logpi_new: logpi(q_new)
            grad_new: gradient at q_new
            accept: Boolean acceptance flag
        """
        k1, k2 = random.split(key)
        
        # Sample momentum
        p = random.normal(k1, shape=q.shape)
        
        # Current Hamiltonian (use cached logpi)
        current_U = -logpi_q
        current_K = 0.5 * jnp.dot(p, p)
        current_H = current_U + current_K
        
        # Leapfrog integration
        q_prop, p_prop = leapfrog_integrate(grad_U, q, p, eps, L)
        p_prop = -p_prop
        
        # Proposed Hamiltonian (compute new logpi and gradient)
        logpi_prop, grad_prop = vgrad_logpi(q_prop)
        proposed_U = -logpi_prop
        proposed_K = 0.5 * jnp.dot(p_prop, p_prop)
        proposed_H = proposed_U + proposed_K
        
        # Metropolis acceptance
        log_accept_prob = current_H - proposed_H
        accept = jnp.log(random.uniform(k2)) < log_accept_prob
        
        q_new = jnp.where(accept, q_prop, q)
        logpi_new = jnp.where(accept, logpi_prop, logpi_q)
        grad_new = jnp.where(accept, grad_prop, grad_q)
        
        return q_new, logpi_new, grad_new, accept
    
    @partial(jax.jit, static_argnames=("L", "n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=0.01, L=10, n_steps=1000, burn=100, thin=1):
        """
        Run optimized HMC chain with cached gradients.
        
        Args:
            key: JAX PRNG key
            theta_init: Initial state [d]
            eps: Leapfrog step size
            L: Number of leapfrog steps per HMC step
            n_steps: Total number of HMC steps
            burn: Burn-in steps to discard
            thin: Thinning interval
        
        Returns:
            samples: [N_keep, d] post burn/thin
            acc_rate: Acceptance rate over kept samples
        """
        logpi0, grad0 = vgrad_logpi(theta_init)
        
        def body(carry, _):
            key, q, lp, g, acc_count = carry
            key, k = random.split(key)
            q, lp, g, a = hmc_step_cached(k, q, lp, g, eps, L)
            return (key, q, lp, g, acc_count + a.astype(jnp.int32)), (q, a)
        
        init = (key, theta_init, logpi0, grad0, jnp.array(0, jnp.int32))
        (keyT, qT, lpT, gT, acc_count), (thetas, accepts) = lax.scan(
            body, init, xs=None, length=n_steps
        )
        
        thetas_kept = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate = jnp.mean(accepts_kept)
        
        return thetas_kept, acc_rate
    
    return hmc_step_cached, run_chain

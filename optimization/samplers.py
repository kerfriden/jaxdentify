import jax
import jax.numpy as jnp
from jax import random, lax

from functools import partial

from optimization.parameter_mappings import project_theta, to_params

# -------------------- Convenience utilities --------------------
def map_to_params(space, thetas):
    """Vectorize θ -> projected θ -> physical params dict (PyTree of arrays)."""
    proj = jax.vmap(lambda th: project_theta(space, th))(thetas)
    params_list = jax.vmap(lambda th: to_params(space, th))(proj)  # PyTree with leading dim N
    return params_list

def posterior_param_summary(thetas, space):
    """
    Compute posterior mean and 5/95 percentiles in parameter space.
    Returns a dict: {name: (mean, p05, p95)}. Each entry can be scalar or array.
    """
    params_list = map_to_params(space, thetas)  # PyTree of arrays with shape [N, ...]
    def _summ(x):
        mean = jnp.mean(x, axis=0)
        # jnp.percentile in recent JAX uses 'method' arg; 'linear' still OK.
        p05  = jnp.percentile(x, 5.0, axis=0, method="linear")
        p95  = jnp.percentile(x, 95.0, axis=0, method="linear")
        return (mean, p05, p95)

    # Use jax.tree.map (or jtu.tree_map) instead of deprecated jax.tree_map
    return jax.tree.map(_summ, params_list)
    # OR: return jtu.tree_map(_summ, params_list)

def posterior_predictive(thetas,space,pred_func):
    """
    Generate predictive mean and 5/95 bands for σ11(t) using forward method
    Assumes pred_func(params) -> [T], same T for all samples.
    """
    # Build params for each sample (using projection internally)
    params_list = map_to_params(space, thetas)

    # vmap the forward through the list of param PyTrees
    preds = jax.vmap(lambda p: pred_func(p))(params_list)  # [N, T]

    mean = jnp.mean(preds, axis=0)
    p05  = jnp.percentile(preds, 5.0, axis=0, method="linear")
    p95  = jnp.percentile(preds, 95.0, axis=0, method="linear")
    return mean, p05, p95, preds

# -------------------- mala --------------------

def _log_q(x, y, g_y, eps):
    """log q(x | y) for Langevin proposal N(y + 0.5 eps^2 g_y, eps^2 I)."""
    mean = y + 0.5 * (eps**2) * g_y
    diff = x - mean
    return -0.5 * jnp.dot(diff, diff) / (eps**2)

def make_mala(logpi):
    """
    Given logpi(theta): R^d -> R (unnormalized log-density),
    return two compiled callables:
      mala_step(key, theta, eps) -> (theta_new, accept_bool)
      run_chain(key, theta_init, eps, n_steps, burn, thin) -> (samples, acc_rate)
    """
    grad_logpi = jax.grad(logpi)

    @jax.jit
    def mala_step(key, theta, eps):
        # propose
        k1, k2 = random.split(key)
        g      = grad_logpi(theta)
        mean   = theta + 0.5 * (eps**2) * g
        prop   = mean + eps * random.normal(k1, shape=theta.shape)

        # MH accept
        g_prop    = grad_logpi(prop)
        log_alpha = (logpi(prop) - logpi(theta)
                     + _log_q(theta, prop, g_prop, eps)
                     - _log_q(prop, theta, g, eps))
        accept    = jnp.log(random.uniform(k2)) < log_alpha
        theta_new = jnp.where(accept, prop, theta)
        return theta_new, accept

    @partial(jax.jit, static_argnames=("n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=1e-2, n_steps=1000, burn=100, thin=1):
        """
        Run MALA for this specific logpi.
        Returns:
          samples: [N_keep, d]  post burn/thin
          acc_rate: scalar      acceptance rate over kept samples
        """
        keys = random.split(key, n_steps)

        def body(theta, k):
            theta_new, acc = mala_step(k, theta, eps)
            return theta_new, (theta_new, acc)

        _, (thetas, accepts) = lax.scan(body, theta_init, keys)

        thetas_kept  = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate     = jnp.mean(accepts_kept)
        return thetas_kept, acc_rate

    return mala_step, run_chain

def make_mala_fast(logpi):
    vgrad_logpi = jax.value_and_grad(logpi)

    @jax.jit
    def mala_step_cached(key, theta, logpi_theta, grad_theta, eps):
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
        grad_new  = jnp.where(accept, grad_prop, grad_theta)

        return theta_new, logpi_new, grad_new, accept

    @partial(jax.jit, static_argnames=("n_steps", "burn", "thin"))
    def run_chain(key, theta_init, eps=1e-2, n_steps=1000, burn=100, thin=1):
        logpi0, grad0 = vgrad_logpi(theta_init)

        def body(carry, _):
            key, theta, lp, g, acc = carry
            key, k = random.split(key)
            theta, lp, g, a = mala_step_cached(k, theta, lp, g, eps)
            return (key, theta, lp, g, acc + a.astype(jnp.int32)), (theta, a)

        init = (key, theta_init, logpi0, grad0, jnp.array(0, jnp.int32))
        (keyT, thetaT, lpT, gT, acc_count), (thetas, accepts) = lax.scan(
            body, init, xs=None, length=n_steps
        )

        thetas_kept  = thetas[burn::thin]
        accepts_kept = accepts[burn::thin]
        acc_rate     = jnp.mean(accepts_kept)

        return thetas_kept, acc_rate

    return mala_step_cached, run_chain


def make_mala_unit_gaussian_prior(loglik):

    def logprior(theta):
        # Unit Gaussian prior on *raw* unconstrained theta
        return -0.5 * jnp.dot(theta, theta)

    def logpost(theta):
        return loglik(theta) + logprior(theta)

    _ , run_mala = make_mala_fast(logpost)
    #_ , run_mala = make_mala(logpost)

    return run_mala
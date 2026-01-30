"""
Variational Inference with Normalizing Flows.

Implements:
- Gaussian VI (baseline)
- RealNVP Normalizing Flow with affine coupling + permutations

Architecture follows best practices for robust training on narrow posteriors.
"""

import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from functools import partial
import numpy as np


# ==================== Gaussian VI ====================

def gaussian_vi_elbo(logpi, mu, log_sigma, key, n_samples=10):
    """
    Compute Evidence Lower BOund (ELBO) for Gaussian VI.
    
    q(theta) = N(mu, diag(sigma^2))
    ELBO = E_q[log pi(theta) - log q(theta)]

    IMPORTANT: In this codebase `logpi` is the *log posterior* (log-likelihood + log-prior)
    constructed via `optimization.targets.as_logpi`. Therefore we must subtract `log q`.
    Subtracting KL(q || prior) here would *double-count the prior* and makes the fitted
    Gaussian far too narrow.
    
    Args:
        logpi: Target log-density theta -> R
        mu: Mean [d]
        log_sigma: Log std [d]
        key: JAX PRNG key
        n_samples: Monte Carlo samples
    
    Returns:
        elbo: Scalar ELBO
    """
    d = mu.shape[0]
    sigma = jnp.exp(log_sigma)
    
    eps = random.normal(key, shape=(n_samples, d))
    theta_samples = mu + sigma * eps
    
    # NOTE: For many PDE/implicit-solvers, `vmap(logpi)` can become *much* slower than
    # a mapped loop due to vectorizing control flow / linear solves. For the small
    # MC sample counts used here, `lax.map` is typically the best tradeoff.
    log_p_samples = lax.map(logpi, theta_samples)

    # log q(theta) for theta = mu + sigma * eps
    # log q = -0.5 * sum(eps^2 + 2 log_sigma + log(2pi))
    log_q_samples = -0.5 * (
        jnp.sum(eps * eps, axis=-1)
        + 2.0 * jnp.sum(log_sigma)
        + d * jnp.log(2.0 * jnp.pi)
    )

    return jnp.mean(log_p_samples - log_q_samples)


def fit_gaussian_vi(
    logpi,
    d,
    key,
    n_iters=1000,
    n_samples=10,
    lr=0.01,
    verbose=True,
    print_every=None,
    *,
    mu0=None,
    log_sigma0=None,
):
    """Fit Gaussian VI using gradient ascent on ELBO.

    Args:
        mu0: Optional initial mean [d]. If None, uses small random init.
        log_sigma0: Optional initial log-std [d]. If None, starts at zeros.
    """
    key, k1, k2 = random.split(key, 3)
    if mu0 is None:
        mu = random.normal(k1, shape=(d,)) * 0.1
    else:
        mu = jnp.asarray(mu0)
    if log_sigma0 is None:
        log_sigma = jnp.zeros(d)
    else:
        log_sigma = jnp.asarray(log_sigma0)
    
    def elbo_fn(mu, log_sigma, key_in):
        return gaussian_vi_elbo(logpi, mu, log_sigma, key_in, n_samples)

    elbo_grad = jax.value_and_grad(elbo_fn, argnums=(0, 1))

    @jax.jit
    def step(key_in, mu_in, log_sigma_in):
        key_out, k_elbo = random.split(key_in)
        elbo, (grad_mu, grad_log_sigma) = elbo_grad(mu_in, log_sigma_in, k_elbo)
        mu_out = mu_in + lr * grad_mu
        log_sigma_out = log_sigma_in + lr * grad_log_sigma
        return key_out, mu_out, log_sigma_out, elbo

    # Default printing cadence: ~10 updates.
    if print_every is None:
        print_every = max(1, int(n_iters) // 10) if int(n_iters) > 0 else 1
    else:
        print_every = max(1, int(print_every))

    elbo_history = []

    # First call compiles the full VI step (can be heavy for implicit solvers).
    if int(n_iters) > 0:
        key, mu, log_sigma, elbo0 = step(key, mu, log_sigma)
        # Synchronize so compilation time isn't hidden.
        (mu.block_until_ready() if hasattr(mu, "block_until_ready") else mu)
        elbo_history.append(float(elbo0))
        if verbose:
            print(
                f"Iter {0:4d}: ELBO = {float(elbo0):.4f}, "
                f"||mu|| = {float(jnp.linalg.norm(mu)):.4f}, "
                f"mean(sigma) = {float(jnp.mean(jnp.exp(log_sigma))):.4f}"
            )

    for i in range(1, int(n_iters)):
        key, mu, log_sigma, elbo = step(key, mu, log_sigma)
        elbo_history.append(float(elbo))

        if verbose and (i % print_every == 0 or i == int(n_iters) - 1):
            print(
                f"Iter {i:4d}: ELBO = {float(elbo):.4f}, "
                f"||mu|| = {float(jnp.linalg.norm(mu)):.4f}, "
                f"mean(sigma) = {float(jnp.mean(jnp.exp(log_sigma))):.4f}"
            )

    return mu, log_sigma, jnp.asarray(elbo_history)


def sample_gaussian_vi(mu, log_sigma, key, n_samples):
    """Sample from fitted Gaussian VI."""
    d = mu.shape[0]
    sigma = jnp.exp(log_sigma)
    eps = random.normal(key, shape=(n_samples, d))
    return mu + sigma * eps


# ==================== RealNVP Normalizing Flow ====================

def affine_coupling_forward(x, params, mask, s_cap=1.5):
    """
    RealNVP affine coupling with robust scale clipping.
    s = s_cap * tanh(s_raw) for bounded scales.
    """
    d = x.shape[0]
    
    x_cond = x * mask
    h1 = jnp.tanh(x_cond @ params['W1'] + params['b1'])
    h2 = jnp.tanh(h1 @ params['W2'] + params['b2'])
    out = h2 @ params['W3'] + params['b3']
    
    s_raw = out[:d]
    t = out[d:]
    s = s_cap * jnp.tanh(s_raw)
    
    s_active = s * (1.0 - mask)
    t_active = t * (1.0 - mask)
    
    y = x * jnp.exp(s_active) + t_active
    log_det = jnp.sum(s_active)
    
    return y, log_det


def affine_coupling_inverse(y, params, mask, s_cap=1.5):
    """Inverse of affine coupling."""
    d = y.shape[0]
    
    x_cond = y * mask
    h1 = jnp.tanh(x_cond @ params['W1'] + params['b1'])
    h2 = jnp.tanh(h1 @ params['W2'] + params['b2'])
    out = h2 @ params['W3'] + params['b3']
    
    s_raw = out[:d]
    t = out[d:]
    s = s_cap * jnp.tanh(s_raw)
    
    s_active = s * (1.0 - mask)
    t_active = t * (1.0 - mask)
    
    x = (y - t_active) * jnp.exp(-s_active)
    return x


def make_flow_vi(
    logpi,
    d,
    n_layers=12,
    hidden_dim=64,
    s_cap=2.2,
    use_random_perm=False,
    *,
    base_mean=None,
    base_chol=None,
):
    """
    RealNVP flow with batched coupling layers (works well for narrow distributions).
    
    Args:
        logpi: log density function
        d: dimensionality (2 for 2D problems)
        n_layers: number of coupling layers (12 for d=2, 24 for d=10)
        hidden_dim: MLP hidden dimension
        s_cap: scale clipping for numerical stability
    """
    
    def init_mlp(key, in_dim, out_dim, n_hidden=2, scale=0.02):
        """Initialize 2-layer MLP."""
        keys = random.split(key, n_hidden + 1)
        dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]
        params = []
        for k, (din, dout) in zip(keys, zip(dims[:-1], dims[1:])):
            W = scale * random.normal(k, (din, dout))
            b = jnp.zeros((dout,))
            params.append({'W': W, 'b': b})
        return params
    
    def mlp_forward(params, x):
        """Forward pass through MLP."""
        h = x
        for i, layer in enumerate(params):
            h = h @ layer['W'] + layer['b']
            if i < len(params) - 1:
                h = jnp.tanh(h)
        return h
    
    def coupling_forward_2d(params_st, x, mask):
        """Batched affine coupling for 2D: x [N, 2] -> y [N, 2]."""
        x0, x1 = x[:, 0:1], x[:, 1:2]
        
        # Use lax.cond for JAX-compatible conditional
        def transform_x1(args):
            x0, x1, params_st = args
            st = mlp_forward(params_st, x0)
            s, t = st[:, 0:1], st[:, 1:2]
            s = s_cap * jnp.tanh(s)
            y0 = x0
            y1 = x1 * jnp.exp(s) + t
            y = jnp.concatenate([y0, y1], axis=1)
            logdet = jnp.squeeze(s, axis=1)
            return y, logdet
        
        def transform_x0(args):
            x0, x1, params_st = args
            st = mlp_forward(params_st, x1)
            s, t = st[:, 0:1], st[:, 1:2]
            s = s_cap * jnp.tanh(s)
            y1 = x1
            y0 = x0 * jnp.exp(s) + t
            y = jnp.concatenate([y0, y1], axis=1)
            logdet = jnp.squeeze(s, axis=1)
            return y, logdet
        
        return lax.cond(mask == 0, transform_x1, transform_x0, (x0, x1, params_st))
    
    def coupling_inverse_2d(params_st, y, mask):
        """Inverse of batched affine coupling for 2D."""
        y0, y1 = y[:, 0:1], y[:, 1:2]
        
        def inverse_x1(args):
            y0, y1, params_st = args
            st = mlp_forward(params_st, y0)
            s, t = st[:, 0:1], st[:, 1:2]
            s = s_cap * jnp.tanh(s)
            x0 = y0
            x1 = (y1 - t) * jnp.exp(-s)
            x = jnp.concatenate([x0, x1], axis=1)
            logdet_inv = -jnp.squeeze(s, axis=1)
            return x, logdet_inv
        
        def inverse_x0(args):
            y0, y1, params_st = args
            st = mlp_forward(params_st, y1)
            s, t = st[:, 0:1], st[:, 1:2]
            s = s_cap * jnp.tanh(s)
            x1 = y1
            x0 = (y0 - t) * jnp.exp(-s)
            x = jnp.concatenate([x0, x1], axis=1)
            logdet_inv = -jnp.squeeze(s, axis=1)
            return x, logdet_inv
        
        return lax.cond(mask == 0, inverse_x1, inverse_x0, (y0, y1, params_st))
    
    def init_flow_params(key, n_layers):
        """Initialize flow parameters with alternating masks."""
        keys = random.split(key, n_layers)
        layers = [init_mlp(k, 1, 2, n_hidden=2) for k in keys]
        masks = tuple(i % 2 for i in range(n_layers))
        return {"layers": layers}, masks
    
    base_mean_arr = jnp.zeros((d,)) if base_mean is None else jnp.asarray(base_mean)
    base_chol_arr = jnp.eye(d) if base_chol is None else jnp.asarray(base_chol)
    base_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(base_chol_arr)) + 1e-32))

    @jax.jit
    def flow_forward(z, params_and_masks):
        """z [N, d] -> theta [N, d]."""
        params, masks = params_and_masks
        x = z
        logdet = jnp.zeros((z.shape[0],))
        for layer, mask in zip(params["layers"], masks):
            x, ld = coupling_forward_2d(layer, x, mask)
            logdet = logdet + ld

        # Optional outer affine transform: theta = mean + x @ L.T
        theta = base_mean_arr[None, :] + x @ base_chol_arr.T
        logdet = logdet + base_logdet
        return theta, logdet
    
    @jax.jit
    def flow_inverse(theta, params_and_masks):
        """theta [N, d] -> z [N, d]."""
        params, masks = params_and_masks

        # Invert outer affine: x = (theta - mean) @ inv(L.T)
        x = (theta - base_mean_arr[None, :])
        x = jax.scipy.linalg.solve_triangular(base_chol_arr, x.T, lower=True).T
        logdet_inv = jnp.zeros((theta.shape[0],)) - base_logdet

        for layer, mask in zip(params["layers"][::-1], masks[::-1]):
            x, ld = coupling_inverse_2d(layer, x, mask)
            logdet_inv = logdet_inv + ld
        return x, logdet_inv
    
    def logp_base(z):
        """Standard Gaussian base density."""
        return -0.5 * (d * jnp.log(2.0 * jnp.pi) + jnp.sum(z**2, axis=-1))
    
    def flow_elbo(params_and_masks, key, n_samples=128):
        """ELBO for flow VI (batched)."""
        z = random.normal(key, (n_samples, d))
        theta, logdet = flow_forward(z, params_and_masks)
        # See note above: `lax.map` avoids pathological slowdowns that can occur
        # when vectorizing implicit solvers with `vmap`.
        logp = lax.map(logpi, theta)
        return jnp.mean(logp - logp_base(z) + logdet)
    
    def adam_init(pytree):
        """Initialize Adam optimizer state."""
        return {
            'm': jax.tree.map(jnp.zeros_like, pytree),
            'v': jax.tree.map(jnp.zeros_like, pytree),
            't': jnp.array(0, dtype=jnp.int32)
        }
    
    def adam_update(params, grads, state, lr=0.002, b1=0.9, b2=0.999, eps=1e-8):
        """Adam optimizer update."""
        t = state['t'] + 1
        m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state['m'], grads)
        v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * (g * g), state['v'], grads)
        mhat = jax.tree.map(lambda m: m / (1 - b1**t), m)
        vhat = jax.tree.map(lambda v: v / (1 - b2**t), v)
        params = jax.tree.map(lambda p, mh, vh: p + lr * mh / (jnp.sqrt(vh) + eps),
                             params, mhat, vhat)
        return params, {'m': m, 'v': v, 't': t}
    
    def fit_flow(
        key,
        n_iters=3000,
        n_samples=128,
        lr=0.002,
        verbose=True,
        print_every=50,
        profile=False,
        profile_n=3,
        *,
        return_info: bool = False,
    ):
        """Fit flow using Adam with Python loop for real-time progress.

        Args:
            profile: If True, prints a timing breakdown for ELBO/value&grad.
            profile_n: Number of profiling repetitions to average (after warmup).
        """
        key, k_init = random.split(key)
        params, masks = init_flow_params(k_init, n_layers)
        
        # We only optimize params, not masks
        state = adam_init(params)
        
        @jax.jit
        def step(key, params, state):
            key, k_elbo = random.split(key)
            params_and_masks = (params, masks)  # masks are constant
            val, grads = jax.value_and_grad(
                lambda p: flow_elbo((p, masks), k_elbo, n_samples)
            )(params)
            params, state = adam_update(params, grads, state, lr=lr)
            return key, params, state, val

        def _block(x):
            # Robustly synchronize on any pytree.
            return jax.tree.map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)

        def profile_once(key, params, state):
            """Profile a single VI step (no parameter update) and return timing breakdown."""
            import time

            key, k_elbo = random.split(key)
            params_and_masks = (params, masks)

            t0 = time.perf_counter()
            z = random.normal(k_elbo, (n_samples, d))
            _block(z)
            t1 = time.perf_counter()

            theta, logdet = flow_forward(z, params_and_masks)
            _block((theta, logdet))
            t2 = time.perf_counter()

            logp = lax.map(logpi, theta)
            _block(logp)
            t3 = time.perf_counter()

            elbo = jnp.mean(logp - logp_base(z) + logdet)
            _block(elbo)
            t4 = time.perf_counter()

            def elbo_from_params(p):
                return flow_elbo((p, masks), k_elbo, n_samples)

            val, grads = jax.value_and_grad(elbo_from_params)(params)
            _block((val, grads))
            t5 = time.perf_counter()

            t6 = time.perf_counter()

            return (
                {
                    "z_sample": t1 - t0,
                    "flow_forward": t2 - t1,
                    "logpi_eval": t3 - t2,
                    "elbo_reduce": t4 - t3,
                    "value_and_grad": t5 - t4,
                    "total": t6 - t0,
                    "elbo": float(val),
                },
                key,
                params,
                state,
            )

        # Time the first step separately (includes compilation on first call).
        import time as _time
        timing = {
            "compile_step": 0.0,
            "steady_total": 0.0,
            "total": 0.0,
            "steady_iters": max(0, int(n_iters) - 1),
            "n_iters": int(n_iters),
        }

        elbo_history = []
        if int(n_iters) > 0:
            t0 = _time.perf_counter()
            key, params, state, val0 = step(key, params, state)
            _block((params, state, val0))
            t1 = _time.perf_counter()
            timing["compile_step"] = t1 - t0
            elbo_history.append(val0)
            if verbose:
                print(f"Iter {0:4d}: ELBO = {float(val0):.4f}")

        if profile:
            reps = max(1, int(profile_n))
            acc = {
                "z_sample": 0.0,
                "flow_forward": 0.0,
                "logpi_eval": 0.0,
                "elbo_reduce": 0.0,
                "value_and_grad": 0.0,
                "total": 0.0,
            }
            last_elbo = None
            for _ in range(reps):
                stats, key, params, state = profile_once(key, params, state)
                last_elbo = stats["elbo"]
                for k in acc:
                    acc[k] += stats[k]

            print("\n[Flow VI timing profile] (averaged over", reps, "steps)")
            for k in [
                "z_sample",
                "flow_forward",
                "logpi_eval",
                "elbo_reduce",
                "value_and_grad",
                "total",
            ]:
                print(f"  {k:>14s}: {acc[k] / reps:.6f} s")
            if last_elbo is not None:
                print(f"  {'elbo(last)':>14s}: {last_elbo:.6f}")

        # Steady-state steps (already compiled).
        if int(n_iters) > 1:
            t0 = _time.perf_counter()
            for i in range(1, int(n_iters)):
                key, params, state, val = step(key, params, state)
                elbo_history.append(val)

                if verbose and (i % int(print_every) == 0):
                    print(f"Iter {i:4d}: ELBO = {float(val):.4f}")

            _block((params, state, elbo_history[-1]))
            t1 = _time.perf_counter()
            timing["steady_total"] = t1 - t0

        timing["total"] = timing["compile_step"] + timing["steady_total"]
        if verbose:
            steady_iters = max(0, int(n_iters) - 1)
            if steady_iters > 0:
                print(
                    f"\n[Flow VI timing] first step (compile+run): {timing['compile_step']:.6f} s; "
                    f"steady-state: {timing['steady_total']:.6f} s over {steady_iters} iters "
                    f"({timing['steady_total']/steady_iters:.6f} s/iter)"
                )
            else:
                print(f"\n[Flow VI timing] first step (compile+run): {timing['compile_step']:.6f} s")

        out = (params, masks), np.array(elbo_history)
        if return_info:
            return out[0], out[1], timing
        return out
    
    def sample_flow(params_and_masks, key, n_samples=2000):
        """Sample from fitted flow."""
        z = random.normal(key, (n_samples, d))
        theta, _ = flow_forward(z, params_and_masks)
        return np.array(theta)
    
    return flow_forward, flow_inverse, fit_flow, sample_flow

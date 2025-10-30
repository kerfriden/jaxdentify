import jax
import jax.numpy as jnp
from jax import  lax
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve as la_solve

from functools import partial


def newton(residual_fn, x0, dyn_args, tol=1e-8, abs_tol=1e-12, max_iter=100):
    tol     = jnp.asarray(tol,     dtype=x0.dtype)
    abs_tol = jnp.asarray(abs_tol, dtype=x0.dtype)

    def res(x): return residual_fn(x, *dyn_args)
    Jfun = jax.jacfwd(res)  # <- comme dans ta version qui converge

    R0   = res(x0)
    Rini = jnp.linalg.norm(R0)

    def cond(c):
        x, i = c
        nR = jnp.linalg.norm(res(x))
        done = (nR < tol * Rini) | (nR < abs_tol)
        return (i < max_iter) & (~done)

    def body(c):
        x, i = c
        R  = res(x)
        J  = Jfun(x)
        #dx = jnp.linalg.solve(J, R)
        dx = la_solve(J, R, assume_a='gen')  # un poil plus robuste
        return (x - dx, i + 1)

    x_fin, iters = lax.while_loop(cond, body, (x0, 0))
    Rend  = jnp.linalg.norm(res(x_fin))
    done  = (Rend < tol*Rini) | (Rend < abs_tol)
    iters = jnp.where(done, iters, -1)
    return x_fin, iters

# ----------------- dict/PyTree wrapper around array Newton -----------------
def newton_unravel(residual_fn_pytree, x0_tree, dyn_args, tol=1e-6, abs_tol=1e-8, max_iter=100):
    """
    residual_fn_pytree(x_tree, *dyn_args) -> residual_tree (same PyTree structure)
    x0_tree: PyTree (dicts, arrays…)
    Returns: (x_tree_solution, iters)
    """
    x0_flat, unravel_x = ravel_pytree(x0_tree)

    def res_flat(x_flat, *dyn):
        x_tree = unravel_x(x_flat)
        r_tree = residual_fn_pytree(x_tree, *dyn)
        r_flat, _ = ravel_pytree(r_tree)
        return r_flat

    x_fin_flat, iters = newton(res_flat, x0_flat, dyn_args, tol, abs_tol, max_iter)
    x_fin_tree = unravel_x(x_fin_flat)
    return x_fin_tree, iters


def newton_fixed_scan(residual_fn, x0, dyn_args, tol=1e-8, abs_tol=1e-12, max_iter=10):
    x_dtype = getattr(x0, "dtype", jnp.float64)
    tol     = jnp.asarray(tol,     dtype=x_dtype)
    abs_tol = jnp.asarray(abs_tol, dtype=x_dtype)

    def res(x): return residual_fn(x, *dyn_args)
    Jfun = jax.jacfwd(res)

    R0   = res(x0)
    Rini = jnp.linalg.norm(R0)

    def do_iter(x, iters):
        R  = res(x)
        J  = Jfun(x)
        dx = jnp.linalg.solve(J, R)
        x_cand = x - dx
        nR = jnp.linalg.norm(res(x_cand))
        conv_now = (nR < tol * Rini) | (nR < abs_tol)
        return x_cand, (iters + 1), conv_now

    def body(carry, _):
        x, iters, done = carry
        def keep(_): return x, iters, done
        def step(_):
            x1, it1, conv_now = do_iter(x, iters)
            done1  = done | conv_now
            x_next = jnp.where(done, x,  x1)
            it_next= jnp.where(done, iters, it1)
            return x_next, it_next, done1
        return lax.cond(done, keep, step, operand=None), None

    # length must be a concrete int
    (x_fin, iters, done_fin), _ = lax.scan(
        body, (x0, jnp.int32(0), jnp.array(False)), xs=None, length=int(max_iter)
    )
    iters = jnp.where(done_fin, iters, jnp.int32(-1))
    return x_fin, iters

# ----------------------------------------------------------
# Implicit-gradient Newton via custom VJP (IFT-based)
# nondiff_argnums: residual_fn, tol, abs_tol, max_iter are static
# ----------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def newton_implicit(residual_fn, x0, dyn_args, tol=1e-8, abs_tol=1e-12, max_iter=10):
    x_star, iters = newton_fixed_scan(residual_fn, x0, dyn_args, tol, abs_tol, max_iter)
    return x_star, iters

# FWD MUST KEEP THE SAME ORDER AS THE PRIMAL
def _newton_fwd(residual_fn, x0, dyn_args, tol=1e-8, abs_tol=1e-12, max_iter=10):
    x_star, iters = newton_fixed_scan(residual_fn, x0, dyn_args, tol, abs_tol, max_iter)
    aux = (x_star, dyn_args)  # stash what we need
    return (x_star, iters), aux

# BWD receives nondiff args first, then aux, ct; return grads for diff args only
def _newton_bwd(residual_fn, tol, abs_tol, max_iter, aux, ct):
    x_star, dyn_args = aux
    ct_x, _ct_iters = ct  # ignore integer cotangent

    def F_x(x):            # ∂F/∂x at solution
        return residual_fn(x, *dyn_args)

    def F_theta(*theta):   # ∂F/∂theta at solution
        return residual_fn(x_star, *theta)

    Jx  = jax.jacfwd(F_x)(x_star)            # (n,n)
    lam = jnp.linalg.solve(Jx.T, ct_x)   # Jx^T λ = \bar{x}

    _, vjp_theta = jax.vjp(F_theta, *dyn_args)
    grads_theta  = vjp_theta(-lam)       # -(∂F/∂theta)^T λ

    grad_x0 = jnp.zeros_like(x_star)     # no grad wrt initial guess (implicit)
    return grad_x0, grads_theta          # EXACTLY matches (x0, dyn_args)

# register VJP
newton_implicit.defvjp(_newton_fwd, _newton_bwd)

# Optional JIT wrapper (residual_fn static)
#newton_implicit_jit = jax.jit(newton_implicit, static_argnums=(0,))

# ----------------- dict/PyTree wrapper around array Newton -----------------
def newton_implicit_unravel(residual_fn_pytree, x0_tree, dyn_args, tol=1e-6, abs_tol=1e-8, max_iter=100):
    """
    residual_fn_pytree(x_tree, *dyn_args) -> residual_tree (same PyTree structure)
    x0_tree: PyTree (dicts, arrays…)
    Returns: (x_tree_solution, iters)
    """
    x0_flat, unravel_x = ravel_pytree(x0_tree)

    def res_flat(x_flat, *dyn):
        x_tree = unravel_x(x_flat)
        r_tree = residual_fn_pytree(x_tree, *dyn)
        r_flat, _ = ravel_pytree(r_tree)
        return r_flat

    x_fin_flat, iters = newton_implicit(res_flat, x0_flat, dyn_args, tol, abs_tol, max_iter)
    x_fin_tree = unravel_x(x_fin_flat)
    return x_fin_tree, iters
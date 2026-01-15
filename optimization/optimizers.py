import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from typing import Callable, Optional, Tuple, Dict, Any


class BFGSState(tuple):
    __slots__ = ()
    # x, f, g, H, k
    def __new__(cls, x, f, g, H, k): return tuple.__new__(cls, (x, f, g, H, k))

def bfgs(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    max_iter: Optional[int] = None,
    gtol: float = 1e-7,
    rtol: Optional[float] = None,   # stop when f <= rtol * f_init (if not None)
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    max_linesearch_evals: int = 20,
    n_display: Optional[int] = None,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """BFGS with Strong-Wolfe (backtrack+zoom), Powell damping; Python loops; returns (theta*, f*, info)."""

    loss_fg = jit(value_and_grad(loss_fn))
    d = int(theta0.size)
    if max_iter is None:
        max_iter = 200 * d
    I = jnp.eye(d, dtype=theta0.dtype)

    # Warmup + initial eval
    f, g = loss_fg(theta0)
    _ = f.block_until_ready(); _ = g.block_until_ready()
    st = BFGSState(theta0, f, g, I, 0)
    f_init = float(f)

    fevals = 1  # include initial call
    gevals = 1

    if n_display:
        print(f"[BFGS] it 0: f={float(st[1]):.6e}, ||g||={float(jnp.linalg.norm(st[2])):.3e}")

    for k in range(1, max_iter + 1):
        x, f, g, H, _ = st
        gn = float(jnp.linalg.norm(g))
        if gn <= gtol or (rtol is not None and float(f) <= rtol * f_init):
            break

        p = -H @ g
        dphi0 = float(g @ p)

        # ---- Strong-Wolfe line search (Python control flow) ----
        a_prev, f_prev, dphi_prev = 0.0, f, dphi0
        a = float(alpha0)
        f_a, g_a = loss_fg(x + a * p); fevals += 1; gevals += 1
        dphi_a = float(g_a @ p)

        ls_it = 1
        while ls_it < max_linesearch_evals:
            armijo_ok = float(f_a) <= float(f + c1 * a * dphi0)
            curve_ok  = abs(dphi_a) <= -c2 * dphi0
            if armijo_ok and curve_ok:
                break

            # Need a bracket? (Armijo fail or nondecrease)
            if (not armijo_ok) or (float(f_a) >= float(f_prev)):
                a_lo, f_lo, dphi_lo = a_prev, f_prev, dphi_prev
                a_hi, f_hi, dphi_hi = a,     f_a,   dphi_a
                for _ in range(max_linesearch_evals):
                    a_j = 0.5 * (a_lo + a_hi)
                    f_j, g_j = loss_fg(x + a_j * p); fevals += 1; gevals += 1
                    dphi_j = float(g_j @ p)
                    if (float(f_j) <= float(f + c1 * a_j * dphi0)) and (abs(dphi_j) <= -c2 * dphi0):
                        a, f_a, g_a, dphi_a = a_j, f_j, g_j, dphi_j
                        break
                    if (float(f_j) > float(f + c1 * a_j * dphi0)) or (float(f_j) >= float(f_lo)):
                        a_hi, f_hi, dphi_hi = a_j, f_j, dphi_j
                    else:
                        if dphi_j * (a_hi - a_lo) >= 0.0:
                            a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
                        a_lo, f_lo, dphi_lo = a_j, f_j, dphi_j
                else:
                    a = 0.5 * (a_lo + a_hi)
                    f_a, g_a = loss_fg(x + a * p); fevals += 1; gevals += 1
                    dphi_a = float(g_a @ p)
                break
            else:
                # derivative still negative but too large (|dphi| big) -> expand
                if dphi_a >= 0.0:
                    # bracket crossed; zoom between [a, a_prev]
                    a_lo, f_lo, dphi_lo = a, f_a, dphi_a
                    a_hi, f_hi, dphi_hi = a_prev, f_prev, dphi_prev
                    for _ in range(max_linesearch_evals):
                        a_j = 0.5 * (a_lo + a_hi)
                        f_j, g_j = loss_fg(x + a_j * p); fevals += 1; gevals += 1
                        dphi_j = float(g_j @ p)
                        if (float(f_j) <= float(f + c1 * a_j * dphi0)) and (abs(dphi_j) <= -c2 * dphi0):
                            a, f_a, g_a, dphi_a = a_j, f_j, g_j, dphi_j
                            break
                        if (float(f_j) > float(f + c1 * a_j * dphi0)) or (float(f_j) >= float(f_lo)):
                            a_hi, f_hi, dphi_hi = a_j, f_j, dphi_j
                        else:
                            if dphi_j * (a_hi - a_lo) >= 0.0:
                                a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
                            a_lo, f_lo, dphi_lo = a_j, f_j, dphi_j
                    else:
                        a = 0.5 * (a_lo + a_hi)
                        f_a, g_a = loss_fg(x + a * p); fevals += 1; gevals += 1
                        dphi_a = float(g_a @ p)
                    break
                else:
                    a_prev, f_prev, dphi_prev = a, f_a, dphi_a
                    a = 2.0 * a
                    f_a, g_a = loss_fg(x + a * p); fevals += 1; gevals += 1
                    dphi_a = float(g_a @ p)
            ls_it += 1

        # ---- BFGS update (Powell-damped) ----
        s = a * p
        x1 = x + s
        y = g_a - g
        ys = float(y @ s)

        sHs  = float(s @ (H @ s))
        theta = 1.0 if ys >= 0.2 * sHs else (0.8 * sHs) / (sHs - ys + 1e-16)
        y_bar = theta * y + (1.0 - theta) * (H @ s)
        yb_s  = float(y_bar @ s)
        if yb_s > 1e-14:
            rho = 1.0 / (yb_s + 1e-16)
            Hyb = H @ y_bar
            ybHy = float(y_bar @ Hyb)
            H = H - (jnp.outer(Hyb, s) + jnp.outer(s, Hyb)) * rho + ((1.0 + ybHy * rho) * rho) * jnp.outer(s, s)

        st = BFGSState(x1, f_a, g_a, H, k)

        if n_display and (k % n_display == 0):
            print(f"[BFGS] it {k}: f={float(f_a):.6e}, ||g||={float(jnp.linalg.norm(g_a)):.3e}, "
                  f"alpha={a:.3e}, fe={fevals}, ge={gevals}")

    if n_display:
        print(f"[BFGS] done at it {st[4]}: f={float(st[1]):.6e}, ||g||={float(jnp.linalg.norm(st[2])):.3e}, "
              f"fe={fevals}, ge={gevals}")

    info = {"n_fval": fevals, "n_gval": gevals, "iters": int(st[4])}
    return st[0], float(st[1]), info



def fd_grad(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    *,
    eps: float = 1e-6,
    method: str = "central",   # "central" or "forward"
) -> jnp.ndarray:
    """Finite-difference gradient. No autodiff."""
    x = jnp.asarray(x)
    d = int(x.size)
    g = []
    fx = loss_fn(x)

    for i in range(d):
        xi = x[i]
        h = eps * jnp.maximum(1.0, jnp.abs(xi))

        if method == "central":
            f1 = loss_fn(x.at[i].set(xi + h))
            f2 = loss_fn(x.at[i].set(xi - h))
            gi = (f1 - f2) / (2.0 * h)
        elif method == "forward":
            f1 = loss_fn(x.at[i].set(xi + h))
            gi = (f1 - fx) / h
        else:
            raise ValueError("method must be 'central' or 'forward'")

        g.append(gi)

    g = jnp.stack(g).reshape(x.shape)
    return g


def bfgs_fd(
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    *,
    max_iter: Optional[int] = None,
    gtol: float = 1e-7,
    rtol: Optional[float] = None,
    c1: float = 1e-4,
    c2: float = 0.9,
    alpha0: float = 1.0,
    max_linesearch_evals: int = 20,
    n_display: Optional[int] = None,
    # FD controls
    fd_eps: float = 1e-6,
    fd_method: str = "central",
    # Optional: if you have your own analytic gradient (still no autodiff)
    grad_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    # Optional jit for loss evaluation only (still no autodiff)
    jit_loss: bool = True,
) -> Tuple[jnp.ndarray, float, Dict[str, Any]]:
    """BFGS + Strong-Wolfe, using finite-difference gradients (no autodiff)."""

    loss_eval = jit(loss_fn) if jit_loss else loss_fn

    def grad_eval(x):
        return grad_fn(x) if grad_fn is not None else fd_grad(loss_eval, x, eps=fd_eps, method=fd_method)

    x0 = jnp.asarray(theta0)
    d = int(x0.size)
    if max_iter is None:
        max_iter = 200 * d

    I = jnp.eye(d, dtype=x0.dtype)

    # initial
    f = loss_eval(x0)
    g = grad_eval(x0)

    if not jnp.isfinite(f):
        raise FloatingPointError("Initial loss is not finite.")
    if not jnp.all(jnp.isfinite(g)):
        raise FloatingPointError("Initial finite-difference gradient has NaN/inf. Try different fd_eps.")

    st = BFGSState(x0, f, g, I, 0)
    f_init = float(f)

    fevals = 1
    gevals = 1 if grad_fn is not None else (1 + (2*d if fd_method == "central" else d))  # rough count

    if n_display:
        print(f"[BFGS-FD] it 0: f={float(f):.6e}, ||g||={float(jnp.linalg.norm(g)):.3e}")

    for k in range(1, max_iter + 1):
        x, f, g, H, _ = st
        gn = float(jnp.linalg.norm(g))
        if gn <= gtol or (rtol is not None and float(f) <= rtol * f_init):
            break

        p = -H @ g
        dphi0 = float(g @ p)

        # if not a descent direction (FD noise), reset H
        if dphi0 >= 0.0:
            H = I
            p = -g
            dphi0 = float(g @ p)

        # ---- Strong-Wolfe line search ----
        a_prev, f_prev, dphi_prev = 0.0, f, dphi0
        a = float(alpha0)

        x_a = x + a * p
        f_a = loss_eval(x_a); fevals += 1
        g_a = grad_eval(x_a); gevals += 1
        dphi_a = float(g_a @ p)

        ls_it = 1
        while ls_it < max_linesearch_evals:
            armijo_ok = float(f_a) <= float(f + c1 * a * dphi0)
            curve_ok  = abs(dphi_a) <= -c2 * dphi0
            if armijo_ok and curve_ok:
                break

            if (not armijo_ok) or (float(f_a) >= float(f_prev)):
                # zoom between [a_prev, a]
                a_lo, f_lo, dphi_lo = a_prev, f_prev, dphi_prev
                a_hi, f_hi, dphi_hi = a,     f_a,   dphi_a
                for _ in range(max_linesearch_evals):
                    a_j = 0.5 * (a_lo + a_hi)
                    x_j = x + a_j * p
                    f_j = loss_eval(x_j); fevals += 1
                    g_j = grad_eval(x_j); gevals += 1
                    dphi_j = float(g_j @ p)

                    if (float(f_j) <= float(f + c1 * a_j * dphi0)) and (abs(dphi_j) <= -c2 * dphi0):
                        a, f_a, g_a, dphi_a = a_j, f_j, g_j, dphi_j
                        break

                    if (float(f_j) > float(f + c1 * a_j * dphi0)) or (float(f_j) >= float(f_lo)):
                        a_hi, f_hi, dphi_hi = a_j, f_j, dphi_j
                    else:
                        if dphi_j * (a_hi - a_lo) >= 0.0:
                            a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
                        a_lo, f_lo, dphi_lo = a_j, f_j, dphi_j
                else:
                    a = 0.5 * (a_lo + a_hi)
                    x_a = x + a * p
                    f_a = loss_eval(x_a); fevals += 1
                    g_a = grad_eval(x_a); gevals += 1
                    dphi_a = float(g_a @ p)
                break

            else:
                if dphi_a >= 0.0:
                    # zoom between [a, a_prev]
                    a_lo, f_lo, dphi_lo = a, f_a, dphi_a
                    a_hi, f_hi, dphi_hi = a_prev, f_prev, dphi_prev
                    for _ in range(max_linesearch_evals):
                        a_j = 0.5 * (a_lo + a_hi)
                        x_j = x + a_j * p
                        f_j = loss_eval(x_j); fevals += 1
                        g_j = grad_eval(x_j); gevals += 1
                        dphi_j = float(g_j @ p)

                        if (float(f_j) <= float(f + c1 * a_j * dphi0)) and (abs(dphi_j) <= -c2 * dphi0):
                            a, f_a, g_a, dphi_a = a_j, f_j, g_j, dphi_j
                            break

                        if (float(f_j) > float(f + c1 * a_j * dphi0)) or (float(f_j) >= float(f_lo)):
                            a_hi, f_hi, dphi_hi = a_j, f_j, dphi_j
                        else:
                            if dphi_j * (a_hi - a_lo) >= 0.0:
                                a_hi, f_hi, dphi_hi = a_lo, f_lo, dphi_lo
                            a_lo, f_lo, dphi_lo = a_j, f_j, dphi_j
                    else:
                        a = 0.5 * (a_lo + a_hi)
                        x_a = x + a * p
                        f_a = loss_eval(x_a); fevals += 1
                        g_a = grad_eval(x_a); gevals += 1
                        dphi_a = float(g_a @ p)
                    break
                else:
                    a_prev, f_prev, dphi_prev = a, f_a, dphi_a
                    a = 2.0 * a
                    x_a = x + a * p
                    f_a = loss_eval(x_a); fevals += 1
                    g_a = grad_eval(x_a); gevals += 1
                    dphi_a = float(g_a @ p)

            ls_it += 1

        # ---- BFGS update (Powell damping) ----
        s = a * p
        x1 = x + s
        y = g_a - g
        ys = float(y @ s)

        sHs = float(s @ (H @ s))
        theta = 1.0 if ys >= 0.2 * sHs else (0.8 * sHs) / (sHs - ys + 1e-16)
        y_bar = theta * y + (1.0 - theta) * (H @ s)
        yb_s = float(y_bar @ s)

        if yb_s > 1e-14:
            rho = 1.0 / (yb_s + 1e-16)
            Hyb = H @ y_bar
            ybHy = float(y_bar @ Hyb)
            H = H - (jnp.outer(Hyb, s) + jnp.outer(s, Hyb)) * rho + ((1.0 + ybHy * rho) * rho) * jnp.outer(s, s)
        else:
            # FD noise can break curvature; reset
            H = I

        st = BFGSState(x1, f_a, g_a, H, k)

        if n_display and (k % n_display == 0):
            print(f"[BFGS-FD] it {k}: f={float(f_a):.6e}, ||g||={float(jnp.linalg.norm(g_a)):.3e}, "
                  f"alpha={a:.3e}, fe={fevals}, ge={gevals}")

    if n_display:
        x, f, g, _, k = st
        print(f"[BFGS-FD] done at it {k}: f={float(f):.6e}, ||g||={float(jnp.linalg.norm(g)):.3e}, "
              f"fe={fevals}, ge={gevals}")

    info = {"n_fval": fevals, "n_gval": gevals, "iters": int(st[4])}
    return st[0], float(st[1]), info
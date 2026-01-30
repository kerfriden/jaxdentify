import os

import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax, jit
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve as la_solve

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_implicit_unravel
from optimization.optimizers import bfgs
from optimization.parameter_mappings import build_param_space, make_loss, to_params

from functools import partial



print(jax.devices())


# ----------------
# Test-mode control
# ----------------
MANUAL_TEST_MODE: bool | None = None
_FROM_PYTEST = os.environ.get("JAXDENTIFY_FROM_PYTEST", "0") == "1" or ("PYTEST_CURRENT_TEST" in os.environ)
if MANUAL_TEST_MODE is None:
    TEST_MODE = (os.environ.get("JAXDENTIFY_TEST", "0") == "1") and _FROM_PYTEST
else:
    TEST_MODE = bool(MANUAL_TEST_MODE)

try:
    import optimistix as optx  # type: ignore

    _HAS_OPTIMISTIX = True
except Exception:
    optx = None
    _HAS_OPTIMISTIX = False

def newton_optx(
    residual_fn,
    x0_tree,
    diff_args,       # pytree: receives gradients
    nondiff_args,    # pytree: NO gradients (stop_gradient applied)
    tol=1e-8,
    abs_tol=1e-12,
    max_iter=50,
):
    """
    Newton using Optimistix with two sets of arguments:
      - diff_args: differentiable (for BFGS/grad)
      - nondiff_args: not differentiable (strain history, old state, etc.)
    """

    if not _HAS_OPTIMISTIX:
        raise ModuleNotFoundError(
            "optimistix is not installed. Install it to use newton_optx, or run with the built-in solver."
        )

    # Freeze nondiff arguments so JAX never differentiates them
    nondiff_args_static = jax.tree.map(lax.stop_gradient, nondiff_args)

    # Pack into a single tuple for Optimistix
    all_args = (diff_args, nondiff_args_static)

    # Wrapper to unpack args correctly
    def fn(x, args):
        diff_args_, nondiff_args_ = args
        return residual_fn(x, diff_args_, nondiff_args_)

    solver = optx.Newton(rtol=tol, atol=abs_tol)

    sol = optx.root_find(
        fn,
        solver,
        x0_tree,
        args=all_args,
        max_steps=max_iter,
        throw=False
    )

    x_fin = sol.value
    iters = jnp.asarray(sol.stats["num_steps"], jnp.int32)
    return x_fin, iters


def C_iso_voigt(E, nu):
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam2 = lam + 2.0 * mu
    return jnp.array([
        [lam2, lam,  lam,  0., 0., 0.],
        [lam,  lam2, lam,  0., 0., 0.],
        [lam,  lam,  lam2, 0., 0., 0.],
        [0.,   0.,   0.,   mu, 0., 0.],
        [0.,   0.,   0.,   0., mu, 0.],
        [0.,   0.,   0.,   0., 0., mu],
    ])

# ----------------- constitutive update (pure function) -----------------
def R_iso(p, params):
    return params["Q"] * (1.0 - jnp.exp(-params["b"] * p))

def vm(sigma):
    s = dev_voigt(sigma)
    return jnp.sqrt(3.0/2.0) * norm_voigt(s)

def f_func(sigma, p, params):
    return vm(sigma) - (params["sigma_y"] + R_iso(p, params))

def Fischer_Burmeister(a,b):
    return jnp.sqrt(a**2+b**2)-a-b # ϕ(a,b)=0 ⟺ a≥0,b≥0,ab=0

def residuals(x, diff_args, nondiff_args):
    params = diff_args
    epsilon, state_old, sigma_idx = nondiff_args

    C = C_iso_voigt(params["E"], params["nu"])
    sigma, eps_p, p, gamma = x["sigma"], x["eps_p"], x["p"], x["gamma"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    X = x["X"]
    X_old = state_old["X"]

    epsilon_eff = epsilon.at[sigma_idx].set(x["eps_cstr"])

    res_sigma = sigma - C @ (epsilon_eff - eps_p)

    df_dsigma = jax.grad(lambda s: f_func(s-X, p, params))(sigma)

    res_epsp = (eps_p - eps_p_old) - gamma * df_dsigma
    res_p    = (p - p_old) - gamma

    res_cstr = sigma[sigma_idx]

    res_gamma = Fischer_Burmeister( -f_func(sigma-X, p, params) , p-p_old )

    res_X = (X - X_old) - ( 2./3. * params['C_kin'] * (eps_p-eps_p_old) - params['D_kin'] * X * (p-p_old) )

    return {
        "res_sigma":  res_sigma,
        "res_epsp":   res_epsp,
        "res_p":      res_p,
        "res_X":      res_X,
        "res_gamma":  res_gamma,
        "res_cstr":   res_cstr,
    }

def solve_eps_cstr(epsilon,eps_p_old,sigma_idx,params):
    C = C_iso_voigt(params["E"], params["nu"])
    A = C[sigma_idx][:, sigma_idx]                                                 # (k,k)
    r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    eps_cstr_trial = epsilon[sigma_idx] - dz                                               
    return epsilon.at[sigma_idx].set(eps_cstr_trial) , eps_cstr_trial

def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):

    C = C_iso_voigt(params["E"], params["nu"])
    eps_p_old, p_old, X_old = state_old["epsilon_p"], state_old["p"], state_old["X"]
    dtype = C.dtype

    epsilon   = step_load["epsilon"]
    sigma_idx = step_load.get("sigma_cstr_idx")

    # ---------- 1) Constrained elastic trial (γ=0, εp=εp_old) ----------
    # Solve for z = eps_cstr_trial so that (C @ (epsilon_eff - eps_p_old))[sigma_idx] = 0
    # A @ (z - epsilon[eps_idx]) = - r
    #A = C[sigma_idx][:, sigma_idx]                                                 # (k,k)
    #r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    #dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    #z  = epsilon[sigma_idx] - dz                                                   # eps_cstr_trial
    #epsilon_eff_trial = epsilon.at[sigma_idx].set(z)
    #sigma_trial       = C @ (epsilon_eff_trial - eps_p_old)

    epsilon_eff_trial , eps_cstr_trial = solve_eps_cstr(epsilon,eps_p_old,sigma_idx,params)
    sigma_trial       = C @ (epsilon_eff_trial - eps_p_old)

    # yield function at trial
    #f_trial = f_func(sigma_trial-X_old, p_old, params)

    x0 = {
        "sigma":    sigma_trial,
        "eps_p":    eps_p_old,
        "p":        jnp.asarray(p_old, dtype=dtype),
        "X":        jnp.asarray(X_old, dtype=dtype),
        "gamma":    jnp.asarray(0.0, dtype=dtype),
        "eps_cstr": jnp.asarray(eps_cstr_trial, dtype=dtype),  # good initial guess
    }
    nondiff_args = (epsilon, state_old, sigma_idx)
    diff_args = params

    use_optx = (_HAS_OPTIMISTIX and (not TEST_MODE))
    if use_optx:
        x_sol, iters = newton_optx(
            residuals,
            x0,
            diff_args,
            nondiff_args,
            tol=alg["tol"],
            abs_tol=alg["abs_tol"],
            max_iter=alg["max_it"],
        )
    else:
        x_sol, iters = newton_implicit_unravel(
            residuals,
            x0,
            (diff_args, nondiff_args),
            tol=alg["tol"],
            abs_tol=alg["abs_tol"],
            max_iter=alg["max_it"],
        )
    new_state = {"epsilon_p": x_sol["eps_p"], "p": x_sol["p"], "X": x_sol["X"]}
    fields    = {"sigma": x_sol["sigma"]}
    logs      = {"conv": jnp.asarray(iters),
                    "eps_cstr": x_sol["eps_cstr"]}
    
    return new_state, fields, logs


print("--------------------")
print("reference simulation")
print("--------------------")

params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "C_kin": 0.25 ,
    "D_kin": 1.0 ,
    "E": 1.0,
    "nu": 0.3
}

# strain history
n_ts = 20 if TEST_MODE else 100
ts = jnp.linspace(0., 1., n_ts)
eps_xx = 4.0 * jnp.sin( ts * 30.0)
epsilon_ts = (jnp.zeros((len(ts), 6))
              .at[:, 0].set(eps_xx))


sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])

load_ts = {
    "epsilon": epsilon_ts,
    "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])) ,
}

state0 = {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0),"X": jnp.zeros(6,)}
state_T, saved = simulate(constitutive_update_fn,state0, load_ts, params)

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,saved["fields"]["sigma"][:,0])
plt.grid()
plt.xlabel(r"$F_{11}$")
plt.ylabel(r"$\sigma_{11}$")

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/plasticity_optx_reference.png", dpi=150, bbox_inches="tight")
if not TEST_MODE:
    plt.show()
plt.close()
print("Saved plots/plasticity_optx_reference.png")

print(saved["logs"]["conv"])

# save reference solution for inverse problem
true_params = params.copy()
sigma_xx_obs = saved["fields"]["sigma"][:, 0]


print("-----------------------------------------------")
print("active/frozen parameter ranges and distribution")
print("-----------------------------------------------")


# --- declare frozen + active (bounds + scale). Omit values in init_params to use interval midpoints ---
init_params = {
    "E": 1.0, "nu": 0.3, "sigma_y": 1.0,  # frozen values supplied
    # "Q": (omitted -> defaults to geom. mean of bounds)
    # "b": (omitted -> defaults to geom. mean of bounds)
    "C_kin": 0.25, "D_kin": 1.0,          # frozen for this example
}

active_specs = {
    "E": False, "nu": False, "sigma_y": False, "C_kin": False, "D_kin": False,
    "Q": {"lower": 1.e-3, "upper": 1.e+2, "scale": "log", "mask": None},
    "b": {"lower": 1.e-3, "upper": 1.e+0, "scale": "log", "mask": None},
}

space, theta0 = build_param_space(init_params, active_specs)


print("-----------------")
print("user-defined loss")
print("-----------------")


def forward_sigma11(params):
    state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0), "X": jnp.zeros(6)}
    _, saved = simulate(constitutive_update_fn, state0, load_ts, params)
    return saved["fields"]["sigma"][:, 0]

def simulate_and_loss(params):
    pred = forward_sigma11(params)
    r = pred - sigma_xx_obs
    return 0.5 * jnp.mean(r * r)

loss = make_loss(space,simulate_and_loss)



print("-------------")
print("run optimizer")
print("-------------")

init = to_params(space, theta0)
print("Initial Q, b:", init["Q"], init["b"])

# --- run BFGS (optionally seed with a few Adam steps you have) ---
t0 = time.perf_counter()
theta_opt, fval, info = bfgs(loss, theta0, rtol=1.e-3, n_display=1)
t1 = time.perf_counter()
print("time for optimizaton:", (t1 - t0), "s")
print("final loss:", fval)
print("info",info)

# --- unpack physical identified parameters ---
identified = to_params(space, theta_opt)
print("Identified Q, b:", identified["Q"], identified["b"])
# Optional: get fitted curve

print("-----------")
print("plot result")
print("-----------")


sigma_fit = forward_sigma11(identified)
sigma_init = forward_sigma11(init)

plt.plot(load_ts['epsilon'][:,0],sigma_fit,'blue',label=r'$\hat{\sigma}_{11}$ (fit)')
plt.plot(load_ts['epsilon'][:,0],sigma_xx_obs,'black',label=r'$\hat{\sigma}_{11}$ (data)')
plt.plot(load_ts['epsilon'][:,0],sigma_init,'green',label=r'$\hat{\sigma}_{11}$ (initial)')
plt.legend(loc='best')
plt.grid()
plt.savefig("plots/plasticity_optx_fit.png", dpi=150, bbox_inches="tight")
if not TEST_MODE:
    plt.show()
plt.close()
print("Saved plots/plasticity_optx_fit.png")

print("--------------")
print("test CPU times")
print("--------------")

v_ = loss(theta0).block_until_ready()
_ = jax.grad(loss)(theta0).block_until_ready()  # warm both paths

t0 = time.perf_counter()
f = loss(theta0).block_until_ready()
t1 = time.perf_counter()
print("one forward loss eval:", (t1 - t0) * 1e3, "ms")

t0 = time.perf_counter()
g = jax.grad(loss)(theta0).block_until_ready()
t1 = time.perf_counter()
print("one grad eval:", (t1 - t0) * 1e3, "ms")
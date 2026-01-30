import os

import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax, jit
from jax.scipy.linalg import solve as la_solve

jax.config.update("jax_platform_name", "cpu")   # force CPU
jax.config.update("jax_enable_x64", True)

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate_unpack
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_implicit_unravel
from simulation.newton import newton_optx

from optimization.optimizers import bfgs
from optimization.parameter_mappings import build_param_space, make_loss, to_params

from functools import partial

print("jax.devices()",jax.devices())


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
    import optimistix as _optx  # type: ignore

    _HAS_OPTIMISTIX = True
except Exception:
    _optx = None
    _HAS_OPTIMISTIX = False

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

    epsilon_eff_trial , eps_cstr_trial = solve_eps_cstr(epsilon,eps_p_old,sigma_idx,params)
    sigma_trial       = C @ (epsilon_eff_trial - eps_p_old)

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

    # Keep this example runnable under pytest without optional deps.
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
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

print("iteration count (first 100)",logs_ts["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/plasticity_vmap_reference.png", dpi=150, bbox_inches="tight")
if not TEST_MODE:
    plt.show()
plt.close()
print("Saved plots/plasticity_vmap_reference.png")

if TEST_MODE:
    print("TEST_MODE: skipping remaining batched/vmap sections.")
    raise SystemExit(0)







def plot_batch(load_ts, fields_ts_batch):
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract sigma batch
    sigma_batch = np.array(fields_ts_batch["sigma"])      # (B, n_ts, 6)
    B, n_ts, _ = sigma_batch.shape

    # Extract epsilon
    eps = np.array(load_ts["epsilon"])

    # If epsilon is batched: (B, n_ts, 6)
    if eps.ndim == 3:
        eps11_batch = eps[:, :, 0]                        # (B, n_ts)
    # If epsilon is single-path: (n_ts, 6)
    else:
        eps11 = eps[:, 0]                                 # (n_ts,)
        eps11_batch = np.broadcast_to(eps11, (B, n_ts))   # make B copies

    plt.figure()
    for i in range(B):
        plt.plot(eps11_batch[i], sigma_batch[i, :, 0])    # σ11 vs ε11 for batch i

    plt.grid(True)
    plt.xlabel(r"$\varepsilon_{11}$")
    plt.ylabel(r"$\sigma_{11}$")
    plt.show()
    

print("---------------------------------------------")
print("batched simulation case 1: several parameters")
print("---------------------------------------------")


# 1) Wrap simulate into a single-sample function
def run_one(params, load_ts, state0):
    # simulate(update_fn, state0, load_ts, params)
    state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)
    return state_T, fields_ts, state_ts, logs_ts

# 2) Vectorize over the first axis of params
batched_run = jax.vmap(run_one, in_axes=(0,  None,   None))  # params batched, load_ts & state0 shared

# 3) Build batched params (pytree of arrays with leading batch dim)
# Example: 3 different sigma_y values, everything else same
params_batch = {
    "sigma_y": jnp.array([0.8, 1.0, 1.2]),
    "Q":       jnp.array([1.0, 1.0, 1.0]),
    "b":       jnp.array([0.1, 0.1, 0.1]),
    "C_kin":   jnp.array([0.25, 0.25, 0.25]),
    "D_kin":   jnp.array([1.0,  1.0,  1.0]),
    "E":       jnp.array([1.0,  1.0,  1.0]),
    "nu":      jnp.array([0.3,  0.3,  0.3]),
}

# 4) Call the batched simulation
state0 = {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0), "X": jnp.zeros(6,)}

state_T_batch, fields_ts_batch, state_ts_batch, logs_ts_batch = batched_run(params_batch, load_ts, state0)

plot_batch(load_ts,fields_ts_batch)

print("----------------------------------------------")
print("batched simulation case 2: several init states")
print("----------------------------------------------")

batched_run = jax.vmap(
    run_one,
    in_axes=(None, 0, 0)   # params shared, load_ts batched, state0 batched
)

B = 4          # batch size
n_ts = ts.shape[0]

# One frequency per batch element (shape (B,))
freqs = jnp.array([20.0, 25, 30.0, 40.0])  # example
amps= jnp.array([4., 2., 5., 3.])  # example

# Build batched eps_xx: shape (B, n_ts)
# ts[None, :]      -> (1, n_ts)
# freqs[:, None]   -> (B, 1)
eps_xx_batch = amps[:, None] * jnp.sin(ts[None, :] * freqs[:, None])  # (B, n_ts)

# Now build epsilon_ts_batch: (B, n_ts, 6), put eps_xx_batch into the 11 component
epsilon_ts_batch = jnp.zeros((B, n_ts, 6))
epsilon_ts_batch = epsilon_ts_batch.at[:, :, 0].set(eps_xx_batch)

sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])
k = sigma_cstr_idx.size
sigma_cstr_idx_ts = jnp.broadcast_to(sigma_cstr_idx, (n_ts, k))      # (n_ts, k)
sigma_cstr_idx_batch = jnp.broadcast_to(sigma_cstr_idx_ts, (B, n_ts, k))

load_ts_batch = {
    "epsilon":        epsilon_ts_batch,      # (B, n_ts, 6)
    "sigma_cstr_idx": sigma_cstr_idx_batch,  # (B, n_ts, k)
}

state0_batch = {
    "epsilon_p": jnp.zeros((B, 6)),
    "p":         jnp.zeros((B,)),
    "X":         jnp.zeros((B, 6)),
}
params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "C_kin": 0.25,
    "D_kin": 1.0,
    "E": 1.0,
    "nu": 0.3,
}

state_T_batch, fields_ts_batch, state_ts_batch, logs_ts_batch = batched_run(params, load_ts_batch, state0_batch)

plot_batch(load_ts_batch,fields_ts_batch)
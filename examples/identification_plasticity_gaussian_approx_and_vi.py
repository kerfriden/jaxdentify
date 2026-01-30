import os
import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random
from jax import config

config.update("jax_enable_x64", True)

from simulation.simulate import simulate
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_implicit_unravel
from optimization.parameter_mappings import build_param_space, make_loss, to_theta
from optimization.postprocess import theta_to_params_samples, posterior_param_summary
from optimization.targets import as_logpi
from optimization.preconditioning import laplace_gaussian
from optimization.vi_flow import fit_gaussian_vi, sample_gaussian_vi, sample_gaussian_vi_full
from jax.scipy.linalg import solve as la_solve


def C_iso_voigt(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam2 = lam + 2.0 * mu
    return jnp.array(
        [
            [lam2, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam2, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ]
    )


def R_iso(p, params):
    return params["Q"] * (1.0 - jnp.exp(-params["b"] * p))


def vm(sigma):
    s = dev_voigt(sigma)
    return jnp.sqrt(3.0 / 2.0) * norm_voigt(s)


def f_func(sigma, p, params):
    return vm(sigma) - (params["sigma_y"] + R_iso(p, params))


def Fischer_Burmeister(a, b):
    return jnp.sqrt(a**2 + b**2) - a - b


def residuals(x, epsilon, state_old, params, sigma_idx):
    C = C_iso_voigt(params["E"], params["nu"])
    sigma, eps_p, p, gamma = x["sigma"], x["eps_p"], x["p"], x["gamma"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    X = x["X"]
    X_old = state_old["X"]

    epsilon_eff = epsilon.at[sigma_idx].set(x["eps_cstr"])

    res_sigma = sigma - C @ (epsilon_eff - eps_p)

    df_dsigma = jax.grad(lambda s: f_func(s - X, p, params))(sigma)

    res_epsp = (eps_p - eps_p_old) - gamma * df_dsigma
    res_p = (p - p_old) - gamma

    res_cstr = sigma[sigma_idx]

    res_gamma = Fischer_Burmeister(-f_func(sigma - X, p, params), p - p_old)

    res_X = (X - X_old) - (
        2.0 / 3.0 * params["C_kin"] * (eps_p - eps_p_old) - params["D_kin"] * X * (p - p_old)
    )

    return {
        "res_sigma": res_sigma,
        "res_epsp": res_epsp,
        "res_p": res_p,
        "res_X": res_X,
        "res_gamma": res_gamma,
        "res_cstr": res_cstr,
    }


def constitutive_update_fn(state_old, step_load, params, alg=None):
    if alg is None:
        alg = {"tol": 1e-8, "abs_tol": 1e-12, "max_it": 100}

    C = C_iso_voigt(params["E"], params["nu"])
    eps_p_old, p_old, X_old = state_old["epsilon_p"], state_old["p"], state_old["X"]
    dtype = C.dtype

    epsilon = step_load["epsilon"]
    sigma_idx = step_load.get("sigma_cstr_idx")

    A = C[sigma_idx][:, sigma_idx]
    r = (C @ (epsilon - eps_p_old))[sigma_idx]
    dz = la_solve(A, r, assume_a="gen")
    z = epsilon[sigma_idx] - dz

    epsilon_eff_trial = epsilon.at[sigma_idx].set(z)
    sigma_trial = C @ (epsilon_eff_trial - eps_p_old)

    x0 = {
        "sigma": sigma_trial,
        "eps_p": eps_p_old,
        "p": jnp.asarray(p_old, dtype=dtype),
        "X": jnp.asarray(X_old, dtype=dtype),
        "gamma": jnp.asarray(0.0, dtype=dtype),
        "eps_cstr": jnp.asarray(z, dtype=dtype),
    }

    x_sol, iters = newton_implicit_unravel(
        residuals,
        x0,
        (epsilon, state_old, params, sigma_idx),
        tol=alg["tol"],
        abs_tol=alg["abs_tol"],
        max_iter=alg["max_it"],
    )
    new_state = {"epsilon_p": x_sol["eps_p"], "p": x_sol["p"], "X": x_sol["X"]}
    fields = {"sigma": x_sol["sigma"]}
    logs = {"conv": jnp.asarray(iters), "eps_cstr": x_sol["eps_cstr"]}

    return new_state, fields, logs


# ----------------
# Test-mode control
# ----------------
MANUAL_TEST_MODE: bool | None = None
_FROM_PYTEST = os.environ.get("JAXDENTIFY_FROM_PYTEST", "0") == "1" or ("PYTEST_CURRENT_TEST" in os.environ)
if MANUAL_TEST_MODE is None:
    TEST_MODE = (os.environ.get("JAXDENTIFY_TEST", "0") == "1") and _FROM_PYTEST
else:
    TEST_MODE = bool(MANUAL_TEST_MODE)

# Helpful diagnostics: users sometimes run this from a terminal session where
# pytest-related env vars are still set.
if MANUAL_TEST_MODE is None:
    if (os.environ.get("JAXDENTIFY_TEST", "0") == "1") and (os.environ.get("JAXDENTIFY_FROM_PYTEST", "0") == "1") and ("PYTEST_CURRENT_TEST" not in os.environ):
        print(
            "NOTE: TEST_MODE is enabled because Env:JAXDENTIFY_TEST=1 and Env:JAXDENTIFY_FROM_PYTEST=1 are set. "
            "For a full manual run either unset those env vars or set MANUAL_TEST_MODE=False in this file."
        )
print(
    f"TEST_MODE={TEST_MODE} (MANUAL_TEST_MODE={MANUAL_TEST_MODE}, "
    f"JAXDENTIFY_TEST={os.environ.get('JAXDENTIFY_TEST','0')}, "
    f"JAXDENTIFY_FROM_PYTEST={os.environ.get('JAXDENTIFY_FROM_PYTEST','0')}, "
    f"PYTEST_CURRENT_TEST={'PYTEST_CURRENT_TEST' in os.environ})"
)


print("--------------------")
print("reference simulation")
print("--------------------")

params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "C_kin": 0.25,
    "D_kin": 1.0,
    "E": 1.0,
    "nu": 0.3,
}

n_ts = 20 if TEST_MODE else 100

# strain history
ts = jnp.linspace(0.0, 1.0, n_ts)
eps_xx = 4.0 * jnp.sin(ts * 30.0)
epsilon_ts = jnp.zeros((len(ts), 6)).at[:, 0].set(eps_xx)

sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])
load_ts = {
    "epsilon": epsilon_ts,
    "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])),
}

state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0), "X": jnp.zeros(6)}

# warmup run
state_T, saved = simulate(constitutive_update_fn, state0, load_ts, params)
jax.block_until_ready(saved["fields"]["sigma"])

# save reference plot
os.makedirs("plots", exist_ok=True)
eps11 = jnp.array(load_ts["epsilon"][:, 0])
plt.plot(eps11, saved["fields"]["sigma"][:, 0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.title("Reference simulation")
plt.savefig("plots/plasticity_gaussian_reference.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/plasticity_gaussian_reference.png")

true_params = params.copy()
sigma_xx_obs = saved["fields"]["sigma"][:, 0]


print("-----------------------------------------------")
print("active/frozen parameter ranges and distribution")
print("-----------------------------------------------")

init_params = {
    "E": 1.0,
    "nu": 0.3,
    "sigma_y": 1.0,
    "C_kin": 0.25,
    "D_kin": 1.0,
}

active_specs = {
    "E": False,
    "nu": False,
    "sigma_y": False,
    "C_kin": False,
    "D_kin": False,
    "Q": {"lower": 1.0e-3, "upper": 1.0e2, "scale": "log", "mask": None},
    "b": {"lower": 1.0e-3, "upper": 1.0e0, "scale": "log", "mask": None},
}

space, theta0 = build_param_space(init_params, active_specs)
theta_true = to_theta(space, true_params)


def forward_sigma11(params_dict):
    state0_local = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0), "X": jnp.zeros(6)}
    _, saved_local = simulate(constitutive_update_fn, state0_local, load_ts, params_dict)
    return saved_local["fields"]["sigma"][:, 0]


def simulate_and_loss(params_dict):
    pred = forward_sigma11(params_dict)
    r = pred - sigma_xx_obs
    return 0.5 * jnp.mean(r * r)


loss = make_loss(space, simulate_and_loss)


def loglik(theta):
    noise_std = 5.0e-2
    return -(0.5 / (noise_std**2)) * loss(theta)


logpi = as_logpi(loglik_theta=loglik)

d = int(theta0.shape[0])
print(f"Parameter dimension: {d}")

# ----------------------------
# Solver 1: Laplace (Gaussian)
# ----------------------------
print("\n----------------------------")
print("Laplace Gaussian approximation")
print("----------------------------")

key = random.PRNGKey(0)

if TEST_MODE:
    # Keep this very light under pytest.
    laplace_mode = "diag"
    map_max_iter = 5
    map_gtol = 1e-3
    map_print_every = None
else:
    laplace_mode = "full"
    map_max_iter = None
    map_gtol = 1e-7
    map_print_every = 1

# Use BFGS for Laplace MAP by default (robust for this problem)
g, lap_info = laplace_gaussian(
    logpi,
    theta0,
    map_method="bfgs",
    mode=laplace_mode,
    jitter=1e-6,
    map_max_iter=map_max_iter,
    map_gtol=map_gtol,
    map_print_every=map_print_every,
)
print("Laplace MAP theta:", np.asarray(g["mean"]))
print("Laplace info:", lap_info)

lap_n = 200 if TEST_MODE else 2000
key, k_lap = random.split(key)
theta_lap = g.sample(k_lap, n=lap_n)

# ----------------------------
# Solver 2: Gaussian VI
# ----------------------------
print("\n----------------------------")
print("Gaussian VI")
print("----------------------------")

# Diagonal (mean-field) Gaussian VI cannot represent correlations.
# For this (Q, b) identification problem the posterior is often close to Gaussian *with*
# correlation, so we use a full-covariance Gaussian VI by default.
VI_FULL_COV = True

vi_n_iters = 5 if TEST_MODE else 1000
vi_n_samples = 2 if TEST_MODE else 4
vi_lr = 0.01

# Initialize VI at Laplace mean/diag-scale (often helps)
mu0 = g["mean"]
diag = jnp.clip(jnp.diag(g["cov"]), a_min=1e-12)
log_sigma0 = 0.5 * jnp.log(diag)

key, k_vi = random.split(key)
if TEST_MODE:
    # Under pytest, keep runtime predictable: differentiating through the
    # implicit solver + simulation is extremely compilation-heavy.
    mu, log_sigma = mu0, log_sigma0
    elbo_hist = jnp.asarray([])
    print("TEST_MODE: skipping Gaussian VI optimization; using Laplace-initialized Gaussian.")
else:
    # Performance note:
    # - The Gaussian-VI optimizer runs its per-iteration update as a single `jax.jit`'d step
    #   (so Python overhead is removed after first compilation).
    # - Inside the ELBO, `logpi(theta_samples)` is evaluated with `lax.map` instead of `vmap`.
    #   For implicit solvers / Newton iterations, `vmap` can trigger pathological slowdowns
    #   by vectorizing control-flow and linear solves.
    if VI_FULL_COV:
        mu, chol, elbo_hist = fit_gaussian_vi(
            logpi,
            d,
            k_vi,
            cov="full",
            n_iters=vi_n_iters,
            n_samples=vi_n_samples,
            lr=vi_lr,
            verbose=not TEST_MODE,
            mu0=mu0,
            #chol0=g["chol"],
            chol0=None
        )
    else:
        mu, log_sigma, elbo_hist = fit_gaussian_vi(
            logpi,
            d,
            k_vi,
            cov="mean-field",
            n_iters=vi_n_iters,
            n_samples=vi_n_samples,
            lr=vi_lr,
            verbose=not TEST_MODE,
            mu0=mu0,
            log_sigma0=log_sigma0,
        )

vi_n = 200 if TEST_MODE else 2000
key, k_samp = random.split(key)
if TEST_MODE:
    theta_vi = sample_gaussian_vi(mu, log_sigma, k_samp, n_samples=vi_n)
else:
    theta_vi = sample_gaussian_vi_full(mu, chol, k_samp, n_samples=vi_n) if VI_FULL_COV else sample_gaussian_vi(mu, log_sigma, k_samp, n_samples=vi_n)

# ----------------------------
# Compare in parameter space
# ----------------------------
params_lap = theta_to_params_samples(space, theta_lap)
params_vi = theta_to_params_samples(space, theta_vi)

Q_lap = np.asarray(jax.device_get(params_lap["Q"]).ravel())
b_lap = np.asarray(jax.device_get(params_lap["b"]).ravel())

Q_vi = np.asarray(jax.device_get(params_vi["Q"]).ravel())
b_vi = np.asarray(jax.device_get(params_vi["b"]).ravel())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(Q_lap, b_lap, s=10, alpha=0.6)
plt.axvline(true_params["Q"], color="r", linestyle="--")
plt.axhline(true_params["b"], color="r", linestyle="--")
plt.grid(True)
plt.xlabel("Q")
plt.ylabel("b")
plt.title("Laplace Gaussian")

plt.subplot(1, 2, 2)
plt.scatter(Q_vi, b_vi, s=10, alpha=0.6)
plt.axvline(true_params["Q"], color="r", linestyle="--")
plt.axhline(true_params["b"], color="r", linestyle="--")
plt.grid(True)
plt.xlabel("Q")
plt.ylabel("b")
plt.title("Gaussian VI")

plt.tight_layout()
plt.savefig("plots/plasticity_gaussian_laplace_vs_gvi.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/plasticity_gaussian_laplace_vs_gvi.png")

# Summaries (parameter space)
summ_lap = posterior_param_summary(theta_lap, space)
summ_vi = posterior_param_summary(theta_vi, space)

print("\nSummary (Laplace):")
print(f"  b: mean={summ_lap['b'][0]:.4f}, p05={summ_lap['b'][1]:.4f}, p95={summ_lap['b'][2]:.4f}")
print(f"  Q: mean={summ_lap['Q'][0]:.4f}, p05={summ_lap['Q'][1]:.4f}, p95={summ_lap['Q'][2]:.4f}")

print("\nSummary (Gaussian VI):")
print(f"  b: mean={summ_vi['b'][0]:.4f}, p05={summ_vi['b'][1]:.4f}, p95={summ_vi['b'][2]:.4f}")
print(f"  Q: mean={summ_vi['Q'][0]:.4f}, p05={summ_vi['Q'][1]:.4f}, p95={summ_vi['Q'][2]:.4f}")

print("\nGaussian approximation + Gaussian VI completed!")

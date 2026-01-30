import os
import jax
import jax.numpy as jnp
from jax import lax, jit, random
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve as la_solve

import time
import matplotlib.pyplot as plt
import numpy as np

from simulation.simulate import simulate
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_implicit_unravel
from optimization.parameter_mappings import build_param_space, make_loss, to_params, to_theta
from optimization.postprocess import theta_to_params_samples, posterior_param_summary, posterior_predictive
from optimization.vi_flow import make_flow_vi
from optimization.preconditioning import map_hessian_preconditioner

from functools import partial


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

# ----------------- constitutive residuals (dict in/out) -----------------
def Fischer_Burmeister(a,b):
    return jnp.sqrt(a**2+b**2)-a-b # ϕ(a,b)=0 ⟺ a≥0,b≥0,ab=0

def residuals(x, epsilon, state_old, params, sigma_idx):

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

def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):

    C = C_iso_voigt(params["E"], params["nu"])
    eps_p_old, p_old, X_old = state_old["epsilon_p"], state_old["p"], state_old["X"]
    dtype = C.dtype

    epsilon   = step_load["epsilon"]
    sigma_idx = step_load.get("sigma_cstr_idx")

    # ---------- 1) Constrained elastic trial (γ=0, εp=εp_old) ----------
    # Solve for z = eps_cstr_trial so that (C @ (epsilon_eff - eps_p_old))[sigma_idx] = 0
    # A @ (z - epsilon[eps_idx]) = - r
    A = C[sigma_idx][:, sigma_idx]                                                 # (k,k)
    r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    z  = epsilon[sigma_idx] - dz                                                   # eps_cstr_trial
    epsilon_eff_trial = epsilon.at[sigma_idx].set(z)
    sigma_trial       = C @ (epsilon_eff_trial - eps_p_old)

    # yield function at trial
    #f_trial = f_func(sigma_trial-X_old, p_old, params)

    x0 = {
        "sigma":    sigma_trial,
        "eps_p":    eps_p_old,
        "p":        jnp.asarray(p_old, dtype=dtype),
        "X":        jnp.asarray(X_old, dtype=dtype),
        "gamma":    jnp.asarray(0.0, dtype=dtype),
        "eps_cstr": jnp.asarray(z, dtype=dtype),  # good initial guess
    }
    x_sol, iters = newton_implicit_unravel(
        residuals, x0, (epsilon, state_old, params, sigma_idx),
        tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
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
n_ts = 100
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

# First run (with JIT compilation)
t0 = time.perf_counter()
state_T, saved = simulate(constitutive_update_fn, state0, load_ts, params)
jax.block_until_ready(saved["fields"]["sigma"])
t1 = time.perf_counter()
print(f"First run (with JIT compilation): {t1-t0:.6f} seconds")

# Compiled steady-state timing (average a few runs to avoid coarse timer artifacts).
_n_ref = 5
_acc = 0.0
for _ in range(_n_ref):
    t0 = time.perf_counter()
    state_T, saved = simulate(constitutive_update_fn, state0, load_ts, params)
    jax.block_until_ready(saved["fields"]["sigma"])
    t1 = time.perf_counter()
    _acc += (t1 - t0)
sim_time = _acc / _n_ref
print(f"Second run (already compiled, avg of {_n_ref}): {sim_time:.6f} seconds")

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,saved["fields"]["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.title("Reference simulation")
plt.savefig('plots/plasticity_flow_vi_reference.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/plasticity_flow_vi_reference.png")

conv = np.asarray(saved["logs"]["conv"])
u, c = np.unique(conv, return_counts=True)
print("Newton convergence codes (value: count):", dict(zip(u.tolist(), c.tolist())))

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

# Normalized theta corresponding to the reference (true) physical parameters.
theta_true = to_theta(space, true_params)


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


print("------------------------")
print("Flow VI for posterior")
print("------------------------")

# Define log-likelihood with Gaussian prior (unit normal in normalized space)
def logpi(theta):
    """
    Log posterior = log likelihood + log prior
    Prior: Unit Gaussian in normalized theta space
    Likelihood: Gaussian noise model
    """
    # Prior: standard Gaussian in normalized space
    log_prior = -0.5 * jnp.sum(theta**2)
    
    # Likelihood: Gaussian noise model
    noise_std = 5.e-2
    log_lik = -(0.5 / (noise_std**2)) * loss(theta)
    
    return log_lik + log_prior


# Get dimensionality
d = len(theta0)
print(f"Parameter dimension: {d}")
print(f"Initial theta: {theta0}")

# Test mode control:
# - Set MANUAL_TEST_MODE=True/False to force behavior directly in this file.
# - Leave as None to let pytest control it (fast settings under tests only).
MANUAL_TEST_MODE: bool | None = False

_FROM_PYTEST = (
    os.environ.get("JAXDENTIFY_FROM_PYTEST", "0") == "1"
    or ("PYTEST_CURRENT_TEST" in os.environ)
)

if MANUAL_TEST_MODE is None:
    # Test mode is meant for pytest only (to keep CI/unit tests fast).
    # When running this script directly, we default to full settings.
    TEST_MODE = (os.environ.get("JAXDENTIFY_TEST", "0") == "1") and _FROM_PYTEST
else:
    TEST_MODE = bool(MANUAL_TEST_MODE)
PROFILE_MODE = os.environ.get("JAXDENTIFY_PROFILE", "0") == "1"

if MANUAL_TEST_MODE is None and os.environ.get("JAXDENTIFY_TEST", "0") == "1" and not _FROM_PYTEST:
    print("Note: ignoring JAXDENTIFY_TEST=1 because this run is not from pytest")

# Flow VI settings
n_layers = 4  # Reduced from 12 to test overhead
hidden_dim = 32  # Reduced from 64
s_cap = 2.2

# Optional: MAP-Hessian preconditioner as an outer affine transform for the flow.
# This can help training by centering/scaling the flow around a Laplace approx.
USE_PRECOND = False
precond = None
theta_map = None

if USE_PRECOND and not TEST_MODE:
    print("\nComputing MAP + Hessian preconditioner (for Flow VI base transform)...")
    theta_map, precond, info = map_hessian_preconditioner(
        logpi,
        theta0,
        mode="full",
        jitter=1e-6,
        map_method="bfgs",
        map_print_every=10,
    )
    print("MAP theta:", theta_map)
    print("Precond info:", info)
if TEST_MODE:
    n_iters = 5
    n_samples = 2
    lr = 0.002
    post_n_samples = 200
    do_predictive = False
    profile = PROFILE_MODE
    profile_n = 3 if PROFILE_MODE else 0
    print_every = 1
else:
    n_iters = 500  # Reduced from 500 - Flow VI is much slower than MALA
    n_samples = 4  # MC samples per ELBO evaluation (reduce for speed)
    lr = 0.002
    post_n_samples = 1000
    do_predictive = True
    profile = True
    profile_n = 3
    print_every = 50

print(f"\nFlow VI settings:")
print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}, s_cap={s_cap}")
print(f"  n_iters={n_iters}, n_samples={n_samples}, lr={lr}")

# ------------------------
# Reference timing (compiled)
# ------------------------
# The earlier `sim_time` is a forward solve at the reference parameters.
# For VI overhead attribution, also benchmark loss/logpi forward and value+grad.

def _block(x):
    return jax.tree.map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def _time_compiled(label, fn, *args, n_runs: int = 3):
    fn_jit = jax.jit(fn)
    # Compile + warmup.
    out = fn_jit(*args)
    _block(out)
    # Timed runs.
    t0 = time.perf_counter()
    for _ in range(int(n_runs)):
        out = fn_jit(*args)
    _block(out)
    t1 = time.perf_counter()
    dt = (t1 - t0) / max(1, int(n_runs))
    print(f"  {label:<28s}: {dt:.6f} s")
    return dt


def _time_compile_vs_steady(label, fn, *args, n_runs: int = 5):
    """Return (compile+first-run time, steady avg time) for a jitted fn."""
    fn_jit = jax.jit(fn)
    t0 = time.perf_counter()
    out = fn_jit(*args)
    _block(out)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(int(n_runs)):
        out = fn_jit(*args)
    _block(out)
    t3 = time.perf_counter()

    compile_dt = t1 - t0
    steady_dt = (t3 - t2) / max(1, int(n_runs))
    print(f"  {label:<28s}: compile {compile_dt:.6f} s | steady {steady_dt:.6f} s")
    return compile_dt, steady_dt


bench_runs = 2 if TEST_MODE else 5
print("\n[Reference timings: compiled averages]")
print(f"  sim_time (forward only)       : {sim_time:.6f} s")

theta_ref = jnp.asarray(theta_true)
theta_batch = random.normal(random.PRNGKey(123), shape=(int(n_samples), int(d)))
theta_true_batch = jnp.broadcast_to(theta_ref[None, :], (int(n_samples), int(d)))

vgrad_logpi = jax.value_and_grad(logpi)
_time_compile_vs_steady("logpi(theta_true)", logpi, theta_ref, n_runs=bench_runs)
_time_compile_vs_steady("value_and_grad(logpi)", vgrad_logpi, theta_ref, n_runs=bench_runs)

BENCH_EXTRA = os.environ.get("JAXDENTIFY_BENCH_EXTRA", "0") == "1"
if BENCH_EXTRA:
    loss_time = _time_compiled("loss(theta_true)", loss, theta_ref, n_runs=bench_runs)

    # Serial multi-eval baseline (inside one compiled function).
    def _serial_logpi_sum(t):
        def body(i, acc):
            return acc + logpi(t)

        return lax.fori_loop(0, int(n_samples), body, 0.0)

    _time_compiled(
        f"{int(n_samples)}x logpi serial (theta_true)",
        _serial_logpi_sum,
        theta_ref,
        n_runs=bench_runs,
    )

    print("  (batched baselines)")
    _time_compiled(
        f"lax.map(logpi) @ theta_true x{int(n_samples)}",
        lambda th: jax.lax.map(logpi, th),
        theta_true_batch,
        n_runs=bench_runs,
    )

    BENCH_VMAP = os.environ.get("JAXDENTIFY_BENCH_VMAP", "0") == "1"
    if BENCH_VMAP:
        _time_compiled(
            f"vmap(logpi) @ theta_true x{int(n_samples)}",
            lambda th: jax.vmap(logpi)(th),
            theta_true_batch,
            n_runs=1,
        )
        _time_compiled(
            f"vmap(logpi) @ random x{int(n_samples)}",
            lambda th: jax.vmap(logpi)(th),
            theta_batch,
            n_runs=1,
        )
        _time_compiled(
            f"vmap(value_and_grad) @ random x{int(n_samples)}",
            lambda th: jax.vmap(vgrad_logpi)(th),
            theta_batch,
            n_runs=1,
        )
    else:
        print("  (set JAXDENTIFY_BENCH_VMAP=1 to benchmark vmap(logpi))")

    if loss_time > 0:
        total_sims = n_iters * n_samples
        theoretical_loss_time = total_sims * loss_time
        print(
            f"  theoretical min using loss()  : {theoretical_loss_time:.6f} s "
            f"(for {int(total_sims)} loss evals)"
        )
else:
    print("  (set JAXDENTIFY_BENCH_EXTRA=1 for more baselines)")

# Create flow
flow_forward, flow_inverse, fit_flow, sample_flow = make_flow_vi(
    logpi,
    d,
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    s_cap=s_cap,
    use_random_perm=False,
    base_mean=(precond["mean"] if precond is not None else None),
    base_chol=(precond["chol"] if precond is not None else None),
)

# Fit flow
print("\nFitting Flow VI...")
print("Note: First iteration will be slow due to JIT compilation...")
print("Progress will be shown in real-time.\n")
key = random.PRNGKey(0)

t0 = time.time()
flow_params, elbo_history, vi_timing = fit_flow(
    key,
    n_iters=n_iters,
    n_samples=n_samples,
    lr=lr,
    verbose=True,
    print_every=print_every,
    profile=profile,
    profile_n=profile_n,
    return_info=True,
)

# Use internal VI timing to separate JIT+first-step from steady-state.
total_time = float(vi_timing["total"]) if isinstance(vi_timing, dict) else (time.time() - t0)
compile_time = float(vi_timing.get("compile_step", 0.0)) if isinstance(vi_timing, dict) else 0.0
steady_time = float(vi_timing.get("steady_total", 0.0)) if isinstance(vi_timing, dict) else 0.0

steady_iters = max(0, int(n_iters) - 1)
time_per_iter_steady = steady_time / steady_iters if steady_iters > 0 else float('nan')

total_sims = int(n_iters) * int(n_samples)
steady_sims = int(steady_iters) * int(n_samples)

theoretical_time_total = total_sims * sim_time
theoretical_time_steady = steady_sims * sim_time

print(f"\nTotal optimization time: {total_time:.6f} seconds")
print(f"  first-step (compile+run): {compile_time:.6f} seconds")
if steady_iters > 0:
    print(f"  steady-state: {steady_time:.6f} seconds ({time_per_iter_steady:.6f} s/iter)")
else:
    print(f"  steady-state: {steady_time:.6f} seconds")
print(f"\nBreakdown:")
print(f"  Total iterations: {n_iters}")
print(f"  Samples per iteration: {n_samples}")
print(f"  Total simulations: {total_sims}")
print(f"  Single simulation time: {sim_time:.6f} seconds")
print(f"  Theoretical minimum (just simulations): {theoretical_time_total:.6f} seconds")
if theoretical_time_total > 0:
    print(f"  Actual overhead factor (total): {total_time/theoretical_time_total:.2f}x (gradients + flow network)")
else:
    print("  Actual overhead factor (total): n/a (theoretical time ~ 0)")
if theoretical_time_steady > 0:
    print(f"  Actual overhead factor (steady): {steady_time/theoretical_time_steady:.2f}x")
else:
    print("  Actual overhead factor (steady): n/a")

print(f"\nFinal ELBO: {elbo_history[-1]:.4f}")

# Sample from posterior
print("\nSampling from posterior...")
key, k_sample = random.split(key)
samples = sample_flow(flow_params, k_sample, n_samples=post_n_samples)

print(f"Generated {len(samples)} posterior samples")

# Visualize in normalized space
pts = np.asarray(samples, dtype=np.float32)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pts[:,0],pts[:,1], s=10, alpha=0.6)
plt.xlabel('normalized log(Q)')
plt.ylabel('normalized log(b)')
plt.grid()
plt.title('Flow VI posterior - normalized space')

# Map back to parameter space
params_post = theta_to_params_samples(space, samples)
Q = jax.device_get(params_post["Q"]).ravel()
b = jax.device_get(params_post["b"]).ravel()

plt.subplot(1, 2, 2)
plt.scatter(Q, b, s=10, alpha=0.6)
plt.xlabel("Q")
plt.ylabel("b")
plt.axvline(true_params["Q"], color='r', linestyle='--', label='True Q')
plt.axhline(true_params["b"], color='r', linestyle='--', label='True b')
plt.grid(True)
plt.legend()
plt.title('Flow VI posterior - parameter space')
plt.tight_layout()
plt.savefig('plots/plasticity_flow_vi_posterior.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots/plasticity_flow_vi_posterior.png")


print("------------------------------------------------")
print("parameter summaries and predictive distributions")
print("------------------------------------------------")

# Posterior summaries in parameter space (dict: name -> (mean, p05, p95))
param_summ = posterior_param_summary(samples, space)
print(f"b: mean={param_summ['b'][0]:.4f}, p05={param_summ['b'][1]:.4f}, p95={param_summ['b'][2]:.4f}")
print(f"   (true: {true_params['b']:.4f})")
print(f"Q: mean={param_summ['Q'][0]:.4f}, p05={param_summ['Q'][1]:.4f}, p95={param_summ['Q'][2]:.4f}")
print(f"   (true: {true_params['Q']:.4f})")

if do_predictive:
    # Predictive band for σ11(t)
    print("\nComputing posterior predictive...")
    idx = np.random.choice(len(samples), size=min(len(samples), 100), replace=False)
    subset = samples[idx]
    sigma_mean, sigma_p05, sigma_p95, sigma_samples = posterior_predictive(subset, space, forward_sigma11)

    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(sigma_p05)), sigma_p05, sigma_p95, alpha=0.3, label='90% credible interval')
    plt.plot(sigma_mean, 'b-', linewidth=2, label='Posterior mean')
    plt.plot(sigma_xx_obs, 'k--', linewidth=2, label='Observations')
    plt.xlabel('Time step')
    plt.ylabel(r'$\sigma_{11}$')
    plt.title('Posterior predictive distribution')
    plt.legend()
    plt.grid()
    plt.savefig('plots/plasticity_flow_vi_predictive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plots/plasticity_flow_vi_predictive.png")

print("\nFlow VI identification completed!")

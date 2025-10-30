import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax, jit
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve as la_solve
from jax.flatten_util import ravel_pytree

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_implicit_unravel
from optimization.optimizers import bfgs




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
def residuals(x, epsilon, state_old, params, sigma_idx):

    C = C_iso_voigt(params["E"], params["nu"])
    sigma, eps_p, p, gamma = x["sigma"], x["eps_p"], x["p"], x["gamma"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    X = x["X"]
    X_old = state_old["X"]

    # Contraintes : on remplace ε[eps_idx] par les inconnues eps_cstr
    epsilon_eff = epsilon.at[sigma_idx].set(x["eps_cstr"])

    # 1) relation élastique
    res_sigma = sigma - C @ (epsilon_eff - eps_p)

    # 2) direction de flux (autodiff) — plus de branchement ici
    df_dsigma = jax.grad(lambda s: f_func(s-X, p, params))(sigma)

    # 3) écoulement associé + cumul de plasticité
    res_epsp = (eps_p - eps_p_old) - gamma * df_dsigma
    res_p    = (p - p_old) - gamma

    # 4) contraintes sur sigma (σ[idx] = 0)
    res_cstr = sigma[sigma_idx]

    # 5) consistance plastique (Newton ne s'exécute que si on est en plastique)
    res_gamma = f_func(sigma-X, p, params)

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
    A = C[sigma_idx][:, sigma_idx]                                                   # (k,k)
    r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    z  = epsilon[sigma_idx] - dz                                                     # eps_cstr_trial
    epsilon_eff_trial = epsilon.at[sigma_idx].set(z)
    sigma_trial       = C @ (epsilon_eff_trial - eps_p_old)

    # yield function at trial
    f_trial = f_func(sigma_trial-X_old, p_old, params)

    it_dtype = jnp.int32  # keep cond branches identical

    # ---------- 2) Elastic branch (skip Newton) ----------
    def elastic_branch(_):
        new_state = {"epsilon_p": eps_p_old, "p": p_old, "X": X_old}
        fields    = {"sigma": sigma_trial}
        logs      = {"conv": jnp.asarray(0, dtype=it_dtype),
                     "eps_cstr": z}
        return new_state, fields, logs

    # ---------- 3) Plastic branch (run Newton), init with the constrained trial ----------
    def plastic_branch(_):
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
        logs      = {"conv": jnp.asarray(iters, dtype=it_dtype),
                     "eps_cstr": x_sol["eps_cstr"]}
        return new_state, fields, logs

    # ---------- 4) Gate on f_trial ----------
    new_state, fields, logs = lax.cond(
        f_trial > 0.0,
        plastic_branch,
        elastic_branch,
        operand=None
    )
    return new_state, fields, logs


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

#load_list = [
#    {"t": ts[i],
#     "epsilon": epsilon_ts[i],
#     "sigma_cstr_idx": jnp.asarray([1, 2, 3, 4, 5])}
#    for i in range(len(ts))
#]
#def stack_load_list(load_list):
#    # Turn list[dict(arrays)] -> dict(arrays with leading time dim)
#    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *load_list)
#load = stack_load_list(load_list)

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
plt.show()

print(saved["logs"]["conv"])

print("--------------------------")
print("save reference sigma_xx(t)")
print("--------------------------")

true_params = params.copy()
sigma_xx_obs = saved["fields"]["sigma"][:, 0]


print("-------------")
print("run optimizer")
print("-------------")



from jax import vjp

# base params
base_params = true_params

# ---- 1) Dict <-> flat vector (identity mapping) ----
def build_param_io_identity(init_phys: dict):
    theta0_vec, unravel = ravel_pytree({k: jnp.asarray(v, dtype=jnp.float64) for k, v in init_phys.items()})

    def pack(phys_dict):
        vec, _ = ravel_pytree({k: jnp.asarray(phys_dict[k], dtype=jnp.float64) for k in init_phys.keys()})
        return vec

    def unpack(theta_vec):
        d = unravel(theta_vec)
        return {k: jnp.asarray(d[k], dtype=jnp.float64) for k in init_phys.keys()}

    return pack, unpack, theta0_vec

# ---- 2) Loss on the flat vector (uses simulate correctly) ----
def make_loss_fn_identity(base_params: dict, epsilon_ts: jnp.ndarray, sigma_xx_obs: jnp.ndarray, unpack_fn):
    @jit
    def forward_sigma_xx_from_vec(theta_vec):
        learn_phys = unpack_fn(theta_vec)
        par = {**base_params, **learn_phys}
        # IMPORTANT: state structure consistent with constitutive_update_fn
        state0 = {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0), "X": jnp.zeros(6)}
        local_load = load_ts
        _, saved = simulate(constitutive_update_fn, state0, local_load, par)
        return saved["fields"]["sigma"][:, 0]

    @jit
    def loss(theta_vec):
        pred = forward_sigma_xx_from_vec(theta_vec)
        resid = pred - sigma_xx_obs
        return 0.5 * jnp.mean(resid**2)

    return loss

# ---- 4) Run identification (fit Q and b) ----

init_phys = {
    "b": jnp.array(0.04),   # initial guess
    "Q": jnp.array(0.5),    # initial guess
}

pack, unpack, theta0 = build_param_io_identity(init_phys)
loss = make_loss_fn_identity(base_params, epsilon_ts, sigma_xx_obs, unpack)

print('running BFGS')
t0 = time.perf_counter()
theta_opt, f_bfgs, info = bfgs(loss, theta0, max_iter=20, gtol=1e-7,rtol=1.e-3)
t1 = time.perf_counter()
print("time for optimization:", (t1 - t0), "s")
print(f"[BFGS]   loss = {f_bfgs:.3e}")
print("info",info)

# Recover physical params as dict (identity mapping)
theta_phys = unpack(theta_opt)
print("Identified params:", {k: float(v) for k, v in theta_phys.items()})



def forward_sigma_xx_from_phys(phys_dict):
    par = {**base_params, **phys_dict}
    state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0),"X": jnp.zeros(6)}
    _, saved_fit = simulate(constitutive_update_fn, state0, load_ts, par)
    return saved_fit["fields"]["sigma"][:, 0]

sigma_fit = forward_sigma_xx_from_phys(theta_phys)

plt.figure()
plt.plot(load_ts['epsilon'][:,0], sigma_xx_obs, label="σ_xx (obs)")
plt.plot(load_ts['epsilon'][:,0], sigma_fit, "--", label="σ_xx (fit)")
plt.grid(True)
plt.legend()
plt.title("Identification (no smoothing)")
plt.show()

print("true_params",true_params)
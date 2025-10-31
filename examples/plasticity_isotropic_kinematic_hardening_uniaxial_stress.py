import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve as la_solve

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate_unpack
from simulation.algebra import dev_voigt, norm_voigt
from simulation.newton import newton_unravel



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

# ----------------- constitutive update (pure function) -----------------
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
    f_trial = f_func(sigma_trial-X_old, p_old, params)

    it_dtype = jnp.int32  # keep cond branches identical

    # ---------- 2) Elastic branch (skip Newton) ----------
    def elastic_branch(_):
        new_state = {"epsilon_p": eps_p_old, "p": p_old, "X": X_old}
        fields    = {"sigma": sigma_trial,"eps_cstr": z}
        logs      = {"conv": jnp.asarray(0, dtype=it_dtype)}
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
        x_sol, iters = newton_unravel(
            residuals, x0, (epsilon, state_old, params, sigma_idx),
            tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
        )
        new_state = {"epsilon_p": x_sol["eps_p"], "p": x_sol["p"], "X": x_sol["X"]}
        fields    = {"sigma": x_sol["sigma"],"eps_cstr": x_sol["eps_cstr"]}
        logs      = {"conv": jnp.asarray(iters, dtype=it_dtype)}
        return new_state, fields, logs

    # ---------- 4) Gate on f_trial ----------
    new_state, fields, logs = lax.cond(
        f_trial > 0.0,
        plastic_branch,
        elastic_branch,
        operand=None
    )
    return new_state, fields, logs

# ----------------- example usage -----------------

# material params
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
eps_xx = 4.0 * jnp.sin(ts * 30.0)
epsilon_ts = (jnp.zeros((len(ts), 6))
              .at[:, 0].set(eps_xx))

sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])

load_ts = {
    "epsilon": epsilon_ts,
    "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])) ,
}

state0 = {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0),"X": jnp.zeros(6,)}
#state_T, saved = simulate(constitutive_update_fn,internal_state0, load, params)
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

print("iter (first 100)",logs_ts["conv"][:100])
print("eps_cstr (first 10)",fields_ts["eps_cstr"][:10,:])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
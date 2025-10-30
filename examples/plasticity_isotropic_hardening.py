import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax
from jax import config
config.update("jax_enable_x64", True)

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate
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

#grad_f_wrt_sigma = jax.grad(lambda sigma, p, params: f_func(sigma, p, params), argnums=0)

# ----------------- constitutive residuals (dict in/out) -----------------
def residuals(x, epsilon, state_old, params):

    C = C_iso_voigt(params["E"], params["nu"])
    sigma, eps_p, p = x["sigma"], x["eps_p"], x["p"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    # 1) relation élastique
    res_sigma = sigma - C @ (epsilon - eps_p)

    # 2) direction de flux par autodiff (comme “celui-là converge”)
    df_dsigma = jax.grad(lambda s: f_func(s, p, params))(sigma)
    # pas de projection supplémentaire ici (reste fidèle à la version qui marche)

    # 3) écoulement associé + cumul de plasticité
    res_epsp  = (eps_p - eps_p_old) -(p - p_old) * df_dsigma

    # 4) consistance plastique vs. γ=0 élastique
    res_p = f_func(sigma, p, params)

    return {"res_sigma": res_sigma, "res_epsp": res_epsp, "res_p": res_p}

# ----------------- constitutive update (pure function) -----------------
def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):

    C = C_iso_voigt(params["E"], params["nu"])
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    epsilon   = step_load["epsilon"]

    sigma_trial = C @ (epsilon - eps_p_old)
    f_trial     = f_func(sigma_trial, p_old, params)
    dtype       = sigma_trial.dtype

    it_dtype = jnp.int32  # <- unifie le dtype des itérations

    def elastic_branch(_):
        new_state = {"epsilon_p": eps_p_old, "p": p_old}
        fields    = {"sigma": sigma_trial}
        logs      = {"conv": jnp.asarray(0, dtype=it_dtype)}
        return new_state, fields, logs

    def plastic_branch(_):
        x0 = {
            "sigma": sigma_trial,
            "eps_p": eps_p_old,
            "p":     jnp.asarray(p_old, dtype=dtype)
        }
        x_sol, iters = newton_unravel(
            residuals, x0, (epsilon, state_old, params),
            tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
        )
        new_state = {"epsilon_p": x_sol["eps_p"], "p": x_sol["p"]}
        fields    = {"sigma": x_sol["sigma"]}
        logs      = {"conv": jnp.asarray(iters, dtype=it_dtype)}
        return new_state, fields, logs

    new_state, fields, logs = lax.cond(
        f_trial > 0.0,
        plastic_branch,
        elastic_branch,
        operand=None
    )
    return new_state, fields, logs

# material params
E, nu = 1.0, 0.3
params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "E" : 1.0,
    "nu": 0.3,
}

# strain history
n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)
eps_xx = 4.0 * jnp.sin(ts * 30.0)
epsilon_ts = (jnp.zeros((n_ts, 6))
              .at[:, 0].set(eps_xx)
              .at[:, 1].set(-0.5 * eps_xx)
              .at[:, 2].set(-0.5 * eps_xx))

print("epsilon_ts.dtype",epsilon_ts.dtype)

#load_list = [
#    {"t": ts[i],
#     "epsilon": epsilon_ts[i],
#     }
#    for i in range(len(ts))
#]
#def stack_load_list(load_list):
#    Turn list[dict(arrays)] -> dict(arrays with leading time dim)
#    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *load_list)
#load = stack_load_list(load_list)
load_ts={"epsilon": epsilon_ts}

# initial state
state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0)}

# run
t0 = time.time()
state_T, saved = simulate(constitutive_update_fn,state0, load_ts, params)
jax.block_until_ready(saved)
print("simulate (first call):  %.2f ms" % ((time.time()-t0)*1e3))

# run
t0 = time.time()
state_T, saved = simulate(constitutive_update_fn,state0, load_ts, params)
jax.block_until_ready(saved)
print("simulate (already compiled):  %.2f ms" % ((time.time()-t0)*1e3))

# quick checks
print("Final p:", state_T["p"])
print("Hist sigma shape:", saved["fields"]["sigma"].shape)
print("Hist eps_p shape:", saved["state"]["epsilon_p"].shape)
print("Hist p shape:", saved["state"]["p"].shape)

#plot_stress_strain(saved, load, comp=0)

print("iteration count (first 100)",saved["logs"]["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,saved["fields"]["sigma"][:,0])
plt.grid()
plt.xlabel(r"$F_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
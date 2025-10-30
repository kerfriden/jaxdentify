import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import config
config.update("jax_enable_x64", True)

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

def R_iso(p, Q, b):
    return Q * (1.0 - jnp.exp(-b * p))

def vm(sigma):
    s = dev_voigt(sigma)
    return jnp.sqrt(3.0/2.0) * norm_voigt(s)

def f_func(sigma, p, sigma_y, Q, b):
    return vm(sigma) - (sigma_y + R_iso(p,Q,b))

def residuals(x, step_load, state_old, params):

    epsilon   = step_load["epsilon"]
    E, nu  = params["E"], params["nu"]
    sigma_y, Q, b = params["sigma_y"], params["Q"], params["b"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]
    sigma, eps_p, p = x["sigma"], x["eps_p"], x["p"]
    sigma, eps_p, p = x["sigma"], x["eps_p"], x["p"]

    C = C_iso_voigt(E, nu)
    sig_trial = C @ (epsilon - eps_p_old)
    f_trial   = f_func(sig_trial, p_old, sigma_y, Q, b)
    H = jnp.heaviside(f_trial , 1.)

    res_sigma = sigma - C @ (epsilon - eps_p)

    df_dsigma = jax.grad(lambda s: f_func(s, p, sigma_y, Q, b))(sigma)
    res_epsp  = (eps_p - eps_p_old) - (p - p_old) * df_dsigma

    res_p = f_func(sigma, p, sigma_y, Q, b) * H + (1.0 - H) * (p - p_old)

    res = {"res_sigma": res_sigma, "res_epsp": res_epsp, "res_p": res_p}

    return res

def initialize(step_load,state_old,params):

    epsilon   = step_load["epsilon"]
    E, nu  = params["E"], params["nu"]
    sigma_y, Q, b = params["sigma_y"], params["Q"], params["b"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    C = C_iso_voigt(params["E"], params["nu"])

    sigma_trial = C @ (epsilon - eps_p_old)
    f_trial     = f_func(sigma_trial, p_old, sigma_y, Q, b)

    x0 = { "sigma": sigma_trial, "eps_p": eps_p_old, "p": jnp.asarray(p_old) }

    return x0

def unpack(x_sol,iters):

    # we need to redefine the structure as jax does not accept mutations
    new_state = {"epsilon_p": x_sol["eps_p"], "p": x_sol["p"]}
    fields    = {"sigma": x_sol["sigma"]}
    logs      = {"conv": jnp.asarray(iters)}

    return new_state, fields, logs

# ----------------- constitutive update (pure function) -----------------
def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):
    x0 = initialize(step_load,state_old,params)
    x_sol, iters = newton_unravel(
        residuals, x0, (step_load, state_old, params),
        tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
    )
    new_state, fields, logs = unpack(x_sol, iters)
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

load_ts={"epsilon": epsilon_ts}

state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0)}
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

print("iteration count (first 100)",logs_ts["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$F_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
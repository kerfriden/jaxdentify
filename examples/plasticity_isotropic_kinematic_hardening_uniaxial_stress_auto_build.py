import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import lax
from jax import config
config.update("jax_enable_x64", True)
from jax.scipy.linalg import solve as la_solve

import time
import matplotlib.pyplot as plt

from simulation.simulate import make_simulate_auto_unpack
from simulation.algebra import dev_voigt, norm_voigt, voigt_to_tensor, tensor_to_voigt

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

def Hooke_law_voigt(eps_e, E, nu):
    lam = E*nu / ((1 + nu) * (1 - 2*nu))
    mu  = E / (2 * (1 + nu))
    I = jnp.eye(3, dtype=eps_e.dtype)
    eps_e_tens = voigt_to_tensor(eps_e)
    sig_tens = lam*jnp.trace(eps_e_tens)*I+2*mu*eps_e_tens
    return tensor_to_voigt(sig_tens)

def R_iso(p, Q, b):
    return Q * (1.0 - jnp.exp(-b * p))

def vm(sigma):
    s = dev_voigt(sigma)
    return jnp.sqrt(3.0/2.0) * norm_voigt(s)

def f_func(sigma, X, p, sigma_y, Q, b):
    return vm(sigma-X) - (sigma_y + R_iso(p,Q,b))

# Solve for z = eps_cstr_trial so that (C @ (epsilon_eff - eps_p_old))[sigma_idx] = 0
# epsilon_eff[sigma_idx] = eps_cstr_trial
def solve_eps_cstr(epsilon,eps_p_old,sigma_idx,E,nu):
    C = C_iso_voigt(params["E"], params["nu"])
    A = C[sigma_idx][:, sigma_idx]                                                 # (k,k)
    r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    eps_cstr_trial = epsilon[sigma_idx] - dz                                               
    return epsilon.at[sigma_idx].set(eps_cstr_trial) , eps_cstr_trial

def make_newton(state_old, step_load, params):

    epsilon = step_load["epsilon"]
    sigma_idx = step_load.get("sigma_cstr_idx")

    E, nu  = params["E"], params["nu"]
    sigma_y, Q, b = params["sigma_y"], params["Q"], params["b"]

    eps_p_old, p_old , X_old = state_old["epsilon_p"], state_old["p"], state_old["X"]

    epsilon_eff_trial , eps_cstr_trial = solve_eps_cstr(epsilon,eps_p_old,sigma_idx,E,nu)
    sigma_trial = Hooke_law_voigt(epsilon_eff_trial - eps_p_old, E, nu)

    f_trial     = f_func(sigma_trial, X_old, p_old, sigma_y, Q, b)
    H = jnp.heaviside(f_trial , 1.)

    def residuals(x):

        sigma, eps_p, p, X = x["sigma"], x["epsilon_p"], x["p"], x["X"]

        epsilon_eff = epsilon.at[sigma_idx].set(x["eps_cstr"])

        res_sigma = sigma - Hooke_law_voigt(epsilon_eff - eps_p, E, nu)

        df_dsigma = jax.grad(lambda s: f_func(s, X, p, sigma_y, Q, b))(sigma)
        res_epsp  = (eps_p - eps_p_old) - (p - p_old) * df_dsigma

        res_p = f_func(sigma, X, p, sigma_y, Q, b) * H + (1.0 - H) * (p - p_old)

        res_X = (X - X_old) - ( 2./3. * params['C_kin'] * (eps_p-eps_p_old) - params['D_kin'] * X * (p-p_old) )

        res_cstr = sigma[sigma_idx]

        res = {"res_sigma": res_sigma, "res_epsp": res_epsp, "res_p": res_p, "res_X": res_X, "res_cstr": res_cstr}
        return res

    def initialize():

        return { "sigma": sigma_trial, "eps_cstr": eps_cstr_trial, **state_old }
    
    return residuals, initialize





# material params
E, nu = 1.0, 0.3
params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "C_kin": 0.25 ,
    "D_kin": 1.0 ,
    "E" : 1.0,
    "nu": 0.3,
}

# strain history
n_ts = 200
ts = jnp.linspace(0., 1., n_ts)
eps_xx = 4.0 * jnp.sin(ts * 30.0)
epsilon_ts = jnp.zeros((n_ts, 6)).at[:, 0].set(eps_xx)

sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])

load_ts={"epsilon": epsilon_ts, 
         "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])) 
         }

state0 = {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0), "X": jnp.zeros(6,)}
state_T, fields_ts, state_ts, logs_ts = make_simulate_auto_unpack(make_newton, state0, load_ts, params)

print("iteration count (first 100)",logs_ts["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import config
config.update("jax_enable_x64", True)

import time
import matplotlib.pyplot as plt

from simulation.simulate import make_simulate_unpack
from simulation.algebra import dev_voigt, norm_voigt

from simulation.algebra import voigt_to_tensor, tensor_to_voigt







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

def f_func(sigma, p, sigma_y, Q, b):
    return vm(sigma) - (sigma_y + R_iso(p,Q,b))

def make_newton(state_old, step_load, params):

    epsilon = step_load["epsilon"]
    E, nu, K, n  = params["E"], params["nu"], params["K"], params["n"]
    sigma_y, Q, b = params["sigma_y"], params["Q"], params["b"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    sigma_trial = Hooke_law_voigt(epsilon - eps_p_old, E, nu)
    f_trial     = f_func(sigma_trial, p_old, sigma_y, Q, b)

    dt = step_load["delta_t"]

    def residuals(x):

        sigma, eps_p, p = x["sigma"], x["eps_p"], x["p"]

        H = jnp.heaviside(f_trial , 1.)

        res_sigma = sigma - Hooke_law_voigt(epsilon - eps_p, E, nu)

        df_dsigma = jax.grad(lambda s: f_func(s, p, sigma_y, Q, b))(sigma)
        res_epsp  = (eps_p - eps_p_old) - (p - p_old) * df_dsigma

        #res_p = f_func(sigma, p, sigma_y, Q, b) * H + (1.0 - H) * (p - p_old)
        res_p = ( (p - p_old)/dt - (f_func(sigma, p, sigma_y, Q, b)/K)**n ) * H + (1.0 - H) * (p - p_old)

        res = {"res_sigma": res_sigma, "res_epsp": res_epsp, "res_p": res_p}

        return res

    def initialize():

        return { "sigma": sigma_trial, "eps_p": eps_p_old, "p": jnp.asarray(p_old) }

    def unpack(x): # we need to redefine the structure as jax does not accept mutations

        state = {"epsilon_p": x["eps_p"], "p": x["p"]}
        fields    = {"sigma": x["sigma"]}

        return state, fields
    
    return residuals, initialize, unpack

def initialize_state():
    return {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0)}



# material params
params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "E" : 1.0,
    "nu": 0.3,
    "K" : 0.001,
    "n": 1.0,
}

# strain history
n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)

omega = 30.0
alpha = 5.0          # example: double the frequency in the second half
t0 = 0.5

phase_second = alpha * omega * ts + (1.0 - alpha) * omega * t0

eps_xx = 4.0 * jnp.where(
    ts <= t0,
    jnp.sin(omega * ts),      # first half: original sinus
    jnp.sin(phase_second),    # second half: higher frequency, continuous
)

#eps_xx = 1.0* jnp.where(
#    ts <= t0,
#    omega * ts,      # first half: original sinus
#    phase_second,    # second half: higher frequency, continuous
#)

plt.plot(eps_xx)
plt.grid()
plt.show()

epsilon_ts = (jnp.zeros((n_ts, 6))
              .at[:, 0].set(eps_xx)
              .at[:, 1].set(-0.5 * eps_xx)
              .at[:, 2].set(-0.5 * eps_xx))

# Δt: first step = ts[1] - ts[0], then forward differences
dt0 = ts[1] - ts[0]
delta_t = jnp.concatenate([jnp.array([dt0]), jnp.diff(ts)])

print("epsilon_ts.dtype",epsilon_ts.dtype)

load_ts={"epsilon": epsilon_ts, "delta_t": delta_t}

state0 = initialize_state()
state_T, fields_ts, state_ts, logs_ts = make_simulate_unpack(make_newton,state0, load_ts, params)

print("iteration count (first 100)",logs_ts["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
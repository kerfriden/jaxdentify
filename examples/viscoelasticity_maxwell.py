import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import config
config.update("jax_enable_x64", True)

import time
import matplotlib.pyplot as plt

from simulation.simulate import simulate_unpack
from simulation.algebra import voigt_to_tensor, tensor_to_voigt



# ----------------- constitutive update (pure function) -----------------
def constitutive_update_fn(state_old, step_load, params):
    K, G, eta = params["K"], params["G"], params["eta"]

    eps      = step_load["epsilon"]         # (6,)
    deps     = step_load["delta_epsilon"]   # (6,)
    dt       = step_load["delta_t"]         # scalar
    sigma_n  = state_old["sigma"]           # (6,)

    # Volumetric (elastic)
    p = K * (eps[0] + eps[1] + eps[2])
    sigma_vol = jnp.array([p, p, p, 0.0, 0.0, 0.0], dtype=eps.dtype)

    # Deviatoric (Maxwell, Backward Euler)
    r = 1.0 / (1.0 + (G*dt)/eta)

    I = jnp.eye(3, dtype=eps.dtype)
    s_n = voigt_to_tensor(sigma_n)
    s_n_dev = s_n - jnp.trace(s_n)/3.0 * I

    dE = voigt_to_tensor(deps)
    dE_dev = dE - jnp.trace(dE)/3.0 * I

    s_np1_dev = r * (s_n_dev + 2.0*G * dE_dev)
    sigma_dev = tensor_to_voigt(s_np1_dev)

    sigma = sigma_vol + sigma_dev

    new_state = {"sigma": sigma}
    fields = {"sigma": sigma}
    logs = {}
    return new_state, fields, logs





# material params
E, nu = 1.0, 0.3
params = {
    "K": 1.0,
    "G": 1.0,
    "eta": 0.1
}

# strain history
n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)

# strain history (Voigt)
eps_xx = 4.0 * jnp.sin(ts * 30.0)
eps_xx = 4.0 * jnp.sin( ts * 30.0)
epsilon_ts = (jnp.zeros((len(ts), 6)).at[:, 0].set(eps_xx))

# Δt: first step = ts[1] - ts[0], then forward differences
dt0 = ts[1] - ts[0]
delta_t = jnp.concatenate([jnp.array([dt0]), jnp.diff(ts)])

# Δε: first step = 0, then forward differences along time
zero6 = jnp.zeros((1, 6), dtype=epsilon_ts.dtype)
delta_epsilon = jnp.concatenate([zero6, epsilon_ts[1:] - epsilon_ts[:-1]], axis=0)

load_ts = {
    "t": ts,                       # (T,)
    "epsilon": epsilon_ts,         # (T, 6)
    "delta_t": delta_t,            # (T,)
    "delta_epsilon": delta_epsilon # (T, 6)
}

state0 = {"sigma": jnp.zeros(6),}
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$F_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()
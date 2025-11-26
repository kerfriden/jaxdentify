import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from simulation.simulate import simulate, simulate_unpack
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

# ----------------- constitutive update (pure function) -----------------
def constitutive_update_fn(state_old, step_load, params):

    C = C_iso_voigt(params["E"], params["nu"])
    eps      = step_load["epsilon"]

    sigma = C @ eps

    new_state = {}
    fields = {"sigma": sigma}
    logs = {}
    return new_state, fields, logs

# ----------------- example usage -----------------
print("----------------------------")
print("Simulating linear elasticity")
print("----------------------------")

# material params
params = {
    "E": 1.0,
    "nu": 0.3,
}

# strain history
n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)

# strain history (Voigt)
eps_xx = 4.0 * jnp.sin(ts * 30.0)
eps_xx = 4.0 * jnp.sin( ts * 30.0)
epsilon_ts = (jnp.zeros((len(ts), 6)).at[:, 0].set(eps_xx))

load_ts = { "epsilon": epsilon_ts }

state0 = {}
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

eps11 = jnp.array(load_ts["epsilon"][:,0])
plt.plot(eps11,fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()



print("----------------------------------------------------------")
print("Reverse mode autodiff with respect to parameter dictionary")
print("----------------------------------------------------------")

params = {
    "E": 1.0,
    "nu": 0.3,
    "C": C_iso_voigt(params["E"], params["nu"])
}

# wrapper: params_dict -> sigma11(t)
def sigma11_time_series(params_dict):
    state0 = {}
    _, saved_local = simulate(constitutive_update_fn,state0,load_ts,params_dict)
    sigma11_T = saved_local["fields"]["sigma"][:, 0]  # (T,)
    return sigma11_T

# we want d sigma11(t) / d params_dict
jac_fn = jax.jacrev(sigma11_time_series) # create derivative function taking parameter dict as input
sens = jac_fn(params) # calls it on instantiated dictionary

# sens is a pytree with same structure as params:
# sens["E"]   has shape (T,)  -> ∂σ11(t)/∂E
# sens["nu"]  has shape (T,)  -> ∂σ11(t)/∂nu

d_sigma11_dE  = sens["E"]
d_sigma11_dnu = sens["nu"]
d_sigma11_dC = sens["C"]

print("d sigma11 / dE:",  d_sigma11_dE[0:5])
print("d sigma11 / dnu:", d_sigma11_dnu[0:5])
print("d sigma11 / dC:", d_sigma11_dC[0:2]) # should be zero as C is not used in constitutive update, just to check that we can differentiate wrt arrays and matrices


print("------------------------------")
print("some parameters are now frozen")
print("------------------------------")

def sigma11_time_series_active(active_params, frozen_params):
    params_all = {**active_params, **frozen_params}

    state0 = {}
    _, saved_local = simulate(constitutive_update_fn,state0,load_ts,params_all)

    return saved_local["fields"]["sigma"][:, 0]  # (T,)

# tell JAX: take jacobian wrt arg 0 only (active_params)
jac_fn = jax.jacrev(sigma11_time_series_active, argnums=0)

active_params = {
    "E":  jnp.array(1.0),
    "C": jnp.array([0.2, 5.0, -1.0])

}
frozen_params = {"nu": jnp.array(0.3)}

sens_active = jac_fn(active_params, frozen_params)

# sens_active is a pytree with the SAME structure as active_params
# i.e. only "E" and "nu"
d_sigma11_dE  = sens_active["E"]   # (T,)
d_sigma11_dC = sens_active["C"]  # (T,)

try:
    d_sigma11_dC = sens_active["nu"]
except Exception as e:
    print("[warning] failed deriviative wrt nu")

print("d sigma11 / dE:",  d_sigma11_dE[0:5])
print("d sigma11 / dC:", d_sigma11_dC[0:5])


import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import  lax
from jax.scipy.linalg import solve as la_solve

from functools import partial

# ----------------- simulate (functional; takes state, load, params) -----------------
@partial(jax.jit, static_argnames=("update_fn",))
def simulate(update_fn, internal_state, load, params):
    def step(state, step_load):
        state, field, logs = update_fn(state, step_load, params)
        return state, (state, field, logs)

    state_T, (states_hist, fields_hist, logs_hist) = lax.scan(step, internal_state, load)
    saved = {"state": states_hist, "fields": fields_hist, "logs": logs_hist}
    return state_T, saved

@partial(jax.jit, static_argnames=("update_fn",))
def simulate_unpack(update_fn, internal_state, load, params):
    state_T, saved = simulate(update_fn, internal_state, load, params)
    fields_ts, state_ts, logs_ts = saved["fields"], saved["state"], saved["logs"]
    return state_T, fields_ts, state_ts, logs_ts
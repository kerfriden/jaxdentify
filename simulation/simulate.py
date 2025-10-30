import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import  lax
from jax.scipy.linalg import solve as la_solve

from functools import partial

from simulation.newton import newton_unravel

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

@partial(jax.jit, static_argnames=("make_newton",))
def simulate_make_unpack(make_newton, internal_state, load, params):
    
    def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):
        residuals, initialize, unpack = make_newton(state_old, step_load, params)
        x0 = initialize()
        x_sol, iters = newton_unravel(
            residuals, x0,(),
            tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
        )
        new_state, fields = unpack(x_sol)
        logs = {"conv": jnp.asarray(iters)}
        return new_state, fields, logs

    state_T, saved = simulate(constitutive_update_fn, internal_state, load, params)
    fields_ts, state_ts, logs_ts = saved["fields"], saved["state"], saved["logs"]
    return state_T, fields_ts, state_ts, logs_ts
import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax import  lax
from jax.scipy.linalg import solve as la_solve

from functools import partial

from simulation.newton import newton_implicit_unravel

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

# depreciated as does not handle implicit differentiation due parameters being closed over in newton
@partial(jax.jit, static_argnames=("make_newton",))
def make_simulate_unpack(make_newton, internal_state, load, params):
    
    def constitutive_update_fn(state_old, step_load, params, alg = {"tol" :1e-8, "abs_tol":1e-12, "max_it":100}):
        residuals, initialize, unpack = make_newton(state_old, step_load, params)
        x0 = initialize()
        x_sol, iters = newton_implicit_unravel(
            residuals, x0,(),
            tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
        )
        new_state, fields = unpack(x_sol)
        logs = {"conv": jnp.asarray(iters)}
        return new_state, fields, logs

    state_T, saved = simulate(constitutive_update_fn, internal_state, load, params)
    fields_ts, state_ts, logs_ts = saved["fields"], saved["state"], saved["logs"]
    return state_T, fields_ts, state_ts, logs_ts 

@partial(jax.jit, static_argnames=("make_newton",))
@partial(jax.jit, static_argnames=("make_newton",))
def make_simulate_unpack(make_newton, internal_state, load, params):

    def constitutive_update_fn(state_old, step_load, params_in,
                               alg={"tol":1e-8, "abs_tol":1e-12, "max_it":100}):

        # Build init/unpack once (ok to close over, not differentiated)
        residuals0, initialize, unpack = make_newton(state_old, step_load, params_in)
        x0 = initialize()

        # Residual that does NOT close over traced values:
        def residuals_parametric(x, state_arg, load_arg, params_arg):
            residuals_dyn, _, _ = make_newton(state_arg, load_arg, params_arg)
            return residuals_dyn(x)

        # IMPORTANT: pass state_old & step_load as part of dyn_args
        x_sol, iters = newton_implicit_unravel(
            residuals_parametric,
            x0,
            (state_old, step_load, params_in),  # <- all explicit
            tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
        )

        new_state, fields = unpack(x_sol)
        logs = {"conv": jnp.asarray(iters)}
        return new_state, fields, logs

    def step(state, step_load):
        state, field, logs = constitutive_update_fn(state, step_load, params)
        return state, (state, field, logs)

    state_T, (states_hist, fields_hist, logs_hist) = lax.scan(step, internal_state, load)
    return state_T, fields_hist, states_hist, logs_hist
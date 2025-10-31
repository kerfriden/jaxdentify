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
from simulation.algebra import dev_voigt, norm_voigt, tensor_to_voigt
from simulation.newton import newton_unravel



# ==================== Neo-Hookean (finite strain) via energy + autodiff; F as input ====================

def lame_from_E_nu(E, nu):
    mu  = E / (2.0*(1.0 + nu))
    lam = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))
    return lam, mu

def psi_neo_hooke(F, params):
    """
    Compressible Neo-Hookean potential (Simo style):
      ψ(F) = (μ/2) (I1 - 3 - 2 ln J) + (λ/2) (ln J)^2
    where I1 = tr(C), C = F^T F, J = det F.
    """
    lam, mu = lame_from_E_nu(params["E"], params["nu"])
    #F = _ensure_F(F)
    J = jnp.linalg.det(F)
    C = F.T @ F
    I1 = jnp.trace(C)
    logJ = jnp.log(jnp.clip(J, a_min=1e-16))   # guard
    return 0.5*mu*(I1 - 3.0 - 2.0*logJ) + 0.5*lam*(logJ**2)

def P_from_F(F, params):
    """First Piola-Kirchhoff stress P = ∂ψ/∂F, via JAX autodiff."""
    grad_fun = jax.grad(lambda F_: psi_neo_hooke(F_, params))
    return grad_fun(F)

def cauchy_from_F(F, params):
    P = P_from_F(F, params)
    J = jnp.linalg.det(F)
    Js = jnp.clip(J, a_min=1e-16)
    sigma = (P @ F.T) / Js
    return tensor_to_voigt(sigma)

# ---------- Residuals: solve for selected entries of F so that σ[idx] = 0 ----------
def _set_F_entries(F, flat_indices, values):
    Ff = F.reshape(-1)
    Ff = Ff.at[flat_indices].set(values)
    return Ff.reshape(3,3)

def residuals(x, F_given, params, sigma_idx, F_cstr_idx):
    """
    x: {"F_cstr": (k,)}  unknown free entries of F at flat indices F_cstr_idx (row-major).
    F_given: prescribed F with some entries already set by the load.
    sigma_idx: Voigt indices where σ must be zero (e.g., [1,2,3,4,5] for uniaxial stress in x).
    F_cstr_idx: flat indices (0..8) of F entries to solve for (e.g., [4,8] for F_yy, F_zz).
    """
    F_eff = _set_F_entries(F_given, F_cstr_idx, x["F_cstr"])
    sigma = cauchy_from_F(F_eff, params)
    return {"res_cstr": sigma[sigma_idx]}

# ---------- Update function (pure; no history) ----------
def constitutive_update_fn(state_old, step_load, params, alg={"tol":1e-10, "abs_tol":1e-14, "max_it":60}):
    #F_in       = _ensure_F(step_load["F"])
    F_in       = step_load["F"]
    sigma_idx  = step_load["sigma_cstr_idx"]
    F_cstr_idx = step_load["F_cstr_idx"]

    # ---- warm-start from previous F_eff ----
    F_eff_prev = state_old.get("F_eff", jnp.eye(3, dtype=F_in.dtype))
    z0 = F_eff_prev.reshape(-1)[F_cstr_idx]      # initial guess for unknown entries

    # Newton only on F_cstr
    x0 = {"F_cstr": z0}
    x_sol, iters = newton_unravel(
        residuals, x0, (F_in, params, sigma_idx, F_cstr_idx),
        tol=alg["tol"], abs_tol=alg["abs_tol"], max_iter=alg["max_it"]
    )

    F_eff = _set_F_entries(F_in, F_cstr_idx, x_sol["F_cstr"])
    sigma = cauchy_from_F(F_eff, params)

    # ---- store F_eff for the next step ----
    new_state = {"F_eff": F_eff}
    fields    = {"sigma": sigma, "F_eff": F_eff}
    logs      = {"conv": jnp.asarray(iters, dtype=jnp.int32)}
    return new_state, fields, logs

# ----- material -----
params = {"E": 1.0, "nu": 0.3}  # stiffer to make plots cleaner

# --- constraints (uniaxial stress in x) ---
# σ_yy, σ_zz, σ_yz, σ_zx, σ_xy = 0  -> Voigt indices [1,2,3,4,5]
sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])

# Unknown F entries (solve 5 unknowns ↔ 5 stress constraints):
# row-major flat indices 0..8 -> [[Fxx,Fxy,Fxz],[Fyx,Fyy,Fyz],[Fzx,Fzy,Fzz]]
# choose: F_yy, F_zz, F_xy, F_xz, F_yz
F_cstr_idx = jnp.asarray([4, 8, 1, 2, 5])

# ----- load history (F as input) -----
n_ts = 10
ts = jnp.linspace(0., 1., n_ts)

# axial stretch ABOUT 1 (e.g., ramp 1.00 -> 1.15)
lam_x = 1.0 + 5. * ts        # or 1.0 + 0.15*jnp.sin(2*jnp.pi*ts)

# start from identity each step, then overwrite F_xx
F_hist = jnp.tile(0.2*jnp.eye(3, dtype=jnp.float64), (len(ts), 1, 1))
F_hist = F_hist.at[:, 0, 0].set(lam_x)

#load_list = [
#    {
#        "t": ts[i],
#        "F": F_hist[i],              # interpreted as F
#        "sigma_cstr_idx": sigma_cstr_idx,  # [1,2,3,4,5]
#        "F_cstr_idx": F_cstr_idx,          # [4,8,1,2,5]
#    }
#    for i in range(n_ts)
#]
#load = stack_load_list(load_list)
#def stack_load_list(load_list):
#    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *load_list)
#load = stack_load_list(load_list)

load_ts = {
    "t": ts,
    "F": F_hist,
    "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])) ,
    "F_cstr_idx": jnp.broadcast_to(F_cstr_idx, (len(ts), F_cstr_idx.shape[0])),
}

# ----- run -----
state0 = {"F_eff": jnp.eye(3, dtype=F_hist.dtype)}
#state_T, saved = simulate(constitutive_update_fn, state0, load, params)
state_T, fields_ts, state_ts, logs_ts = simulate_unpack(constitutive_update_fn,state0, load_ts, params)

print("iters (first 10):", logs_ts["conv"][:10])
print("first 3 solved [F_yy, F_zz, F_xy, F_xz, F_yz]:\n", fields_ts["F_eff"][:3])

#print("saved[fields][F_eff][:,0,0]",saved["fields"]["F_eff"][:,0,0])
#print("first 3 stress", saved["fields"]["sigma"][:3])
#print("saved_nhF[fields][sigma][:,0]",saved["fields"]["sigma"][:,0])

plt.plot(fields_ts["F_eff"][:,0,0],fields_ts["sigma"][:,0])
plt.grid()
plt.xlabel(r"$F_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()

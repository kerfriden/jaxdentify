import jax
import jax.numpy as jnp
#from jax import value_and_grad, jit, lax, tree, jacfwd
from jax.scipy.linalg import solve as la_solve

jax.config.update("jax_enable_x64", True)

import time
import matplotlib.pyplot as plt

from simulation.simulate import make_simulate_unpack
from simulation.simulate import simulate

from simulation.algebra import dev_voigt, norm_voigt, voigt_to_tensor, tensor_to_voigt

from optimization.optimizers import bfgs
from optimization.parameter_mappings import build_param_space, make_loss, to_params


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

def f_func(sigma, p, sigma_y, Q, b):
    return vm(sigma) - (sigma_y + R_iso(p,Q,b))

def Fischer_Burmeister(a,b):
    return jnp.sqrt(a**2+b**2)-a-b # ϕ(a,b)=0 ⟺ a≥0,b≥0,ab=0

def solve_eps_cstr(epsilon,eps_p_old,sigma_idx,params):
    C = C_iso_voigt(params["E"], params["nu"])
    A = C[sigma_idx][:, sigma_idx]                                                 # (k,k)
    r = (C @ (epsilon - eps_p_old))[sigma_idx]                                     # (k,)
    dz = la_solve(A, r, assume_a='gen')                                            # (k,)
    eps_cstr_trial = epsilon[sigma_idx] - dz                                               
    return epsilon.at[sigma_idx].set(eps_cstr_trial) , eps_cstr_trial

def make_newton(state_old, step_load, params):

    epsilon = step_load["epsilon"]
    E, nu, K_visco, n_visco  = params["E"], params["nu"], params["K"], params["n"]
    #n_visco   = jnp.asarray(params["n"], dtype=float)

    sigma_y, Q, b = params["sigma_y"], params["Q"], params["b"]
    eps_p_old, p_old = state_old["epsilon_p"], state_old["p"]

    sigma_idx = step_load.get("sigma_cstr_idx")

    epsilon_eff_trial , eps_cstr_trial = solve_eps_cstr(epsilon,eps_p_old,sigma_idx,params)

    sigma_trial = Hooke_law_voigt(epsilon_eff_trial - eps_p_old, E, nu)

    dt = step_load["delta_t"]

    def residuals_clipping(x):

        sigma, eps_p, p = x["sigma"], x["eps_p"], x["p"]

        eps_cstr = x["eps_cstr"]
        epsilon_eff = epsilon.at[sigma_idx].set(eps_cstr)

        res_sigma = sigma - Hooke_law_voigt(epsilon_eff - eps_p, E, nu)

        df_dsigma = jax.grad(lambda s: f_func(s, p, sigma_y, Q, b))(sigma)
        res_epsp  = (eps_p - eps_p_old) - (p - p_old) * df_dsigma

        #res_p = f_func(sigma, p, sigma_y, Q, b) * H + (1.0 - H) * (p - p_old)
        x = jax.nn.relu(f_func(sigma, p, sigma_y, Q, b) / K_visco)
        x = jnp.clip(x, 1e-12, None)   # <---- otherwise gradients NAN, To be investigated
        res_p = ((p - p_old)/dt - x**n_visco) #* H + (1.0 - H)*(p - p_old)

        res_cstr = sigma[sigma_idx]

        res = {"res_sigma": res_sigma, "res_epsp": res_epsp, "res_p": res_p, "res_cstr": res_cstr}

        return res

    def residuals_double_where(x):
        # unpack unknowns
        sigma = x["sigma"]
        eps_p = x["eps_p"]
        p     = x["p"]
        eps_cstr = x["eps_cstr"]

        # effective total strain with constraint dof
        epsilon_eff = epsilon.at[sigma_idx].set(eps_cstr)

        # elastic residual (Hooke in Voigt notation)
        res_sigma = sigma - Hooke_law_voigt(epsilon_eff - eps_p, E, nu)

        # plastic flow direction df/dsigma
        def f_sigma(s):
            return f_func(s, p, sigma_y, Q, b)

        df_dsigma = jax.grad(f_sigma)(sigma)

        # plastic strain residual
        res_epsp = (eps_p - eps_p_old) - (p - p_old) * df_dsigma

        # viscoplastic residual with ReLU and double-where style safety
        phi = f_func(sigma, p, sigma_y, Q, b)    # overstress-like yield function value

        # Perzyna-type positive-part: <phi/K_visco>_+ using ReLU
        y = phi / K_visco
        overstress = jax.nn.relu(y)              # >= 0

        # condition: only phi > 0 contributes viscoplastic flow
        cond = phi > 0.0

        # ---- double-where / safe-input trick ----
        # For entries where cond == False, we replace overstress by a
        # *safe* positive value (e.g. 1.0) so that the derivative of
        # overstress**n_visco is finite there. The output is then
        # masked back to 0 in the final where.
        overstress_safe = jnp.where(cond, overstress, 1.0)

        # inner where: dangerous power only sees safe inputs
        g = jnp.where(cond, overstress_safe**n_visco, 0.0)

        # outer where: final masking (often redundant but matches "double where" pattern)
        #g = jnp.where(cond, g, 0.0)
        # -----------------------------------------

        res_p = (p - p_old) / dt - g

        # constraint residual on selected stress component
        res_cstr = sigma[sigma_idx]

        return {
            "res_sigma": res_sigma,
            "res_epsp":  res_epsp,
            "res_p":     res_p,
            "res_cstr":  res_cstr,
        }

    def initialize():

        return { "sigma": sigma_trial, "eps_cstr": eps_cstr_trial, "eps_p": eps_p_old, "p": jnp.asarray(p_old) }

    def unpack(x): 
 
        state = {"epsilon_p": x["eps_p"], "p": x["p"]}
        fields    = {"sigma": x["sigma"]}

        return state, fields
    
    return residuals_double_where, initialize, unpack


def initialize_state():
    return {"epsilon_p": jnp.zeros(6,), "p": jnp.array(0.0)}


# material params
params = {
    "sigma_y": 1.0,
    "Q": 1.0,
    "b": jnp.array(0.1),
    "E" : 1,
    "nu": 0.3,
    "K" : 0.001,
    "n": 1.0,
}

# strain history
n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)

omega = 30.0
alpha = 5.0
t0 = 0.5

phase_second = alpha * omega * ts + (1.0 - alpha) * omega * t0

eps_xx = 4.0 * jnp.where(
    ts <= t0,
    jnp.sin(omega * ts),
    jnp.sin(phase_second),
)

epsilon_ts = (jnp.zeros((n_ts, 6))
                .at[:, 0].set(eps_xx)
                .at[:, 1].set(-0.5 * eps_xx)
                .at[:, 2].set(-0.5 * eps_xx))

dt0 = ts[1] - ts[0]
delta_t = jnp.concatenate([jnp.array([dt0]), jnp.diff(ts)])

sigma_cstr_idx = jnp.asarray([1, 2, 3, 4, 5])

load_ts = {"epsilon": epsilon_ts, "delta_t": delta_t, "sigma_cstr_idx": jnp.broadcast_to(sigma_cstr_idx, (len(ts), sigma_cstr_idx.shape[0])) }

state0 = initialize_state()
state_T, fields_ts, state_ts, logs_ts = make_simulate_unpack(
    make_newton, state0, load_ts, params
)

print("iteration count (first 100)", logs_ts["conv"][:100])

eps11 = jnp.array(load_ts["epsilon"][:, 0])
plt.plot(eps11, fields_ts["sigma"][:, 0])
plt.grid()
plt.xlabel(r"$\epsilon_{11}$")
plt.ylabel(r"$\sigma_{11}$")
plt.show()


print(logs_ts["conv"])

# save reference solution for inverse problem
true_params = params.copy()
sigma_xx_obs = fields_ts["sigma"][:, 0]


print("-----------------------------------------------")
print("active/frozen parameter ranges and distribution")
print("-----------------------------------------------")


# --- declare frozen + active (bounds + scale). Omit values in init_params to use interval midpoints ---
init_params = {
    "E": 1.0, "nu": 0.3, "sigma_y": 1.0,  # frozen values supplied
    "C_kin": 0.25, "D_kin": 1.0,          # frozen for this example
    "n": 1.0
}

active_specs = {
    "E": {"lower": 1.e-1, "upper": 1.e+2, "scale": "log"},
    "sigma_y": {"lower": 0.5, "upper": 3.e0, "scale": "linear"},
    "Q": {"lower": 1.e-3, "upper": 1.e+2, "scale": "log"},
    "b": {"lower": 1.e-3, "upper": 1.e+0, "scale": "log"},
    "K": {"lower": 1.e-4, "upper": 1.e-2, "scale": "log"},
    "n": {"lower": 0.9, "upper": 1.1, "scale": "linear"},
}

space, theta0 = build_param_space(init_params, active_specs)

print("theta0 (latent):", theta0)

init_phys = to_params(space, theta0)
print("initial physical params:")
for k, v in init_phys.items():
    print(f"  {k}: {v}")

print("-----------------")
print("user-defined loss")
print("-----------------")


def forward_sigma11(params):
    state0 = initialize_state()
    state_T, fields_ts, state_ts, logs_ts = make_simulate_unpack(
        make_newton, state0, load_ts, params
    )
    return fields_ts["sigma"][:, 0]

def simulate_and_loss(params):
    pred = forward_sigma11(params)
    r = pred - sigma_xx_obs
    return 0.5 * jnp.mean(r * r)

loss = make_loss(space,simulate_and_loss)

print("-------------")
print("run optimizer")
print("-------------")

init = to_params(space, theta0)

# --- run BFGS (optionally seed with a few Adam steps you have) ---
t0 = time.perf_counter()
theta_opt, fval, info = bfgs(loss, theta0, rtol=1.e-3, n_display=1)
t1 = time.perf_counter()
print("time for optimizaton:", (t1 - t0), "s")
print("final loss:", fval)
print("info",info)

# --- unpack physical identified parameters ---
identified = to_params(space, theta_opt)
print("Identified Q, b:", identified["Q"], identified["b"])
# Optional: get fitted curve

print("-----------")
print("plot result")
print("-----------")


sigma_fit = forward_sigma11(identified)
sigma_init = forward_sigma11(init)

plt.plot(load_ts['epsilon'][:,0],sigma_fit,'blue',label=r'$\hat{\sigma}_{11}$ (fit)')
plt.plot(load_ts['epsilon'][:,0],sigma_xx_obs,'black',label=r'$\hat{\sigma}_{11}$ (data)')
plt.plot(load_ts['epsilon'][:,0],sigma_init,'green',label=r'$\hat{\sigma}_{11}$ (initial)')
plt.legend(loc='best')
plt.grid()
plt.show()

print("--------------")
print("test CPU times")
print("--------------")

v_ = loss(theta0).block_until_ready()
_ = jax.grad(loss)(theta0).block_until_ready()  # warm both paths

t0 = time.perf_counter()
f = loss(theta0).block_until_ready()
t1 = time.perf_counter()
print("one forward loss eval:", (t1 - t0) * 1e3, "ms")

t0 = time.perf_counter()
g = jax.grad(loss)(theta0).block_until_ready()
t1 = time.perf_counter()
print("one grad eval:", (t1 - t0) * 1e3, "ms")
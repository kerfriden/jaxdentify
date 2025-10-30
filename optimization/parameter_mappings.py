import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri  # Φ and Φ^{-1} # Returns the area under the Gaussian probability density function

from functools import partial

# ================= Param-space helpers (pure JAX) =================

def _phi(x):       # standard normal CDF
    return ndtr(x)

def _phinv(p, eps=1e-12):   # inverse CDF (probit)
    p = jnp.clip(p, eps, 1.0 - eps)
    return ndtri(p)

def _broadcast_pair(lower, upper, shape, dtype):
    lo = jnp.asarray(lower, dtype=dtype)
    hi = jnp.asarray(upper, dtype=dtype)
    return jnp.broadcast_to(lo, shape), jnp.broadcast_to(hi, shape)

def _mid_from_bounds(lower, upper, scale):
    if scale == "linear":
        return 0.5 * (lower + upper)
    elif scale == "log":
        ll = jnp.log(jnp.clip(lower, 1e-30))
        uu = jnp.log(jnp.clip(upper, 1e-30))
        return jnp.exp(0.5 * (ll + uu))
    else:
        raise ValueError(f"Unknown scale {scale!r}")

def _infer_shape_from(spec, init_val):
    if init_val is not None:
        return jnp.asarray(init_val).shape
    if spec is not None and spec.get("mask", None) is not None:
        return jnp.asarray(spec["mask"]).shape
    lb = jnp.asarray(spec["lower"])
    ub = jnp.asarray(spec["upper"])
    # jnp.broadcast_shapes is the JAX analogue to np.broadcast(...).shape
    return jnp.broadcast_shapes(lb.shape, ub.shape)

def _forward_key(space, k, x_vec):
    lo = space["lowers"][k]; hi = space["uppers"][k]; sc = space["scales"][k]  # 0 lin / 1 log
    p = _phi(x_vec)
    y_lin = lo + (hi - lo) * p
    y_log = jnp.exp(jnp.log(jnp.clip(lo, 1e-30)) +
                    (jnp.log(jnp.clip(hi, 1e-30)) - jnp.log(jnp.clip(lo, 1e-30))) * p)
    return jnp.where(sc == 0, y_lin, y_log)

def _inverse_key(space, k, y_vec):
    lo = space["lowers"][k]; hi = space["uppers"][k]; sc = space["scales"][k]
    p_lin = (y_vec - lo) / (hi - lo + 1e-30)
    p_log = (jnp.log(jnp.clip(y_vec, 1e-30)) - jnp.log(jnp.clip(lo, 1e-30))) / \
            (jnp.log(jnp.clip(hi, 1e-30)) - jnp.log(jnp.clip(lo, 1e-30)) + 1e-30)
    p = jnp.where(sc == 0, p_lin, p_log)
    return _phinv(p)

def build_param_space(init_params: dict, active_specs: dict):
    # union of keys so we can default actives even if missing in init
    init_params = init_params or {}
    active_specs = active_specs or {}
    all_keys = tuple(sorted(set(init_params.keys()) | set(active_specs.keys())))

    init = {}
    for k in all_keys:
        spec = active_specs.get(k, None)
        val  = init_params.get(k, None)
        if val is not None:
            init[k] = jnp.asarray(val)
        else:
            if spec is None:
                raise ValueError(f"Missing init for '{k}' and no spec to infer defaults.")
            shape = _infer_shape_from(spec, None)
            dtype = jnp.float64
            lo, hi = _broadcast_pair(spec["lower"], spec["upper"], shape, dtype)
            init[k] = _mid_from_bounds(lo, hi, spec["scale"])

    keys = tuple(init.keys())
    idx_map, shapes, lowers, uppers, scales, active_keys = {}, {}, {}, {}, {}, []
    for k in keys:
        v = jnp.asarray(init[k])
        spec = active_specs.get(k, False)
        if spec is False:
            continue
        active_keys.append(k)
        if spec.get("mask", None) is None:
            idx = jnp.arange(v.size, dtype=jnp.int32)
            lo, hi = _broadcast_pair(spec["lower"], spec["upper"], v.shape, init[k].dtype)
            lo = lo.reshape(-1); hi = hi.reshape(-1)
        else:
            m = jnp.asarray(spec["mask"], dtype=bool)
            assert m.shape == v.shape, f"mask for {k} must match {v.shape}"
            # flat indices where mask is True
            idx = jnp.flatnonzero(m.reshape(-1)).astype(jnp.int32)
            lo_full, hi_full = _broadcast_pair(spec["lower"], spec["upper"], v.shape, init[k].dtype)
            lo, hi = lo_full.reshape(-1)[idx], hi_full.reshape(-1)[idx]
        idx_map[k] = idx
        shapes[k]  = v.shape
        lowers[k]  = jnp.asarray(lo, dtype=init[k].dtype)
        uppers[k]  = jnp.asarray(hi, dtype=init[k].dtype)
        scales[k]  = jnp.asarray(0 if spec["scale"] == "linear" else 1, dtype=jnp.int32)

    start, slices = 0, {}
    for k in active_keys:
        n = int(idx_map[k].size)
        slices[k] = (start, start + n)
        start += n
    dim = start

    space = {
        "init": init,
        "keys": keys,
        "active_keys": tuple(active_keys),
        "idx_map": idx_map,
        "shapes": shapes,
        "slices": slices,
        "lowers": lowers,
        "uppers": uppers,
        "scales": {k: scales[k] for k in active_keys},
        "dim": jnp.asarray(dim, dtype=jnp.int32),
    }
    theta0 = to_theta(space, init)
    return space, theta0

def to_params(space, theta):
    full = {**space["init"]}
    for k in space["active_keys"]:
        a, b = space["slices"][k]
        yk = _forward_key(space, k, theta[a:b])
        flat = full[k].reshape(-1); idx = space["idx_map"][k]
        flat = flat.at[idx].set(yk)
        full[k] = flat.reshape(space["shapes"][k])
    return full

def to_theta(space, params_phys):
    chunks = []
    for k in space["active_keys"]:
        idx = space["idx_map"][k]
        y = jnp.asarray(params_phys[k]).reshape(-1)[idx]
        chunks.append(_inverse_key(space, k, y))
    return jnp.concatenate(chunks) if chunks else jnp.zeros((0,), dtype=jnp.float32)

def project_theta(space, theta, margin=1e-6):
    out = []
    for k in space["active_keys"]:
        a, b = space["slices"][k]
        p = _phi(theta[a:b])
        p = jnp.clip(p, margin, 1.0 - margin)
        out.append(_phinv(p))
    return jnp.concatenate(out) if out else theta

def sample_theta(space, key, n=1):
    thetas, rng = [], key
    for k in space["active_keys"]:
        rng, k1 = jax.random.split(rng)
        lo = space["lowers"][k]; hi = space["uppers"][k]; sc = int(space["scales"][k]); m = lo.size
        if sc == 0:
            y = jax.random.uniform(k1, (n, m)) * (hi - lo) + lo
        else:
            ll, uu = jnp.log(jnp.clip(lo, 1e-30)), jnp.log(jnp.clip(hi, 1e-30))
            y = jnp.exp(jax.random.uniform(k1, (n, m)) * (uu - ll) + ll)
        thetas.append(_inverse_key(space, k, y))
    return jnp.concatenate(thetas, axis=-1) if thetas else jnp.zeros((n, 0), dtype=jnp.float32)

# --- generic loss function with parameter unpacking (vectorised to dictionary) --
def make_loss(space, simulate_and_loss):
    @jax.jit
    def loss(theta):
        theta_  = project_theta(space, theta)
        params  = to_params(space, theta_)
        return simulate_and_loss(params)
    return loss
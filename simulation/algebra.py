import jax
import jax.numpy as jnp

# ----------------- elastic stiffness (engineering Voigt) -----------------
# ---- keep your C_iso_voigt exactly like this (no extra text on lines) ----
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

# ----------------- small helpers (engineering-Voigt) -----------------
W = jnp.array([1., 1., 1., 2., 2., 2.])

def dev_voigt(sig):
    tr = sig[0] + sig[1] + sig[2]
    return sig.at[0:3].add(-tr / 3.0)

def norm_voigt(sig, eps=None):
    if eps is None:
        eps = jnp.asarray(1e-16, dtype=sig.dtype)
    return jnp.sqrt(jnp.dot(W * sig, sig) + eps)

def voigt_to_tensor(v):
    # Engineering Voigt: [xx, yy, zz, yz, zx, xy]
    return jnp.array([
        [v[0], v[5]/2.0, v[4]/2.0],
        [v[5]/2.0, v[1],  v[3]/2.0],
        [v[4]/2.0, v[3]/2.0, v[2]],
    ], dtype=v.dtype)

def tensor_to_voigt(T):
    return jnp.array([
        T[0,0], T[1,1], T[2,2],
        2.0*T[1,2], 2.0*T[0,2], 2.0*T[0,1]
    ], dtype=T.dtype)
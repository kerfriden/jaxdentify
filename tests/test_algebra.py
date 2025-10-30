# run with python3 -m examples.test_AD_API
# or pytest examples/test_jacobian.py

import jax
import jax.numpy as jnp

from simulation.algebra import norm_voigt

def test_vm():

    sig = jnp.array([1., 0., 0., 0., 0., 0.])

    assert ( norm_voigt(sig) == 1. ).all()

if __name__ == "__main__":

    print("running test algebra")
    test_vm()
    print("success")
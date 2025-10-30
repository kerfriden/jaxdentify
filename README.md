# JAX for Constitutive Laws — User Guide

---

[![Open demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kerfriden/jaxdentify/blob/main/demo.ipynb)


---

## Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
  - [Voigt notation](#voigt-notation)
  - [Isotropic stiffness](#isotropic-stiffness)
  - [Newton solvers](#newton-solvers)
  - [Time stepping driver](#time-stepping-driver)
- [Built-in Examples](#built-in-examples)
  - [Example 1 — J2 plasticity (small strain, isotropic hardening)](#example-1--j2-plasticity-small-strain-isotropic-hardening)
  - [Example 1.2 — Linear elasticity + sensitivities](#example-12--linear-elasticity--sensitivities)
  - [Example 1.3 — J2 as single-branch Newton](#example-13--j2-as-single-branch-newton)
  - [Example 2 — J2 with stress constraints](#example-2--j2-with-stress-constraints)
  - [Example 3 — Maxwell viscoelasticity (small strain)](#example-3--maxwell-viscoelasticity-small-strain)
  - [Example 4 — Compressible Neo-Hookean (finite strain) + constraints](#example-4--compressible-neo-hookean-finite-strain--constraints)
  - [Example 5 — J2 with isotropic + kinematic hardening + constraints](#example-5--j2-with-isotropic--kinematic-hardening--constraints)
  - [Parameter Identification with Implicit Gradients](#parameter-identification-with-implicit-gradients)
- [API Reference (most-used functions)](#api-reference-most-used-functions)
- [Extending the Notebook](#extending-the-notebook)
- [Tips & Troubleshooting](#tips--troubleshooting)

---

## Overview

The notebook provides:

- **Reusable Newton solvers** (array and PyTree), including an **implicit-gradient** variant (custom VJP / IFT) to differentiate through solves.
- A **generic time-stepping driver** (`simulate`) using `jax.lax.scan`.
- Implementations of:
  - Small-strain **J2 plasticity** (isotropic + kinematic hardening) with optional **stress constraints**.
  - **Linear elasticity** with **automatic sensitivities** to parameters.
  - **Maxwell viscoelasticity** (small strain, deviatoric BE update).
  - **Compressible Neo-Hookean** (finite strain, stresses by AD) with Newton on a subset of **F** entries to enforce stress constraints.
- A **parameter identification** pipeline that fits model parameters with gradients passing through the Newton solver.

All code is **JIT-friendly**, **float64** by default, and uses **engineering Voigt** conventions.

---

## Quickstart

1. **Set material parameters and build a load history**:

```python
params = {"E": 1.0, "nu": 0.3, "sigma_y": 1.0, "Q": 1.0, "b": jnp.array(0.1)}

n_ts = 1000
ts = jnp.linspace(0., 1., n_ts)
eps_xx = 4.0 * jnp.sin(ts * 30.0)

epsilon_ts = (jnp.zeros((n_ts, 6))
              .at[:, 0].set(eps_xx)
              .at[:, 1].set(-0.5 * eps_xx)
              .at[:, 2].set(-0.5 * eps_xx))

load = {"epsilon": epsilon_ts}
```

2. **Choose an update function** and initial state:

```python
internal_state0 = {"epsilon_p": jnp.zeros(6), "p": jnp.array(0.0)}
state_T, saved = simulate(constitutive_update_fn, internal_state0, load, params)
```

3. **Plot a result**:

```python
eps11 = jnp.array(load["epsilon"][:,0])
plt.plot(eps11, saved["fields"]["sigma"][:,0])
plt.grid(); plt.xlabel("ε11"); plt.ylabel("σ11")
```

---

## Core Concepts

### Voigt notation

- Engineering Voigt vector ordering:

$$ v = [\,xx,\ yy,\ zz,\ yz,\ zx,\ xy\,] $$

- Shear weights for norms:

$$ W = \mathrm{diag}(1,1,1,2,2,2) $$

- Deviatoric projection (Voigt):

$$ s = v - \tfrac{1}{3}(v_{xx}+v_{yy}+v_{zz})[1,1,1,0,0,0] $$


### Isotropic stiffness

Given \(E,\nu\):

$$ \mu=\frac{E}{2(1+\nu)},\quad \lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\quad \lambda_2=\lambda+2\mu. $$

In Voigt form:

$$
\mathbf{C}=
\begin{bmatrix}
\lambda_2 & \lambda & \lambda & 0 & 0 & 0\\
\lambda & \lambda_2 & \lambda & 0 & 0 & 0\\
\lambda & \lambda & \lambda_2 & 0 & 0 & 0\\
0 & 0 & 0 & \mu & 0 & 0\\
0 & 0 & 0 & 0 & \mu & 0\\
0 & 0 & 0 & 0 & 0 & \mu
\end{bmatrix}.
$$


### Newton solvers

Residual solve: find \(x\) such that \(r(x)=0\).

- **`newton`**: array unknown, \(J=\partial r/\partial x\) via `jacfwd`, linear solve, abs/rel stopping; returns \((x^\*, \text{iters})\) or `iters = -1`.
- **`newton_unravel`**: PyTree wrapper around `newton` (dicts of arrays).
- **`newton_fixed_scan`**: fixed-iteration Newton with `lax.scan` (XLA-friendly).
- **`newton_implicit`**: same solve with a **custom VJP** (IFT). Backward pass solves

$$ J_x^\top \lambda = \bar{x},\quad \frac{d\mathcal{L}}{d\theta} = - (\partial_\theta r)^\top \lambda. $$

Use `newton_implicit_unravel` for the PyTree version.


### Time stepping driver

**`simulate(update_fn, state, load, params)`** uses `lax.scan`. At step \(t\):

$$ (s_{t+1},\ \text{field}_t,\ \text{logs}_t) = \text{update\_fn}(s_t,\ \text{load}_t,\ \text{params}) $$

It returns the final state and stacked histories under:

```
saved = {"state": states_hist, "fields": fields_hist, "logs": logs_hist}
```

---

## Built-in Examples

### Example 1 — J2 plasticity (small strain, isotropic hardening)

Yield function:

$$ f(\sigma,p) = \sqrt{\tfrac{3}{2}}\ \|\,\mathrm{dev}\,\sigma\,\|_W - (\sigma_y + Q(1-e^{-bp})) $$

Unknowns per step (one variant):

$$ x = [\sigma,\ \epsilon^p,\ p,\ \gamma]. $$

Residuals:

- Elastic relation:

$$ r_\sigma = \sigma - \mathbf{C}(\epsilon - \epsilon^p). $$

- Flow rule (associated):

$$ r_{\epsilon^p} = (\epsilon^p-\epsilon^p_\text{old}) - \gamma \frac{\partial f}{\partial \sigma}. $$

- Plastic strain increment:

$$ r_p = (p-p_\text{old}) - \gamma. $$

- Consistency: trial-based branch (elastic if \(f_\text{trial} \le 0\); else plastic Newton).

Trial state:

$$ \sigma_\text{trial}=\mathbf{C}(\epsilon-\epsilon^p_\text{old}),\quad f_\text{trial}=f(\sigma_\text{trial},p_\text{old}). $$


### Example 1.2 — Linear elasticity + sensitivities

- Update:

$$ \sigma=\mathbf{C}\epsilon. $$

- Differentiate time series outputs w.r.t. \(E, \nu\) (and other active params) with `jax.jacrev`.


### Example 1.3 — J2 as single-branch Newton

No branch: use

$$ H=\mathrm{heaviside}(f_\text{trial},1) $$

to turn plastic equations on/off at the trial level.


### Example 2 — J2 with stress constraints

**Goal:** enforce selected stresses to zero (e.g., uniaxial stress).

Constrained trial: solve for unknown strain components \(\epsilon_\text{cstr}\) so that \((\mathbf{C} (\epsilon_\text{eff} - \epsilon^p_\text{old}))[\text{idx}] = 0\).

Formal step:

$$ \mathbf{A}\,(z - \epsilon[\text{idx}]) = -\mathbf{r},\quad \mathbf{A} = \mathbf{C}[\text{idx},\text{idx}],\quad \mathbf{r} = (\mathbf{C}(\epsilon-\epsilon^p_\text{old}))[\text{idx}]. $$

Then \( z = \epsilon_\text{cstr}^\text{trial} = \epsilon[\text{idx}] - \mathbf{A}^{-1} \mathbf{r} \).

Unknowns include \(\epsilon_\text{cstr}\); residuals include \(\sigma[\text{idx}]\).


### Example 3 — Maxwell viscoelasticity (small strain)

Deviatoric BE update:

$$ r=\frac{1}{1 + \frac{G\Delta t}{\eta}},\quad \boldsymbol{\sigma}'_{n+1} = r\left(\boldsymbol{\sigma}'_n + 2G\,\Delta\boldsymbol{\varepsilon}'\right). $$

Volumetric part:

$$ p = K\,\mathrm{tr}(\varepsilon),\quad \boldsymbol{\sigma}^{vol} = p\,\mathbf{I}. $$

Total:

$$ \boldsymbol{\sigma}_{n+1} = \boldsymbol{\sigma}^{vol}_{n+1} + \boldsymbol{\sigma}'_{n+1}. $$

No Newton needed.


### Example 4 — Compressible Neo-Hookean (finite strain) + constraints

Energy (Simo style):

$$ \psi(F)=\frac{\mu}{2}(I_1-3-2\ln J)+\frac{\lambda}{2}(\ln J)^2,\quad I_1=\mathrm{tr}(C),\ C=F^\top F,\ J=\det F. $$

Stresses by AD:

$$ P=\frac{\partial \psi}{\partial F},\qquad \sigma=\frac{P F^\top}{J}. $$

Uniaxial stress: pick a subset of \(F\) entries as unknowns and solve by Newton so that \(\sigma[\text{idx}]=0\). Warm-start with previous \(F_\text{eff}\).


### Example 5 — J2 with isotropic + kinematic hardening + constraints

Backstress (Armstrong–Frederick type):

$$ \Delta X = \frac{2}{3} C_{\text{kin}}\,\Delta \epsilon^p - D_{\text{kin}}\, X\, \Delta p, \qquad f(\sigma-X,p)=0\ \text{on plastic branch}. $$

Unknowns include \(X\) and \(\epsilon_\text{cstr}\); constraints identical to Example 2.


### Parameter Identification with Implicit Gradients

Loss on the simulated \(\sigma_{11}\) vs observations:

$$ \mathcal{L}(\theta)=\frac{1}{2N}\sum_{t=1}^N\left(\sigma_{11}(t;\theta)-\sigma_{11}^{\text{obs}}(t)\right)^2. $$

Use `newton_implicit_unravel` inside the forward model so gradients \(d\mathcal{L}/d\theta\) pass through the solve. Optimize with **Adam** then **BFGS** (both implemented in JAX).

---

## API Reference (most-used functions)

```
C_iso_voigt(E, nu) -> (6,6)            # isotropic stiffness (engineering Voigt)
dev_voigt(sig6) -> (6,)                # deviatoric part in Voigt
norm_voigt(sig6, eps=1e-16) -> float   # weighted Voigt norm

newton(res_fn, x0, dyn_args, tol, abs_tol, max_iter) -> (x*, iters)
newton_unravel(res_fn_pytree, x0_tree, dyn_args, ...) -> (x_tree*, iters)

newton_fixed_scan(...): fixed-iteration Newton (XLA-friendly)
newton_implicit(...): Newton with custom VJP (IFT) for backprop
newton_implicit_unravel(...): PyTree version

simulate(update_fn, internal_state, load, params) -> (state_T, saved)
# where update_fn(state_t, load_t, params) -> (state_{t+1}, fields_t, logs_t)
```

**Data conventions**

- Small strain (Voigt): vectors length 6; loads typically supply `{"epsilon": (T,6)}`.
- Finite strain: loads supply `{"F": (T,3,3)}`; provide constraint indices for stress (`sigma_cstr_idx`) and unknown F entries (`F_cstr_idx`); broadcast over time as needed.
- State is a dict whose structure **must remain identical** across time steps.

---

## Extending the Notebook

1. **New material**

   - Write residuals in a PyTree:

     ```python
     def residuals(x, step_load, state_old, params, *extras):
         return {"r1": ..., "r2": ...}
     ```

   - Initialize Newton unknowns and unpack to `(new_state, fields, logs)`.
   - Use `newton_unravel` (or `newton_implicit_unravel` for differentiable ID).

2. **Change/add state variables**

   - Add entries to `state_old`, mirror in unknowns as needed, keep the **state shape constant**.

3. **Add constraints**

   - Add constraint residuals (e.g., `sigma[idx]=0`) and corresponding unknowns (e.g., `epsilon_cstr` or selected `F` entries).

---

## Tips & Troubleshooting

- Prefer **float64** (already enabled) for robustness.
- Avoid evaluating undefined directions at \(s=0\): gate with `lax.cond` (trial-based branching).
- Heaviside single-branch J2 introduces a kink; if Newton struggles, add damping/line search or revert to branching.
- Constraints: ensure selected indices form an invertible block of \(\mathbf{C}\) (true for standard uniaxial setups).
- Finite strain: clipping in \(\ln J\) avoids \(-\infty\); keep \(J>0\).
- Performance: `simulate` treats the **function object** as static (retrace only when you swap the update function).

---

*End of document.*

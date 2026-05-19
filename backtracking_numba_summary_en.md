# Summary of `backtracking_numba.py`

This file implements backward tracing of a relativistic proton in the IGRF
magnetic field. The code is intentionally simple: the main public entry point is
`trace_proton_backtracking(...)`, while the helper functions only compute vector
norms, the IGRF field, the ODE right-hand side, and one RK45 step.

## Traced State

The particle state is stored as 6 numbers:

```text
y = [x, y, z, ux, uy, uz]
```

where:

```text
x, y, z     Earth-centered Cartesian coordinates, km
ux, uy, uz  unit direction vector
```

The output array `trajectory` has 7 columns:

```text
t_s, x_km, y_km, z_km, ux, uy, uz
```

Important: `ux, uy, uz` are not velocity components in km/s. They are the unit
direction vector. The speed magnitude is computed separately from the kinetic
energy and remains constant.

## Coordinate System

The tracer coordinates are Earth-centered Cartesian coordinates in km:

```text
X: longitude 0 degrees, equator
Y: longitude 90 degrees east, equator
Z: north geographic pole
```

To evaluate IGRF, the current Cartesian point is converted to geocentric
spherical coordinates:

```text
r     = sqrt(x^2 + y^2 + z^2)
rho   = sqrt(x^2 + y^2)
colat = acos(z / r)
elong = atan2(y, x)
```

`igrf14syn(...)` is called with `itype = 2`, so `alt` means distance from the
Earth's center in km, not altitude above the surface:

```text
bn, be, bd = IGRF north, east, down components, nT
```

The local `north/east/down` components are then transformed back to Cartesian
components. If

```text
sin_phi = y / rho
cos_phi = x / rho
sin_theta = rho / r
cos_theta = z / r
```

then the local basis vectors are:

```text
north = (-cos_theta*cos_phi, -cos_theta*sin_phi,  sin_theta)
east  = (-sin_phi,             cos_phi,           0)
down  = (-sin_theta*cos_phi, -sin_theta*sin_phi, -cos_theta)
```

and the field is:

```text
B_cart_nT = bn*north + be*east + bd*down
B_cart_T  = B_cart_nT * 1e-9
```

The integrator uses the magnetic field in Tesla.

## Relativistic Speed

The input kinetic energy `kinetic_energy_gev` is given in GeV.

The code uses:

```text
c = 299792.458 km/s
proton rest energy = 0.9382720813 GeV
proton mass = 1.67262192369e-27 kg
elementary charge = 1.602176634e-19 C
```

The Lorentz factor is:

```text
gamma = 1 + K / mp_c2
```

where `K` is the kinetic energy in GeV and `mp_c2` is the proton rest energy in
GeV.

Then:

```text
beta^2 = 1 - 1/gamma^2
v = c * sqrt(beta^2)
```

`v` is stored as `speed_km_s`, i.e. in km/s.

## Equations of Motion

The integration variable is not time directly, but path length `s` in km. This
is convenient because the speed magnitude is constant while the direction
changes in the magnetic field.

The Lorentz force is:

```text
dp/dt = q * (v_vec x B)
```

For constant speed magnitude:

```text
v_vec = v * u
p = gamma * m * v * u
```

Therefore:

```text
du/dt = q / (gamma*m) * (u x B)
```

Since `ds/dt = v`, and `s` is measured in km in the code:

```text
du/ds_km = q / (gamma*m*v_km_s) * (u x B)
```

The ODE right-hand side in the code is:

```text
dr/ds = u
du/ds = curvature * (u x B)
```

where:

```text
curvature = charge_sign * e / (gamma * proton_mass_kg * speed_km_s)
```

Here `B` is in Tesla and `speed_km_s` is in km/s, so the derivative is with
respect to km of path length.

## Charge Sign and Backtracking

The function contains:

```text
charge_sign = sign(charge) * -1
```

This means the user-supplied `charge` is converted to an internal effective
charge with the opposite sign. This implements backward tracing: a positive
proton input is traced as a particle with the opposite sign while the integration
parameter advances forward.

In practice:

```text
charge = +1  backtracking for a proton
charge = -1  opposite sign, useful for reversibility tests
```

## RK45

One step is performed by `_rk45_step(...)`. The method is Dormand-Prince RK45:

```text
k1 ... k7
```

Using the same intermediate right-hand-side evaluations, two estimates are
constructed:

```text
y_next  5th-order solution
y4      4th-order solution
```

The difference `y_next - y4` is used as the local error estimate.

The normalized error is computed separately for position and direction:

```text
scale_position  = 1e-6  + 1e-8 * max(abs(y[i]), abs(y_next[i]))
scale_direction = 1e-10 + 1e-8 * max(abs(y[i]), abs(y_next[i]))
```

The final step error is:

```text
err = max(abs(y_next[i] - y4[i]) / scale_i)
```

The step is accepted if:

```text
err <= 1
```

After an accepted step, the direction vector is normalized again so that
numerical error does not move `u` away from unit length.

## Adaptive Step Size

The internal step is always stored as `ds` in km.

There are two modes for specifying the initial step:

```text
step_mode = 0  initial step is given as ds0_km
step_mode = 1  initial step is given as dt0_s, then ds = v * dt0_s
```

After the first step, adaptation still works in `ds`.

If a step is accepted:

```text
factor = 0.9 * err^(-0.2)
```

If a step is rejected:

```text
factor = 0.9 * err^(-0.25)
```

The factor is limited to:

```text
0.2 <= factor <= 5.0
```

If `err == 0`, the next step is increased by a factor of 5.

## Time and Path Length

Although the integration variable is `s`, the output array stores time:

```text
t = t + ds / speed_km_s
```

So `t_s` is the accumulated physical time in seconds corresponding to the
traveled path length at the relativistic speed.

The path length `s` is accumulated inside the function, but it is not stored in
the output array.

## Stopping Conditions

The parameter `stop_cond_mode` selects when tracing stops:

```text
0  radius only: r < r_min_km or r > r_max_km
1  path length only: s >= s_stop_km
2  time only: t >= t_stop_s
3  radius or path length
4  radius or time
```

For fixed-path modes, the final step is shortened so that it does not overshoot
`s_stop_km`. Similarly, for fixed-time modes, the step is shortened using the
remaining time:

```text
remaining_s = (t_stop_s - t) * speed_km_s
```

For radial boundaries there is currently no interpolation to the exact boundary
surface. Tracing stops after a step when the new point is already outside the
allowed radial interval.

## Return Status

The function returns:

```text
trajectory, n, status
```

where `n` is the index of the last real stored point. The real trajectory is:

```python
traj_real = trajectory[:n + 1]
```

Status values:

```text
 1  crossed r_max_km
 2  reached s_stop_km
 3  reached t_stop_s
 0  reached max_steps
-1  crossed r_min_km
-2  invalid initial direction, charge, or energy
-3  went below the IGRF inner validity radius
-4  invalid stop_cond_mode / step_mode / boundary settings
```

## Memory and Numba

The trajectory array is allocated at the beginning:

```text
trajectory.shape = (max_steps + 1, 7)
```

This is convenient for `numba` because the array size is known in advance and
the JIT function does not need to dynamically grow a Python list. The unused
part of the array after `n` is not part of the real trajectory.

## Current Limitations

The current implementation:

```text
- uses only IGRF;
- does not include TS05 or another external magnetospheric field;
- keeps energy and speed magnitude constant;
- uses a fixed IGRF date for the whole trace;
- integrates in Earth-centered Cartesian coordinates, not GSM;
- does not interpolate exact radial crossings at r_min/r_max;
- stores time in the output, but not accumulated path length s.
```

To add TS05, the most natural next step is to integrate in GSM, while computing
IGRF at each step by temporarily transforming the point back to GEO and then
transforming the IGRF field back to GSM.

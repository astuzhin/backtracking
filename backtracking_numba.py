from __future__ import annotations

import math

import numpy as np

from igrf14_numba import igrf14syn

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


C_KM_S = 299792.458
PROTON_REST_GEV = 0.9382720813
ELEMENTARY_CHARGE_C = 1.602176634e-19
PROTON_MASS_KG = 1.67262192369e-27

EARTH_REFERENCE_RADIUS_KM = 6371.2
IGRF_MIN_RADIUS_KM = 3485.0


@njit(cache=True)
def _norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True)
def _igrf_b_cartesian_t(date, x_km, y_km, z_km):
    """IGRF field in Earth-centered Cartesian coordinates, Tesla."""

    r = _norm3(x_km, y_km, z_km)
    rho = math.sqrt(x_km * x_km + y_km * y_km)

    if r == 0.0:
        return 0.0, 0.0, 0.0

    colat = math.acos(z_km / r) * 180.0 / math.pi
    elong = math.atan2(y_km, x_km) * 180.0 / math.pi
    if elong < 0.0:
        elong = elong + 360.0

    bn, be, bd, _ = igrf14syn(0, date, 2, r, colat, elong)

    if rho == 0.0:
        # Longitude is undefined at the pole. This convention keeps the basis
        # finite; exact-pole trajectories should normally be avoided anyway.
        sin_phi = 0.0
        cos_phi = 1.0
    else:
        sin_phi = y_km / rho
        cos_phi = x_km / rho

    sin_theta = rho / r
    cos_theta = z_km / r

    north_x = -cos_theta * cos_phi
    north_y = -cos_theta * sin_phi
    north_z = sin_theta

    east_x = -sin_phi
    east_y = cos_phi
    east_z = 0.0

    down_x = -sin_theta * cos_phi
    down_y = -sin_theta * sin_phi
    down_z = -cos_theta

    bx_nt = bn * north_x + be * east_x + bd * down_x
    by_nt = bn * north_y + be * east_y + bd * down_y
    bz_nt = bn * north_z + be * east_z + bd * down_z

    return bx_nt * 1.0e-9, by_nt * 1.0e-9, bz_nt * 1.0e-9


@njit(cache=True)
def _rhs(date, gamma, speed_km_s, charge_sign, state, dst):
    """Derivative with respect to path length in km."""

    bx, by, bz = _igrf_b_cartesian_t(date, state[0], state[1], state[2])

    ux = state[3]
    uy = state[4]
    uz = state[5]

    curvature = charge_sign * ELEMENTARY_CHARGE_C / (
        gamma * PROTON_MASS_KG * speed_km_s
    )

    dst[0] = ux
    dst[1] = uy
    dst[2] = uz

    dst[3] = curvature * (uy * bz - uz * by)
    dst[4] = curvature * (uz * bx - ux * bz)
    dst[5] = curvature * (ux * by - uy * bx)


@njit(cache=True)
def _normalize_direction(state):
    n = _norm3(state[3], state[4], state[5])
    if n > 0.0:
        state[3] = state[3] / n
        state[4] = state[4] / n
        state[5] = state[5] / n


@njit(cache=True)
def _rk45_step(date, gamma, speed_km_s, charge_sign, y, h, y_next):
    """One Dormand-Prince RK45 step. Returns an error estimate."""

    k1 = np.empty(6, dtype=np.float64)
    k2 = np.empty(6, dtype=np.float64)
    k3 = np.empty(6, dtype=np.float64)
    k4 = np.empty(6, dtype=np.float64)
    k5 = np.empty(6, dtype=np.float64)
    k6 = np.empty(6, dtype=np.float64)
    k7 = np.empty(6, dtype=np.float64)
    yt = np.empty(6, dtype=np.float64)
    y4 = np.empty(6, dtype=np.float64)

    _rhs(date, gamma, speed_km_s, charge_sign, y, k1)

    for i in range(6):
        yt[i] = y[i] + h * (1.0 / 5.0) * k1[i]
    _rhs(date, gamma, speed_km_s, charge_sign, yt, k2)

    for i in range(6):
        yt[i] = y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i])
    _rhs(date, gamma, speed_km_s, charge_sign, yt, k3)

    for i in range(6):
        yt[i] = y[i] + h * (
            (44.0 / 45.0) * k1[i]
            + (-56.0 / 15.0) * k2[i]
            + (32.0 / 9.0) * k3[i]
        )
    _rhs(date, gamma, speed_km_s, charge_sign, yt, k4)

    for i in range(6):
        yt[i] = y[i] + h * (
            (19372.0 / 6561.0) * k1[i]
            + (-25360.0 / 2187.0) * k2[i]
            + (64448.0 / 6561.0) * k3[i]
            + (-212.0 / 729.0) * k4[i]
        )
    _rhs(date, gamma, speed_km_s, charge_sign, yt, k5)

    for i in range(6):
        yt[i] = y[i] + h * (
            (9017.0 / 3168.0) * k1[i]
            + (-355.0 / 33.0) * k2[i]
            + (46732.0 / 5247.0) * k3[i]
            + (49.0 / 176.0) * k4[i]
            + (-5103.0 / 18656.0) * k5[i]
        )
    _rhs(date, gamma, speed_km_s, charge_sign, yt, k6)

    for i in range(6):
        y_next[i] = y[i] + h * (
            (35.0 / 384.0) * k1[i]
            + (500.0 / 1113.0) * k3[i]
            + (125.0 / 192.0) * k4[i]
            + (-2187.0 / 6784.0) * k5[i]
            + (11.0 / 84.0) * k6[i]
        )
    _rhs(date, gamma, speed_km_s, charge_sign, y_next, k7)

    for i in range(6):
        y4[i] = y[i] + h * (
            (5179.0 / 57600.0) * k1[i]
            + (7571.0 / 16695.0) * k3[i]
            + (393.0 / 640.0) * k4[i]
            + (-92097.0 / 339200.0) * k5[i]
            + (187.0 / 2100.0) * k6[i]
            + (1.0 / 40.0) * k7[i]
        )

    err = 0.0
    for i in range(3):
        scale = 1.0e-6 + 1.0e-8 * max(abs(y[i]), abs(y_next[i]))
        e = abs(y_next[i] - y4[i]) / scale
        if e > err:
            err = e

    for i in range(3, 6):
        scale = 1.0e-10 + 1.0e-8 * max(abs(y[i]), abs(y_next[i]))
        e = abs(y_next[i] - y4[i]) / scale
        if e > err:
            err = e

    return err


@njit(cache=True)
def trace_proton_backtracking(
    x0_km,
    y0_km,
    z0_km,
    ux0,
    uy0,
    uz0,
    charge,
    kinetic_energy_gev,
    date=2025.0,
    r_min_km=EARTH_REFERENCE_RADIUS_KM,
    r_max_km=20.0 * EARTH_REFERENCE_RADIUS_KM,
    max_steps=100000,
    ds0_km=1.0,
    stop_cond_mode=0,
    s_stop_km=0.0,
    t_stop_s=0.0,
    step_mode=0,
    dt0_s=1.0e-6,
):
    """Trace a relativistic proton backward through IGRF.

    Input coordinates are Earth-centered Cartesian km. The direction vector is
    normalized inside the function. The returned trajectory columns are:

        t_s, x_km, y_km, z_km, ux, uy, uz

    stop_cond_mode:
        0  stop only by radius: r < r_min_km or r > r_max_km
        1  stop only by path length: s >= s_stop_km
        2  stop only by time: t >= t_stop_s
        3  stop by radius or path length
        4  stop by radius or time

    step_mode:
        0  use ds0_km as the initial adaptive RK step
        1  use dt0_s as the initial adaptive RK step, converted to ds by v*dt

    Status values:
        1  crossed r_max_km
        2  reached s_stop_km
        3  reached t_stop_s
        0  reached max_steps
       -1  crossed r_min_km
       -2  invalid initial direction, charge, or energy
       -3  went below the IGRF inner radius
       -4  invalid stop condition settings
    """

    trajectory = np.empty((max_steps + 1, 7), dtype=np.float64)

    un = _norm3(ux0, uy0, uz0)
    if un == 0.0 or charge == 0.0 or kinetic_energy_gev <= 0.0:
        return trajectory, 0, -2
    if stop_cond_mode < 0 or stop_cond_mode > 4:
        return trajectory, 0, -4
    if step_mode < 0 or step_mode > 1:
        return trajectory, 0, -4
    if (stop_cond_mode == 1 or stop_cond_mode == 3) and s_stop_km <= 0.0:
        return trajectory, 0, -4
    if (stop_cond_mode == 2 or stop_cond_mode == 4) and t_stop_s <= 0.0:
        return trajectory, 0, -4
    if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r_min_km >= r_max_km:
        return trajectory, 0, -4
    if step_mode == 0 and ds0_km <= 0.0:
        return trajectory, 0, -4
    if step_mode == 1 and dt0_s <= 0.0:
        return trajectory, 0, -4

    gamma = 1.0 + kinetic_energy_gev / PROTON_REST_GEV
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    if beta2 <= 0.0:
        return trajectory, 0, -2
    speed_km_s = C_KM_S * math.sqrt(beta2)

    y = np.empty(6, dtype=np.float64)
    y_next = np.empty(6, dtype=np.float64)

    charge_sign = np.sign(charge) * -1.0

    y[0] = x0_km
    y[1] = y0_km
    y[2] = z0_km
    y[3] = ux0 / un
    y[4] = uy0 / un
    y[5] = uz0 / un

    t = 0.0
    s = 0.0
    if step_mode == 0:
        ds = ds0_km
    else:
        ds = dt0_s * speed_km_s

    trajectory[0, 0] = t
    for j in range(6):
        trajectory[0, j + 1] = y[j]

    r = _norm3(y[0], y[1], y[2])
    if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r > r_max_km:
        return trajectory, 0, 1
    if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r < r_min_km:
        return trajectory, 0, -1
    if r < IGRF_MIN_RADIUS_KM:
        return trajectory, 0, -3

    accepted = 0
    for step in range(1, max_steps + 1):
        r = _norm3(y[0], y[1], y[2])
        if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r > r_max_km:
            return trajectory, accepted, 1
        if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r < r_min_km:
            return trajectory, accepted, -1
        if (stop_cond_mode == 1 or stop_cond_mode == 3) and s >= s_stop_km:
            return trajectory, accepted, 2
        if (stop_cond_mode == 2 or stop_cond_mode == 4) and t >= t_stop_s:
            return trajectory, accepted, 3
        if r < IGRF_MIN_RADIUS_KM:
            return trajectory, accepted, -3

        ds_step = ds
        if stop_cond_mode == 1 or stop_cond_mode == 3:
            remaining_s = s_stop_km - s
            if ds_step > remaining_s:
                ds_step = remaining_s
        if stop_cond_mode == 2 or stop_cond_mode == 4:
            remaining_s = (t_stop_s - t) * speed_km_s
            if ds_step > remaining_s:
                ds_step = remaining_s
        if ds_step <= 0.0:
            if stop_cond_mode == 1 or stop_cond_mode == 3:
                return trajectory, accepted, 2
            return trajectory, accepted, 3

        err = _rk45_step(date, gamma, speed_km_s, charge_sign, y, ds_step, y_next)

        if err <= 1.0:
            s = s + ds_step
            t = t + ds_step / speed_km_s
            for j in range(6):
                y[j] = y_next[j]
            _normalize_direction(y)

            accepted = accepted + 1
            trajectory[accepted, 0] = t
            for j in range(6):
                trajectory[accepted, j + 1] = y[j]

            r = _norm3(y[0], y[1], y[2])
            if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r > r_max_km:
                return trajectory, accepted, 1
            if (stop_cond_mode == 0 or stop_cond_mode == 3 or stop_cond_mode == 4) and r < r_min_km:
                return trajectory, accepted, -1
            if (stop_cond_mode == 1 or stop_cond_mode == 3) and s >= s_stop_km:
                return trajectory, accepted, 2
            if (stop_cond_mode == 2 or stop_cond_mode == 4) and t >= t_stop_s:
                return trajectory, accepted, 3
            if r < IGRF_MIN_RADIUS_KM:
                return trajectory, accepted, -3

            if err == 0.0:
                factor = 5.0
            else:
                factor = 0.9 * err ** (-0.2)
                if factor < 0.2:
                    factor = 0.2
                if factor > 5.0:
                    factor = 5.0
            ds = ds_step * factor
        else:
            factor = 0.9 * err ** (-0.25)
            if factor < 0.2:
                factor = 0.2
            ds = ds_step * factor

    return trajectory, accepted, 0


__all__ = ["trace_proton_backtracking"]

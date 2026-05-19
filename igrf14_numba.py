"""Numba-compatible Python translation of the IGRF-14 synthesis routine.

The numerical algorithm follows ``igrf14syn`` from ``igrf14.f`` closely. The
Gauss coefficients are loaded from ``igrf14coeffs.txt`` at import time, while
the original Fortran DATA-block loader is kept as a verification path. Values
are kept as Python/NumPy float64 numbers rather than emulating legacy Fortran
default real rounding.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - keeps the module usable without numba.

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


def _load_gh_from_fortran() -> np.ndarray:
    """Load the IGRF coefficient array from the fixed-form Fortran source."""

    source_path = Path(__file__).with_name("igrf14.f")
    lines = source_path.read_text().splitlines()

    # Fixed-form Fortran only uses columns 7-72 for statements.
    statements = "\n".join(line[:72].ljust(72)[6:72] for line in lines)
    data_blocks = re.findall(
        r"data\s+(g\w)\s*/(.*?)/",
        statements,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not data_blocks:
        raise RuntimeError(f"No IGRF DATA blocks found in {source_path}")

    values: list[float] = []
    for _name, body in data_blocks:
        for token in re.split(r"[\s,]+", body.strip()):
            if not token:
                continue
            if "*" in token:
                count_text, value_text = token.split("*", 1)
                values.extend([float(value_text)] * int(count_text))
            else:
                values.append(float(token))

    if len(values) != 3840:
        raise RuntimeError(
            f"Expected 3840 IGRF coefficients from igrf14.f, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64)


def _load_gh_from_coeffs() -> np.ndarray:
    """Load the Fortran-packed IGRF coefficient array from the text table."""

    source_path = Path(__file__).with_name("igrf14coeffs.txt")
    if not source_path.exists():
        raise FileNotFoundError(f"File {source_path} not found")
    rows: list[list[float]] = []

    for line in source_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if parts[0] not in ("g", "h"):
            continue

        values = [float(token) for token in parts[3:]]
        if len(values) != 27:
            raise RuntimeError(
                f"Expected 27 epoch/SV values per IGRF coefficient row, "
                f"got {len(values)} in {source_path}"
            )
        rows.append(values)

    if len(rows) != 195:
        raise RuntimeError(
            f"Expected 195 IGRF coefficient rows from {source_path}, got {len(rows)}"
        )

    values: list[float] = []
    for epoch_index in range(19):
        for row in rows[:120]:
            values.append(row[epoch_index])

    for epoch_index in range(19, 27):
        for row in rows:
            values.append(row[epoch_index])

    if len(values) != 3840:
        raise RuntimeError(
            f"Expected 3840 IGRF coefficients from {source_path}, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64)


GH = _load_gh_from_coeffs()


@njit(cache=True)
def igrf14syn(isv, date, itype, alt, colat, elong):
    """Compute IGRF-14 field components.

    Args match the original Fortran subroutine:
        isv: 0 for main-field values, 1 for secular variation.
        date: Decimal year, valid from 1900.0 through 2035.0.
        itype: 1 for geodetic coordinates, 2 for geocentric coordinates.
        alt: Height above sea level in km when ``itype == 1``; distance from
            Earth's center in km when ``itype == 2``.
        colat: Colatitude in degrees.
        elong: East longitude in degrees.

    Returns:
        ``(x, y, z, f)`` as floats. For invalid dates, this mirrors Fortran by
        returning ``(0.0, 0.0, 0.0, 1.0e8)``.
    """

    x = 0.0
    y = 0.0
    z = 0.0

    if date < 1900.0 or date > 2035.0:
        return x, y, z, 1.0e8

    if date >= 2025.0:
        t = date - 2025.0
        tc = 1.0
        if isv == 1:
            t = 1.0
            tc = 0.0
        ll = 3450
        nmx = 13
        nc = nmx * (nmx + 2)
        kmx = (nmx + 1) * (nmx + 2) // 2
    else:
        t = 0.2 * (date - 1900.0)
        ll = int(t)
        one = float(ll)
        t = t - one

        if date < 1995.0:
            nmx = 10
            nc = nmx * (nmx + 2)
            ll = nc * ll
            kmx = (nmx + 1) * (nmx + 2) // 2
        else:
            nmx = 13
            nc = nmx * (nmx + 2)
            ll = int(0.2 * (date - 1995.0))
            ll = 120 * 19 + nc * ll
            kmx = (nmx + 1) * (nmx + 2) // 2

        tc = 1.0 - t
        if isv == 1:
            tc = -0.2
            t = 0.2

    r = alt
    one = colat * 0.017453292
    ct = math.cos(one)
    st = math.sin(one)
    one = elong * 0.017453292

    cl = np.zeros(14, dtype=np.float64)
    sl = np.zeros(14, dtype=np.float64)
    p = np.zeros(106, dtype=np.float64)
    q = np.zeros(106, dtype=np.float64)

    cl[1] = math.cos(one)
    sl[1] = math.sin(one)
    cd = 1.0
    sd = 0.0
    l = 1
    m = 1
    n = 0

    if itype != 2:
        a2 = 40680631.6
        b2 = 40408296.0
        one = a2 * st * st
        two = b2 * ct * ct
        three = one + two
        rho = math.sqrt(three)
        r = math.sqrt(alt * (alt + 2.0 * rho) + (a2 * one + b2 * two) / three)
        cd = (alt + rho) / r
        sd = (a2 - b2) / rho * ct * st / r
        one = ct
        ct = ct * cd - st * sd
        st = st * cd + one * sd

    ratio = 6371.2 / r
    rr = ratio * ratio

    p[1] = 1.0
    p[3] = st
    q[1] = 0.0
    q[3] = ct

    for k in range(2, kmx + 1):
        if n < m:
            m = 0
            n = n + 1
            rr = rr * ratio
            fn = float(n)
            gn = float(n - 1)

        fm = float(m)
        if m == n:
            if k != 3:
                one = math.sqrt(1.0 - 0.5 / fm)
                j = k - n - 1
                p[k] = one * st * p[j]
                q[k] = one * (st * q[j] + ct * p[j])
                cl[m] = cl[m - 1] * cl[1] - sl[m - 1] * sl[1]
                sl[m] = sl[m - 1] * cl[1] + cl[m - 1] * sl[1]
        else:
            gmm = m * m
            one = math.sqrt(fn * fn - float(gmm))
            two = math.sqrt(gn * gn - float(gmm)) / one
            three = (fn + gn) / one
            i = k - n
            j = i - n + 1
            p[k] = three * ct * p[i] - two * p[j]
            q[k] = three * (ct * q[i] - st * p[i]) - two * q[j]

        lm = ll + l
        one = (tc * GH[lm - 1] + t * GH[lm + nc - 1]) * rr
        if m == 0:
            x = x + one * q[k]
            z = z - (fn + 1.0) * one * p[k]
            l = l + 1
        else:
            two = (tc * GH[lm] + t * GH[lm + nc]) * rr
            three = one * cl[m] + two * sl[m]
            x = x + three * q[k]
            z = z - (fn + 1.0) * three * p[k]
            if st == 0.0:
                y = y + (one * sl[m] - two * cl[m]) * q[k] * ct
            else:
                y = y + (one * sl[m] - two * cl[m]) * fm * p[k] / st
            l = l + 2

        m = m + 1

    one = x
    x = x * cd + z * sd
    z = z * cd - one * sd
    f = math.sqrt(x * x + y * y + z * z)
    return x, y, z, f


__all__ = ["GH", "igrf14syn"]

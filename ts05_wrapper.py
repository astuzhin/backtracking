"""Small ctypes wrapper for the local TS05/T04_s Fortran model.

The Fortran routine returns the external magnetospheric magnetic field in GSM
coordinates. It does not include Earth's internal field; use IGRF separately
and add the two fields only after transforming them to the same coordinate
system.

Input coordinate system:
    X, Y, Z are GSM coordinates in Earth radii, Re = 6371.2 km.

Output coordinate system:
    BX, BY, BZ are GSM magnetic-field components in nT.

TS05/T04_s parameters:
    parmod[0]  PDYN   solar wind dynamic pressure, nPa
    parmod[1]  DST    Dst index, nT
    parmod[2]  BYIMF  IMF By in GSM, nT
    parmod[3]  BZIMF  IMF Bz in GSM, nT
    parmod[4]  W1     storm-time history parameter W1
    parmod[5]  W2     storm-time history parameter W2
    parmod[6]  W3     storm-time history parameter W3
    parmod[7]  W4     storm-time history parameter W4
    parmod[8]  W5     storm-time history parameter W5
    parmod[9]  W6     storm-time history parameter W6

Other inputs:
    ps is the geodipole tilt angle in radians.

OMNI/TS05 yearly-file records contain 23 values:
    Year, Day, Hour, Minute, BXGSM, BYGSM, BZGSM, VXGSE, VYGSE, VZGSE,
    DEN, TEMP, SYMH, IMFFLAG, ISWFLAG, TILT, Pdyn, W1, W2, W3, W4, W5, W6.

Only the following values are passed directly into T04_s:
    parmod = [Pdyn, SYMH, BYGSM, BZGSM, W1, W2, W3, W4, W5, W6]
    ps = TILT

The time columns select the correct record. BXGSM, solar-wind velocity,
density, temperature, and flags are not direct T04_s arguments in this file;
Pdyn and W1..W6 have already absorbed the needed solar-wind history.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TS05_DATA_DIR = ROOT / "ts05"


def _shared_library_suffix():
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Linux":
        return ".so"
    raise RuntimeError(f"Unsupported platform for TS05 library build: {system}")


LIB_PATH = ROOT / f"libts05{_shared_library_suffix()}"

TS05_COLUMNS = [
    "IYEAR",
    "IDAY",
    "IHOUR",
    "MIN",
    "BXGSM",
    "BYGSM",
    "BZGSM",
    "VXGSE",
    "VYGSE",
    "VZGSE",
    "DEN",
    "TEMP",
    "SYMH",
    "IMFFLAG",
    "ISWFLAG",
    "TILT",
    "Pdyn",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
]


def build_ts05_library() -> None:
    """Build the local TS05 shared library for the current platform."""

    subprocess.check_call(
        [
            "gfortran",
            "-shared",
            "-fPIC",
            "-fno-automatic",
            "-std=legacy",
            "-ffixed-line-length-none",
            str(ROOT / "ts05_fixed.f"),
            str(ROOT / "ts05_c_wrapper.f90"),
            "-o",
            str(LIB_PATH),
        ]
    )


def _load_library():
    if not LIB_PATH.exists():
        build_ts05_library()

    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ts05_field.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ts05_field.restype = None
    return lib


_LIB = None
_TS05_DATA_CACHE = {}


def ts05_params_from_omni_record(record):
    """Convert one 23-column OMNI/TS05 record to ``(parmod, ps)``.

    Args:
        record: Sequence in the yearly-file order:
            ``Year, Day, Hour, Minute, BXGSM, BYGSM, BZGSM, VXGSE, VYGSE,
            VZGSE, DEN, TEMP, SYMH, IMFFLAG, ISWFLAG, TILT, Pdyn,
            W1, W2, W3, W4, W5, W6``.

    Returns:
        ``parmod, ps`` where ``parmod`` has length 10 and ``ps`` is the
        geodipole tilt angle in radians.
    """

    arr = np.asarray(record, dtype=np.float64)
    if arr.shape != (23,):
        raise ValueError("OMNI/TS05 record must contain exactly 23 values")

    parmod = np.empty(10, dtype=np.float64)
    parmod[0] = arr[16]  # Pdyn, nPa
    parmod[1] = arr[12]  # SYMH/Dst-like index, nT
    parmod[2] = arr[5]   # BYGSM, nT
    parmod[3] = arr[6]   # BZGSM, nT
    parmod[4] = arr[17]  # W1
    parmod[5] = arr[18]  # W2
    parmod[6] = arr[19]  # W3
    parmod[7] = arr[20]  # W4
    parmod[8] = arr[21]  # W5
    parmod[9] = arr[22]  # W6

    ps = float(arr[15])
    return parmod, ps


def _timestamp_to_year_minute(timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    start = pd.Timestamp(year=ts.year, month=1, day=1)
    delta = ts - start
    minute = delta.days * 1440.0 + delta.seconds / 60.0 + delta.microseconds / 6.0e7
    return ts.year, minute


def _load_ts05_year(year, data_dir=None):
    data_root = TS05_DATA_DIR if data_dir is None else Path(data_dir)
    path = data_root / f"{int(year):04d}_OMNI_5m_with_TS05_variables.dat"
    cache_key = str(path.resolve())

    if cache_key in _TS05_DATA_CACHE:
        return _TS05_DATA_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(f"TS05 parameter file not found: {path}")

    data = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        names=TS05_COLUMNS,
        engine="python",
    )

    minute = (
        (data["IDAY"].to_numpy(dtype=np.float64) - 1.0) * 1440.0
        + data["IHOUR"].to_numpy(dtype=np.float64) * 60.0
        + data["MIN"].to_numpy(dtype=np.float64)
    )
    values = data[TS05_COLUMNS].to_numpy(dtype=np.float64)

    _TS05_DATA_CACHE[cache_key] = (minute, values)
    return minute, values


def ts05_params_at_time(timestamp, data_dir=None, max_time_diff_min=None):
    """Return ``(parmod, ps)`` for the nearest TS05 data record.

    Args:
        timestamp: Time accepted by ``pd.Timestamp``. If timezone-aware, it is
            converted to UTC before selecting the nearest yearly-file record.
        data_dir: Directory with ``YYYY_OMNI_5m_with_TS05_variables.dat`` files.
            Defaults to the package ``ts05`` directory.
        max_time_diff_min: Optional maximum allowed time difference in minutes.

    Returns:
        ``(parmod, ps)`` for the nearest record.
    """

    year, target_minute = _timestamp_to_year_minute(timestamp)
    minute, values = _load_ts05_year(year, data_dir=data_dir)

    idx = int(np.searchsorted(minute, target_minute))
    if idx <= 0:
        nearest = 0
    elif idx >= len(minute):
        nearest = len(minute) - 1
    else:
        prev_diff = abs(target_minute - minute[idx - 1])
        next_diff = abs(minute[idx] - target_minute)
        if prev_diff <= next_diff:
            nearest = idx - 1
        else:
            nearest = idx

    diff_min = abs(float(minute[nearest]) - float(target_minute))
    if max_time_diff_min is not None and diff_min > float(max_time_diff_min):
        raise ValueError(
            f"Nearest TS05 record is {diff_min:.3f} minutes away, "
            f"larger than max_time_diff_min={max_time_diff_min}"
        )

    return ts05_params_from_omni_record(values[nearest])


def ts05_field(parmod, ps, x_gsm_re, y_gsm_re, z_gsm_re):
    """Return TS05/T04_s external field in GSM coordinates.

    Args:
        parmod: Length-10 array with ``PDYN, DST, BYIMF, BZIMF, W1..W6``.
            Units are nPa for ``PDYN`` and nT for ``DST/BYIMF/BZIMF``.
        ps: Geodipole tilt angle in radians.
        x_gsm_re: GSM X coordinate in Earth radii.
        y_gsm_re: GSM Y coordinate in Earth radii.
        z_gsm_re: GSM Z coordinate in Earth radii.

    Returns:
        ``(bx, by, bz)`` in nT, GSM components, external TS05/T04_s field only.
    """

    global _LIB
    if _LIB is None:
        _LIB = _load_library()

    parmod_arr = np.ascontiguousarray(parmod, dtype=np.float64)
    if parmod_arr.shape != (10,):
        raise ValueError("parmod must contain exactly 10 values")

    ps_c = ctypes.c_double(float(ps))
    x_c = ctypes.c_double(float(x_gsm_re))
    y_c = ctypes.c_double(float(y_gsm_re))
    z_c = ctypes.c_double(float(z_gsm_re))
    bx_c = ctypes.c_double()
    by_c = ctypes.c_double()
    bz_c = ctypes.c_double()

    _LIB.ts05_field(
        parmod_arr,
        ctypes.byref(ps_c),
        ctypes.byref(x_c),
        ctypes.byref(y_c),
        ctypes.byref(z_c),
        ctypes.byref(bx_c),
        ctypes.byref(by_c),
        ctypes.byref(bz_c),
    )

    return bx_c.value, by_c.value, bz_c.value


def ts05_field_at_time(
    timestamp,
    x_gsm_re,
    y_gsm_re,
    z_gsm_re,
    data_dir=None,
    max_time_diff_min=None,
):
    """Return TS05/T04_s external field for a timestamp and GSM position.

    Args:
        timestamp: Time accepted by ``pd.Timestamp``. The nearest 5-minute TS05
            parameter record is used.
        x_gsm_re: GSM X coordinate in Earth radii.
        y_gsm_re: GSM Y coordinate in Earth radii.
        z_gsm_re: GSM Z coordinate in Earth radii.
        data_dir: Optional directory with yearly TS05 parameter files.
        max_time_diff_min: Optional maximum allowed time difference in minutes.

    Returns:
        ``(bx, by, bz)`` in nT, GSM components, external TS05/T04_s field only.
    """

    parmod, ps = ts05_params_at_time(
        timestamp,
        data_dir=data_dir,
        max_time_diff_min=max_time_diff_min,
    )
    return ts05_field(parmod, ps, x_gsm_re, y_gsm_re, z_gsm_re)


__all__ = [
    "build_ts05_library",
    "ts05_params_from_omni_record",
    "ts05_params_at_time",
    "ts05_field",
    "ts05_field_at_time",
]

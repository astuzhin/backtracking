# backtracking

Minimal Python package for proton backtracking in the IGRF-14 magnetic field.
The main function is `trace_proton_backtracking(...)`.

## Installation

Use a virtual environment. This avoids the `externally-managed-environment`
error from Homebrew/system Python.

From the project root:

```bash
cd /Users/alex/Documents/Science/Backtracking

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e ./backtracking
```

Check that the package is importable:

```bash
python -c "import backtracking; print(backtracking.GH.shape)"
```

Expected output:

```text
(3840,)
```

## Usage

```python
import backtracking as bktg

traj, n, status = bktg.trace_proton_backtracking(
    x0_km=6771.2,
    y0_km=0.0,
    z0_km=0.0,
    ux0=0.0,
    uy0=1.0,
    uz0=0.0,
    charge=1.0,
    kinetic_energy_gev=1.0,
    date=2025.0,
    stop_cond_mode=1,
    s_stop_km=1000.0,
)

traj = traj[:n + 1]
```

Trajectory columns:

```text
t_s, x_km, y_km, z_km, ux, uy, uz
```

`ux, uy, uz` are unit direction components, not velocity components.

## TS05 Field By Time

TS05 parameter files are read from `backtracking/ts05`. The timestamp is used to
select the nearest 5-minute record.

TS05 uses the local Fortran source `ts05_fixed.f`. On first use, the package
builds the shared library for the current platform:

```text
macOS: libts05.dylib
Linux: libts05.so
```

This requires `gfortran` to be available.

```python
import pandas as pd
import backtracking as bktg

bx, by, bz = bktg.ts05_field_at_time(
    pd.Timestamp("2023-01-03 01:15:00"),
    x_gsm_re=5.0,
    y_gsm_re=0.0,
    z_gsm_re=0.0,
)
```

The input position is in GSM coordinates, Earth radii `Re`. The returned field
is the external TS05/T04_s field in GSM coordinates, nT.

## Notes

- Coordinates are Earth-centered Cartesian coordinates in km.
- The current implementation uses IGRF-14 only.
- IGRF coefficients are loaded from `igrf14coeffs.txt`.
- `status` describes why tracing stopped; see `backtracking_numba_summary_en.md`
  or `backtracking_numba_summary.md` for details.

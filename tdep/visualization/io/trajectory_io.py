"""Reader for flat TDEP trajectory files (``infile.positions`` / ``infile.forces``).

These files are headerless: 3 columns per line, ``natoms * nsteps`` lines, no
blank lines or frame markers. Frame boundaries are implicit (every ``natoms``
lines). See ``memnotes/Projects/MYTRIALS/notes/tdep-file-formats.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_frames(
    path: str | Path,
    natoms: int,
    nsteps: int | None = None,
) -> np.ndarray:
    """Read a flat positions/forces file into a ``(T, N, 3)`` array.

    Parameters
    ----------
    path:
        Path to ``infile.positions`` or ``infile.forces``.
    natoms:
        Number of atoms per frame (from ``infile.meta`` / ``infile.ssposcar``).
    nsteps:
        Expected number of frames. If ``None``, it is inferred from the row
        count. If given, the row count must be exactly ``natoms * nsteps``.
    """
    path = Path(path)
    if natoms <= 0:
        raise ValueError(f"natoms must be positive, got {natoms}")

    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:  # single coordinate line
        data = data.reshape(1, -1)
    if data.shape[1] != 3:
        raise ValueError(f"{path}: expected 3 columns, found {data.shape[1]}")

    nrows = data.shape[0]
    if nrows % natoms != 0:
        raise ValueError(
            f"{path}: {nrows} rows not divisible by natoms={natoms}"
        )
    inferred = nrows // natoms
    if nsteps is not None and nsteps != inferred:
        raise ValueError(
            f"{path}: expected {natoms * nsteps} rows for nsteps={nsteps}, "
            f"found {nrows} (= {inferred} frames)"
        )

    return data.reshape(inferred, natoms, 3)

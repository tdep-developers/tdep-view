"""Reader for ``infile.meta``.

Format (one value per line, optional ``#`` comment)::

        40     # N atoms
      8000     # N timesteps
       1.0     # timestep in fs
     300.0     # temperature in K

Only the first whitespace token of each line is significant. See
``memnotes/Projects/MYTRIALS/notes/tdep-file-formats.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Meta:
    natoms: int
    nsteps: int
    dt_fs: float | None = None
    temperature: float | None = None


def read_meta(path: str | Path) -> Meta:
    """Parse ``infile.meta`` into a :class:`Meta`.

    The file is authoritative for ``natoms`` and ``nsteps``. ``dt_fs`` and
    ``temperature`` are optional and default to ``None`` if absent.
    """
    path = Path(path)
    values: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(stripped.split()[0])

    if len(values) < 2:
        raise ValueError(
            f"{path}: expected at least 2 data lines (natoms, nsteps), "
            f"found {len(values)}"
        )

    natoms = int(values[0])
    nsteps = int(values[1])
    dt_fs = float(values[2]) if len(values) > 2 else None
    temperature = float(values[3]) if len(values) > 3 else None
    return Meta(natoms=natoms, nsteps=nsteps, dt_fs=dt_fs, temperature=temperature)

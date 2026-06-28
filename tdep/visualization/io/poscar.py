"""Reader for VASP POSCAR files (``infile.ssposcar`` / ``infile.ucposcar``).

Handles the TDEP-flavoured POSCAR layout: a scale factor (possibly negative,
encoding a target volume), an optional species line, per-species counts, an
optional "Selective dynamics" line, a Direct/Cartesian mode line, and
coordinate lines that may carry trailing label text. See
``memnotes/Projects/MYTRIALS/notes/tdep-file-formats.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Poscar:
    lattice: np.ndarray          # (3,3), scale already applied
    symbols: list[str]           # length N
    scaled_positions: np.ndarray  # (N,3) fractional coordinates


def _looks_like_counts(tokens: list[str]) -> bool:
    """True if every token parses as an int (a VASP counts line)."""
    if not tokens:
        return False
    try:
        for tok in tokens:
            int(tok)
    except ValueError:
        return False
    return True


def read_poscar(path: str | Path) -> Poscar:
    """Parse a POSCAR file into a :class:`Poscar`.

    The returned ``lattice`` already includes the scale factor. Cartesian
    coordinate blocks are converted to fractional. Symbols are expanded from
    the species/counts lines (e.g. ``Pb Cs I`` + ``8 8 24`` -> 40 entries).
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"{path}: too short to be a POSCAR ({len(lines)} lines)")

    # Line 0: comment. Line 1: scale factor.
    scale = float(lines[1].split()[0])
    matrix = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)],
        dtype=float,
    )
    if scale < 0:
        # Negative scale encodes a target cell volume.
        scale = (-scale / abs(np.linalg.det(matrix))) ** (1.0 / 3.0)
    lattice = scale * matrix

    # Line 5: species names (VASP5) or directly the counts (VASP4).
    idx = 5
    tokens = lines[idx].split()
    if _looks_like_counts(tokens):
        counts = [int(t) for t in tokens]
        species: list[str] | None = None
        idx += 1
    else:
        species = tokens
        idx += 1
        counts = [int(t) for t in lines[idx].split()]
        idx += 1

    if species is not None and len(species) != len(counts):
        raise ValueError(
            f"{path}: species ({len(species)}) and counts ({len(counts)}) "
            "lengths differ"
        )

    # Optional "Selective dynamics" line, then the Direct/Cartesian mode line.
    if lines[idx].strip()[:1] in ("S", "s"):
        idx += 1
    mode = lines[idx].strip()[:1].lower()  # 'd' (direct) or 'c'/'k' (cartesian)
    idx += 1

    natoms = sum(counts)
    coords = np.array(
        [[float(x) for x in lines[idx + k].split()[:3]] for k in range(natoms)],
        dtype=float,
    )

    if mode in ("c", "k"):
        # Cartesian coords are also multiplied by the scale factor in VASP, so
        # the real positions are ``scale * coords``; convert those to fractional.
        coords = (scale * coords) @ np.linalg.inv(lattice)

    if species is not None:
        symbols = [sym for sym, n in zip(species, counts) for _ in range(n)]
    else:
        symbols = ["X"] * natoms

    return Poscar(lattice=lattice, symbols=symbols, scaled_positions=coords)

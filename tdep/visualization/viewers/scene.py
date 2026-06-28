"""Pure-numpy scene geometry for the viewers (no rendering dependency).

Everything a backend needs to draw a frame -- atom radii and colours, the
periodic-cell wireframe, and per-atom displacement magnitudes -- is computed
here so it can be unit-tested without a GPU or windowing system.

Radii/colours come from ASE's data tables when available, with deterministic
fallbacks so the module works even if ASE is not installed.
"""

from __future__ import annotations

import numpy as np

try:  # ASE is optional; fall back to defaults if absent.
    from ase.data import atomic_numbers as _Z
    from ase.data import covalent_radii as _RADII
    from ase.data.colors import jmol_colors as _COLORS

    _HAVE_ASE = True
except Exception:  # pragma: no cover - exercised only without ASE
    _HAVE_ASE = False

_FALLBACK_RADIUS = 1.0          # Angstrom, for unknown elements
_FALLBACK_COLOR = (0.7, 0.7, 0.7)


def covalent_radius(symbol: str) -> float:
    """Covalent radius (Angstrom) for an element symbol."""
    if _HAVE_ASE:
        z = _Z.get(symbol)
        if z is not None:
            return float(_RADII[z])
    return _FALLBACK_RADIUS


def cpk_color(symbol: str) -> tuple[float, float, float]:
    """RGB colour in [0, 1] for an element symbol (JMol/CPK scheme)."""
    if _HAVE_ASE:
        z = _Z.get(symbol)
        if z is not None:
            return tuple(float(c) for c in _COLORS[z])
    return _FALLBACK_COLOR


def atom_radii(symbols: list[str], scale: float = 0.5) -> np.ndarray:
    """Per-atom display radii (N,), covalent radius times ``scale``."""
    return np.array([covalent_radius(s) for s in symbols]) * scale


def atom_colors(symbols: list[str]) -> np.ndarray:
    """Per-atom RGB colours, shape (N, 3) in [0, 1]."""
    return np.array([cpk_color(s) for s in symbols])


def cell_corners(lattice: np.ndarray) -> np.ndarray:
    """The 8 Cartesian corners of the cell parallelepiped, shape (8, 3).

    Corner ``k`` uses the bits of ``k`` as fractional coordinates
    ``(c0, c1, c2)`` along the three lattice vectors.
    """
    bits = np.array([[(k >> i) & 1 for i in range(3)] for k in range(8)], dtype=float)
    return bits @ lattice


def cell_edges(lattice: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return ``(corners (8,3), edges)`` for the cell wireframe.

    An edge connects two corners whose fractional coordinates differ in exactly
    one component (12 edges of a parallelepiped).
    """
    corners = cell_corners(lattice)
    edges: list[tuple[int, int]] = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                edges.append((a, b))
    return corners, edges


def min_image_displacement(
    scaled: np.ndarray, reference: np.ndarray, lattice: np.ndarray
) -> np.ndarray:
    """Cartesian displacement of ``scaled`` from ``reference`` (both fractional,
    shape (N,3)) under the minimum-image convention. Returns (N,3) in Angstrom.
    """
    d = scaled - reference
    d -= np.round(d)            # nearest periodic image
    return d @ lattice


def displacement_magnitude(
    scaled: np.ndarray, reference: np.ndarray, lattice: np.ndarray
) -> np.ndarray:
    """Per-atom minimum-image displacement magnitude (N,) in Angstrom."""
    return np.linalg.norm(min_image_displacement(scaled, reference, lattice), axis=1)

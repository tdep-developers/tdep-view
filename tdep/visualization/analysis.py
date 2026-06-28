"""Trajectory analysis (numpy-only, no rendering dependency).

Unwrapping of periodic trajectories, per-atom vibration amplitudes, and
comparison of the trajectory-averaged structure against a reference (e.g. the
``infile.ssposcar`` equilibrium). All operations work on fractional coordinates
of shape ``(T, N, 3)`` with a fixed ``(3, 3)`` lattice (rows = lattice vectors).

Important: TDEP's ``infile.positions`` is not guaranteed to share the per-atom
ordering of ``infile.ssposcar`` (it does not for the bundled CsPbI3 data). So
averages are computed *self-referentially* (centred on each atom's own first
frame) and only compared to the reference via nearest-site matching per species.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unwrap_scaled(scaled: np.ndarray) -> np.ndarray:
    """Unwrap a wrapped fractional trajectory into a continuous one.

    For each frame-to-frame step the minimum-image displacement is taken and
    accumulated, removing artificial jumps across periodic boundaries. The first
    frame is left unchanged; later frames may fall outside ``[0, 1)``.
    """
    if scaled.ndim != 3 or scaled.shape[2] != 3:
        raise ValueError(f"expected (T, N, 3), got {scaled.shape}")
    out = np.empty_like(scaled)
    out[0] = scaled[0]
    if scaled.shape[0] > 1:
        steps = np.diff(scaled, axis=0)
        steps -= np.round(steps)            # minimum-image step
        out[1:] = scaled[0] + np.cumsum(steps, axis=0)
    return out


def average_scaled(scaled: np.ndarray) -> np.ndarray:
    """Trajectory-averaged fractional positions, wrapped into ``[0, 1)``.

    Each atom is centred on its own first frame before averaging (minimum-image),
    so the mean is unbiased for atoms that straddle a periodic boundary and needs
    no external reference. Assumes atoms vibrate about a site (no long-range
    diffusion within half a cell of the first frame).
    """
    ref0 = scaled[0]
    d = scaled - ref0
    d -= np.round(d)
    return (ref0 + d.mean(axis=0)) % 1.0


def displacement_from_mean(scaled: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Per-frame Cartesian displacement magnitude from each atom's mean, (T, N).

    A robust 'vibration amplitude' field for colouring: independent of any
    external reference or atom ordering.
    """
    mean = average_scaled(scaled)
    d = scaled - mean
    d -= np.round(d)
    return np.linalg.norm(d @ lattice, axis=2)


def vibration_rms(scaled: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Per-atom RMS vibration amplitude about the trajectory mean, (N,)."""
    inst = displacement_from_mean(scaled, lattice)
    return np.sqrt(np.mean(inst**2, axis=0))


def match_to_reference(
    average: np.ndarray,
    reference: np.ndarray,
    lattice: np.ndarray,
    symbols: list[str],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Match averaged atoms to the nearest reference site of the same species.

    Returns ``(matched_index (N,), deviation (N,), bijective)`` where
    ``matched_index[i]`` is the reference atom assigned to trajectory atom ``i``,
    ``deviation[i]`` is the minimum-image distance to it (Angstrom), and
    ``bijective`` is False if two atoms were assigned the same reference site
    (a sign of large distortion or genuinely ambiguous matching).
    """
    syms = np.asarray(symbols)
    matched = np.full(len(average), -1, dtype=int)
    deviation = np.zeros(len(average))
    for el in dict.fromkeys(symbols):                # per species
        idx = np.where(syms == el)[0]
        for i in idx:
            d = average[i][None, :] - reference[idx]
            d -= np.round(d)
            dist = np.linalg.norm(d @ lattice, axis=1)
            j = int(dist.argmin())
            matched[i] = idx[j]
            deviation[i] = dist[j]
    bijective = len(set(matched.tolist())) == len(matched)
    return matched, deviation, bijective


@dataclass
class AverageComparison:
    """Result of comparing trajectory-averaged positions to a reference."""

    symbols: list[str]
    average_scaled: np.ndarray     # (N, 3) wrapped average fractional positions
    matched_index: np.ndarray      # (N,) assigned reference-atom index
    deviation: np.ndarray          # (N,) distance to matched site (Angstrom)
    bijective: bool                # False if the site assignment collided

    @property
    def rms(self) -> float:
        """RMS of the per-atom deviation magnitudes (Angstrom)."""
        return float(np.sqrt(np.mean(self.deviation**2)))

    @property
    def max_atom(self) -> int:
        return int(np.argmax(self.deviation))

    def per_species(self) -> dict[str, dict[str, float]]:
        """Per-element ``{rms, max, count}`` of the deviation magnitudes."""
        out: dict[str, dict[str, float]] = {}
        syms = np.array(self.symbols)
        for el in dict.fromkeys(self.symbols):           # first-appearance order
            dev = self.deviation[syms == el]
            out[el] = {
                "rms": float(np.sqrt(np.mean(dev**2))),
                "max": float(dev.max()),
                "count": int(dev.size),
            }
        return out


def compare_to_reference(
    scaled: np.ndarray,
    reference: np.ndarray,
    lattice: np.ndarray,
    symbols: list[str],
) -> AverageComparison:
    """Compare trajectory-averaged positions against ``reference`` sites.

    A large deviation flags a static distortion of the time-averaged structure
    relative to the ideal reference lattice (off-centering, tilting, drift).
    Atom ordering need not match the reference: averaged atoms are matched to the
    nearest reference site of the same species.
    """
    avg = average_scaled(scaled)
    matched, deviation, bijective = match_to_reference(avg, reference, lattice, symbols)
    return AverageComparison(
        symbols=list(symbols),
        average_scaled=avg,
        matched_index=matched,
        deviation=deviation,
        bijective=bijective,
    )

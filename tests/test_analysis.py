"""Tests for trajectory analysis: unwrapping and average-vs-reference."""

from __future__ import annotations

import numpy as np

from tdep.visualization import Cell, Trajectory
from tdep.visualization import analysis


def test_unwrap_removes_boundary_jump() -> None:
    # One atom drifting steadily in +x by 0.3/step (unambiguous min image),
    # wrapping at 1.0 several times.
    t = np.arange(6) * 0.3                   # 0, 0.3, ... 1.5 -> crosses boundary
    scaled = np.zeros((6, 1, 3))
    scaled[:, 0, 0] = t % 1.0               # wrapped storage
    unwrapped = analysis.unwrap_scaled(scaled)
    # Unwrapped x should be monotonic and match the true (unwrapped) drift.
    assert np.allclose(unwrapped[:, 0, 0], t)
    assert np.all(np.diff(unwrapped[:, 0, 0]) > 0)


def test_unwrap_first_frame_unchanged() -> None:
    rng = np.random.default_rng(0)
    scaled = rng.random((4, 3, 3))
    assert np.allclose(analysis.unwrap_scaled(scaled)[0], scaled[0])


def test_unwrap_single_frame() -> None:
    scaled = np.random.default_rng(1).random((1, 5, 3))
    assert np.allclose(analysis.unwrap_scaled(scaled), scaled)


def test_average_handles_boundary_straddle() -> None:
    # Atom oscillating around 0 at the cell boundary; self-centred mean ~0 (mod 1).
    scaled = np.zeros((4, 1, 3))
    scaled[:, 0, 0] = [0.98, 0.02, 0.97, 0.03]   # ~ +/-0.02 around 0
    avg = analysis.average_scaled(scaled)
    d = (avg[0, 0] + 0.5) % 1.0 - 0.5
    assert abs(d) < 1e-9


def test_compare_to_reference_offcenter() -> None:
    lattice = np.eye(3) * 5.0
    ref = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    # Atom 0 sits 0.1 (fractional) off-centre in x on average -> 0.5 A; atom 1 centred.
    scaled = np.tile(ref, (10, 1, 1))
    scaled[:, 0, 0] += 0.1
    cmp = analysis.compare_to_reference(scaled, ref, lattice, ["Pb", "I"])
    assert np.isclose(cmp.deviation[0], 0.5, atol=1e-6)
    assert np.isclose(cmp.deviation[1], 0.0, atol=1e-9)
    assert cmp.max_atom == 0
    assert cmp.bijective
    per = cmp.per_species()
    assert np.isclose(per["Pb"]["max"], 0.5, atol=1e-6)
    assert per["I"]["count"] == 1


def test_compare_tolerates_shuffled_ordering() -> None:
    # The averaged structure equals the reference but with atoms (within a
    # species) in a different order -> nearest-site matching must still find ~0.
    lattice = np.eye(3) * 6.0
    ref = np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.0, 0.0, 0.0]])
    symbols = ["Pb", "Pb", "I"]
    # Trajectory sits exactly on the reference sites but Pb order is swapped.
    shuffled = ref[[1, 0, 2]]
    scaled = np.tile(shuffled, (5, 1, 1))
    cmp = analysis.compare_to_reference(scaled, ref, lattice, symbols)
    assert np.allclose(cmp.deviation, 0.0, atol=1e-9)
    assert cmp.bijective
    assert cmp.matched_index.tolist() == [1, 0, 2]


def test_vibration_rms() -> None:
    lattice = np.eye(3) * 10.0
    # Atom 0 oscillates +/-0.01 frac (=0.1 A) in x; atom 1 is static.
    scaled = np.zeros((4, 2, 3))
    scaled[:, 0, 0] = [0.51, 0.49, 0.51, 0.49]
    scaled[:, 1, 0] = 0.5
    rms = analysis.vibration_rms(scaled, lattice)
    assert np.isclose(rms[0], 0.1, atol=1e-9)
    assert np.isclose(rms[1], 0.0, atol=1e-12)


def test_trajectory_unwrapped_method() -> None:
    lattice = np.eye(3) * 4.0
    scaled = np.zeros((3, 1, 3))
    scaled[:, 0, 0] = [0.9, 0.0, 0.1]       # steps of +0.1 across boundary
    traj = Trajectory(cell=Cell(lattice=lattice), symbols=["H"], scaled_positions=scaled)
    uw = traj.unwrapped()
    assert np.allclose(uw.scaled_positions[:, 0, 0], [0.9, 1.0, 1.1])
    # Original is untouched.
    assert np.allclose(traj.scaled_positions[:, 0, 0], [0.9, 0.0, 0.1])

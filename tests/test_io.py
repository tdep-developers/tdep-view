"""Tests for the TDEP I/O layer.

The real ``infile.*`` files in ``tests/data`` are used as ground-truth
fixtures; synthetic cases cover edge behaviour (scale factor, Cartesian mode,
shape validation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdep.visualization import Trajectory
from tdep.visualization.io import read_frames, read_meta, read_poscar

DATA = Path(__file__).resolve().parent / "data"
PREFIX = DATA / "infile"
HAVE_FIXTURES = (DATA / "infile.ssposcar").exists() and (
    DATA / "infile.positions"
).exists()

requires_fixtures = pytest.mark.skipif(
    not HAVE_FIXTURES, reason="real infile.* fixtures not present"
)


# --------------------------------------------------------------------------- #
# Synthetic unit tests (no external data)
# --------------------------------------------------------------------------- #
def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_read_meta(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "infile.meta",
        "   40   # N atoms\n 8000 # N timesteps\n 1.0 # dt fs\n 300.0 # T\n",
    )
    meta = read_meta(p)
    assert meta.natoms == 40
    assert meta.nsteps == 8000
    assert meta.dt_fs == 1.0
    assert meta.temperature == 300.0


def test_read_meta_minimal(tmp_path: Path) -> None:
    meta = read_meta(_write(tmp_path, "infile.meta", "2\n5\n"))
    assert (meta.natoms, meta.nsteps) == (2, 5)
    assert meta.dt_fs is None and meta.temperature is None


def test_poscar_scale_applied(tmp_path: Path) -> None:
    poscar_txt = (
        "comment\n"
        "2.0\n"
        "1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n"
        "H\n1\n"
        "Direct\n"
        "0.25 0.5 0.75 site 1\n"
    )
    poscar = read_poscar(_write(tmp_path, "infile.ssposcar", poscar_txt))
    assert np.allclose(poscar.lattice, 2.0 * np.eye(3))
    assert poscar.symbols == ["H"]
    assert np.allclose(poscar.scaled_positions, [[0.25, 0.5, 0.75]])


def test_poscar_negative_scale_is_volume(tmp_path: Path) -> None:
    # Negative scale = target volume. Cube matrix det = 8 -> per-vector scale
    # such that final volume = 64.
    txt = (
        "c\n-64.0\n"
        "2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\n"
        "H\n1\nDirect\n0 0 0\n"
    )
    poscar = read_poscar(_write(tmp_path, "infile.ssposcar", txt))
    assert np.isclose(abs(np.linalg.det(poscar.lattice)), 64.0)


def test_poscar_cartesian_to_fractional(tmp_path: Path) -> None:
    txt = (
        "c\n1.0\n"
        "2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\n"
        "H\n1\nCartesian\n1.0 1.0 1.0\n"
    )
    poscar = read_poscar(_write(tmp_path, "infile.ssposcar", txt))
    assert np.allclose(poscar.scaled_positions, [[0.5, 0.5, 0.5]])


def test_poscar_cartesian_with_scale(tmp_path: Path) -> None:
    # scale=2, M=2*I -> lattice=4*I. Cartesian "1 1 1" is scaled by 2 to
    # real (2,2,2), so fractional = (2,2,2) @ inv(4I) = (0.5,0.5,0.5).
    txt = (
        "c\n2.0\n"
        "2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\n"
        "H\n1\nCartesian\n1.0 1.0 1.0\n"
    )
    poscar = read_poscar(_write(tmp_path, "infile.ssposcar", txt))
    assert np.allclose(poscar.lattice, 4.0 * np.eye(3))
    assert np.allclose(poscar.scaled_positions, [[0.5, 0.5, 0.5]])


def test_read_frames_reshape(tmp_path: Path) -> None:
    # 2 atoms, 3 frames -> 6 rows.
    rows = "\n".join(f"{i}.0 {i}.1 {i}.2" for i in range(6))
    arr = read_frames(_write(tmp_path, "infile.positions", rows), natoms=2)
    assert arr.shape == (3, 2, 3)
    assert np.isclose(arr[2, 1, 0], 5.0)


def test_read_frames_bad_natoms(tmp_path: Path) -> None:
    rows = "\n".join("0 0 0" for _ in range(5))
    with pytest.raises(ValueError):
        read_frames(_write(tmp_path, "infile.positions", rows), natoms=2)


def test_read_frames_nsteps_mismatch(tmp_path: Path) -> None:
    rows = "\n".join("0 0 0" for _ in range(6))
    with pytest.raises(ValueError):
        read_frames(_write(tmp_path, "infile.positions", rows), natoms=2, nsteps=2)


# --------------------------------------------------------------------------- #
# Integration tests against the real fixtures
# --------------------------------------------------------------------------- #
@requires_fixtures
def test_from_prefix_real_data() -> None:
    traj = Trajectory.from_prefix(PREFIX)

    # Metadata from infile.meta: 40 atoms, 8000 frames.
    assert traj.natoms == 40
    assert traj.nframes == 8000
    assert traj.dt_fs == 1.0
    assert traj.temperature == 300.0

    # Species expansion Pb*8, Cs*8, I*24.
    assert traj.symbols.count("Pb") == 8
    assert traj.symbols.count("Cs") == 8
    assert traj.symbols.count("I") == 24
    assert traj.symbols[0] == "Pb"

    # Shapes and forces present.
    assert traj.scaled_positions.shape == (8000, 40, 3)
    assert traj.forces is not None
    assert traj.forces.shape == (8000, 40, 3)
    assert traj.reference_scaled.shape == (40, 3)


@requires_fixtures
def test_frame_views_real_data() -> None:
    traj = Trajectory.from_prefix(PREFIX)
    frame = traj.frame(0)
    assert frame.scaled_positions.shape == (40, 3)
    assert np.shares_memory(frame.scaled_positions, traj.scaled_positions)
    assert frame.cartesian_positions.shape == (40, 3)
    assert frame.time == 0.0
    # Fractional coordinates stay in [0, 1) range (wrapped storage).
    assert traj.scaled_positions.min() >= -1e-6


@requires_fixtures
def test_lattice_scale_real_data() -> None:
    poscar = read_poscar(DATA / "infile.ssposcar")
    # ssposcar scale 6.25773..., diagonal 2x2x2 -> ~12.515 Å edges.
    assert np.allclose(np.diag(poscar.lattice), 2 * 6.257730007200)

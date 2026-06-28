"""Tests for the export layer.

ASE (if installed) is used as an independent oracle: we write with our own
numpy-only exporters and read back with ``ase.io.read``. Comparisons use
fractional coordinates (mod 1), which are invariant to the frame rotation the
LAMMPS exporter applies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdep.visualization import Cell, Trajectory
from tdep.visualization.exporters import get_exporter, lattice_to_lammps

ase = pytest.importorskip("ase")
from ase import io as ase_io  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def _frac_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-5) -> bool:
    """Compare fractional coords modulo 1 (minimum-image difference)."""
    d = (a - b + 0.5) % 1.0 - 0.5
    return bool(np.all(np.abs(d) < tol))


def _triclinic_traj(seed: int = 0, nframes: int = 3) -> Trajectory:
    """A triclinic cell in *general* orientation (not lower-triangular), so the
    LAMMPS exporter's frame rotation is non-trivial and actually exercised."""
    rng = np.random.default_rng(seed)
    restricted = np.array([[4.0, 0.0, 0.0], [1.3, 3.7, 0.0], [0.8, 0.6, 4.2]])
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1  # ensure a proper rotation
    lattice = restricted @ q
    scaled = rng.random((nframes, 5, 3))
    forces = rng.standard_normal((nframes, 5, 3))
    return Trajectory(
        cell=Cell(lattice=lattice),
        symbols=["Si", "Si", "O", "O", "O"],
        scaled_positions=scaled,
        forces=forces,
        dt_fs=2.0,
    )


# --------------------------------------------------------------------------- #
# lattice_to_lammps geometry
# --------------------------------------------------------------------------- #
def test_lattice_to_lammps_preserves_volume() -> None:
    lattice = np.array([[4.0, 0.0, 0.0], [1.3, 3.7, 0.0], [0.8, 0.6, 4.2]])
    cell = lattice_to_lammps(lattice)
    assert np.isclose(cell.lx * cell.ly * cell.lz, abs(np.linalg.det(lattice)))
    # Same metric (Gram matrix) => same physical cell.
    assert np.allclose(cell.matrix @ cell.matrix.T, lattice @ lattice.T)


def test_lattice_to_lammps_rejects_left_handed() -> None:
    left = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    with pytest.raises(ValueError):
        lattice_to_lammps(left)


def test_unknown_format() -> None:
    with pytest.raises(ValueError):
        get_exporter("nope")


# --------------------------------------------------------------------------- #
# extxyz round-trip via ASE
# --------------------------------------------------------------------------- #
def test_extxyz_roundtrip(tmp_path: Path) -> None:
    traj = _triclinic_traj()
    out = tmp_path / "traj.xyz"
    traj.export(out, "extxyz")

    images = ase_io.read(out, index=":")
    assert len(images) == traj.nframes
    atoms0 = images[0]
    assert atoms0.get_chemical_symbols() == traj.symbols
    assert np.allclose(atoms0.cell.array, traj.cell.lattice)
    assert _frac_close(atoms0.get_scaled_positions(wrap=False), traj.scaled_positions[0])
    # Forces survive the round-trip (original frame for extxyz).
    assert np.allclose(atoms0.get_forces(), traj.forces[0])


def test_extxyz_no_forces(tmp_path: Path) -> None:
    traj = _triclinic_traj()
    traj.forces = None
    out = tmp_path / "traj.xyz"
    traj.export(out, "extxyz")
    atoms = ase_io.read(out, index="0")
    assert "forces" not in atoms.arrays
    assert atoms.calc is None or "forces" not in atoms.calc.results


# --------------------------------------------------------------------------- #
# LAMMPS dump round-trip via ASE
# --------------------------------------------------------------------------- #
def test_lammps_roundtrip(tmp_path: Path) -> None:
    traj = _triclinic_traj()
    out = tmp_path / "traj.dump"
    traj.export(out, "lammps")

    images = ase_io.read(out, index=":", format="lammps-dump-text")
    assert len(images) == traj.nframes
    atoms0 = images[0]
    # Volume preserved and fractional coords recovered.
    assert np.isclose(atoms0.get_volume(), abs(np.linalg.det(traj.cell.lattice)))
    assert _frac_close(atoms0.get_scaled_positions(wrap=False), traj.scaled_positions[0])

    # Forces are rotated into the LAMMPS frame: per-atom magnitude is preserved
    # (rotation-invariant, implementation-independent check) and rotating back
    # by the exporter's orthogonal transform recovers the originals.
    f_lammps = atoms0.get_forces()
    assert np.allclose(
        np.linalg.norm(f_lammps, axis=1),
        np.linalg.norm(traj.forces[0], axis=1),
    )
    rotation = np.linalg.inv(traj.cell.lattice) @ lattice_to_lammps(
        traj.cell.lattice
    ).matrix
    assert np.allclose(f_lammps @ rotation.T, traj.forces[0])


def test_lammps_orthogonal_box(tmp_path: Path) -> None:
    traj = Trajectory(
        cell=Cell(lattice=np.diag([10.0, 11.0, 12.0])),
        symbols=["H", "H"],
        scaled_positions=np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
    )
    out = tmp_path / "o.dump"
    traj.export(out, "ovito")
    text = out.read_text()
    assert "ITEM: BOX BOUNDS pp pp pp" in text
    assert "xy xz yz" not in text


# --------------------------------------------------------------------------- #
# stride
# --------------------------------------------------------------------------- #
def test_stride(tmp_path: Path) -> None:
    traj = _triclinic_traj(nframes=10)
    out = tmp_path / "s.xyz"
    traj.export(out, "extxyz", stride=4)
    images = ase_io.read(out, index=":")
    assert len(images) == 3  # frames 0, 4, 8


# --------------------------------------------------------------------------- #
# Integration: real data, strided
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (DATA / "infile.positions").exists(), reason="real fixtures absent"
)
def test_real_data_export_strided(tmp_path: Path) -> None:
    traj = Trajectory.from_prefix(DATA / "infile")
    out = tmp_path / "real.xyz"
    traj.export(out, "extxyz", stride=2000)  # 4 frames of 8000
    images = ase_io.read(out, index=":")
    assert len(images) == 4
    assert len(images[0]) == 40
    assert images[0].get_chemical_symbols().count("I") == 24

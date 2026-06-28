"""LAMMPS dump exporter (OVITO-readable).

One snapshot per (strided) frame in the LAMMPS text dump format. General cells
are written in LAMMPS "restricted triclinic" form. Because fractional
coordinates are rotation-invariant, atoms are placed by recomputing Cartesian
positions in the LAMMPS frame (``scaled @ A_lammps``) rather than rotating the
original Cartesian coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TextIO

import numpy as np

from .base import BaseExporter

if TYPE_CHECKING:
    from ..models import Trajectory


class LammpsCell(NamedTuple):
    matrix: np.ndarray              # (3,3) restricted-triclinic lattice, rows = vectors
    lx: float
    ly: float
    lz: float
    xy: float
    xz: float
    yz: float


def lattice_to_lammps(lattice: np.ndarray) -> LammpsCell:
    """Convert an arbitrary lattice (rows = vectors) to LAMMPS restricted form.

    Returns the rotated lattice matrix plus the six box parameters
    ``(lx, ly, lz, xy, xz, yz)``. ``lx*ly*lz`` equals ``|det(lattice)|``.
    """
    a, b, c = lattice[0], lattice[1], lattice[2]
    lx = np.linalg.norm(a)
    if lx == 0:
        raise ValueError("degenerate cell: |a| == 0")
    ahat = a / lx
    xy = float(np.dot(b, ahat))
    ly = float(np.sqrt(max(np.dot(b, b) - xy * xy, 0.0)))
    if ly == 0:
        raise ValueError("degenerate cell: a and b are colinear")
    xz = float(np.dot(c, ahat))
    yz = float((np.dot(b, c) - xy * xz) / ly)
    lz = float(np.sqrt(max(np.dot(c, c) - xz * xz - yz * yz, 0.0)))

    matrix = np.array(
        [[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]], dtype=float
    )
    # Preserve handedness: a left-handed input cell would give a negative
    # signed volume but lx*ly*lz is always positive. Flag it rather than emit a
    # silently mirrored cell.
    if np.linalg.det(lattice) < 0:
        raise ValueError(
            "left-handed cell (negative determinant) is not supported by the "
            "LAMMPS exporter; use the extxyz exporter instead"
        )
    return LammpsCell(matrix, float(lx), ly, float(lz), xy, xz, float(yz))


class LammpsExporter(BaseExporter):
    def write(
        self,
        trajectory: "Trajectory",
        path: str | Path,
        *,
        stride: int = 1,
    ) -> None:
        traj = trajectory
        cell = lattice_to_lammps(traj.cell.lattice)
        # Orthogonal transform from the original Cartesian frame to the LAMMPS
        # frame (A_lammps = A_orig @ rotation). Positions are placed via
        # scaled @ A_lammps; vector forces must be rotated by the same matrix.
        rotation = np.linalg.inv(traj.cell.lattice) @ cell.matrix

        # Stable integer type per species, in first-appearance order.
        type_of: dict[str, int] = {}
        for sym in traj.symbols:
            if sym not in type_of:
                type_of[sym] = len(type_of) + 1
        types = np.array([type_of[s] for s in traj.symbols])

        has_forces = traj.forces is not None
        triclinic = not (cell.xy == 0.0 and cell.xz == 0.0 and cell.yz == 0.0)

        with open(path, "w") as fh:
            for i in self._frame_indices(traj.nframes, stride):
                self._write_frame(
                    fh, traj, i, cell, rotation, types, triclinic, has_forces
                )

    @staticmethod
    def _write_frame(
        fh: TextIO,
        traj: "Trajectory",
        i: int,
        cell: LammpsCell,
        rotation: np.ndarray,
        types: np.ndarray,
        triclinic: bool,
        has_forces: bool,
    ) -> None:
        cart = traj.scaled_positions[i] @ cell.matrix

        fh.write("ITEM: TIMESTEP\n")
        fh.write(f"{i}\n")
        fh.write("ITEM: NUMBER OF ATOMS\n")
        fh.write(f"{traj.natoms}\n")

        if triclinic:
            xlo_b = min(0.0, cell.xy, cell.xz, cell.xy + cell.xz)
            xhi_b = cell.lx + max(0.0, cell.xy, cell.xz, cell.xy + cell.xz)
            ylo_b = min(0.0, cell.yz)
            yhi_b = cell.ly + max(0.0, cell.yz)
            fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            fh.write(f"{xlo_b:.8f} {xhi_b:.8f} {cell.xy:.8f}\n")
            fh.write(f"{ylo_b:.8f} {yhi_b:.8f} {cell.xz:.8f}\n")
            fh.write(f"0.00000000 {cell.lz:.8f} {cell.yz:.8f}\n")
        else:
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            fh.write(f"0.00000000 {cell.lx:.8f}\n")
            fh.write(f"0.00000000 {cell.ly:.8f}\n")
            fh.write(f"0.00000000 {cell.lz:.8f}\n")

        cols = "id type element x y z"
        if has_forces:
            cols += " fx fy fz"
        fh.write(f"ITEM: ATOMS {cols}\n")

        forces = traj.forces[i] @ rotation if has_forces else None
        for a in range(traj.natoms):
            x, y, z = cart[a]
            line = (
                f"{a + 1} {types[a]} {traj.symbols[a]} "
                f"{x:.8f} {y:.8f} {z:.8f}"
            )
            if has_forces:
                fx, fy, fz = forces[a]
                line += f" {fx:.8f} {fy:.8f} {fz:.8f}"
            fh.write(line + "\n")

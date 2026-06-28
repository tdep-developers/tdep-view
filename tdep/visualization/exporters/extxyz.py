"""Extended XYZ exporter.

Extended XYZ is read natively by OVITO and ASE, so this doubles as the quickest
end-to-end sanity check. One block per (strided) frame::

    <natoms>
    Lattice="r1x r1y r1z r2x r2y r2z r3x r3y r3z" Properties=species:S:1:pos:R:3[:forces:R:3] Time=<t> pbc="T T T"
    <symbol> x y z [fx fy fz]
    ...

Positions are Cartesian (Angstrom); the lattice is written row-major (rows =
lattice vectors), matching the ASE convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np

from .base import BaseExporter

if TYPE_CHECKING:
    from ..models import Trajectory


class ExtxyzExporter(BaseExporter):
    def write(
        self,
        trajectory: "Trajectory",
        path: str | Path,
        *,
        stride: int = 1,
    ) -> None:
        traj = trajectory
        lattice = traj.cell.lattice
        lat_str = " ".join(f"{v:.8f}" for v in lattice.reshape(-1))
        pbc_str = " ".join("T" if p else "F" for p in traj.cell.pbc)
        has_forces = traj.forces is not None

        props = "species:S:1:pos:R:3"
        if has_forces:
            props += ":forces:R:3"

        with open(path, "w") as fh:
            for i in self._frame_indices(traj.nframes, stride):
                self._write_frame(fh, traj, i, lat_str, pbc_str, props, has_forces)

    @staticmethod
    def _write_frame(
        fh: TextIO,
        traj: "Trajectory",
        i: int,
        lat_str: str,
        pbc_str: str,
        props: str,
        has_forces: bool,
    ) -> None:
        cart = traj.cell.to_cartesian(traj.scaled_positions[i])
        forces = traj.forces[i] if has_forces else None
        time = i * traj.dt_fs if traj.dt_fs is not None else None

        comment = f'Lattice="{lat_str}" Properties={props} pbc="{pbc_str}"'
        if time is not None:
            comment += f" Time={time:.6f}"

        fh.write(f"{traj.natoms}\n{comment}\n")
        for a in range(traj.natoms):
            x, y, z = cart[a]
            line = f"{traj.symbols[a]:<2s} {x:18.8f} {y:18.8f} {z:18.8f}"
            if has_forces:
                fx, fy, fz = forces[a]
                line += f" {fx:18.8f} {fy:18.8f} {fz:18.8f}"
            fh.write(line + "\n")

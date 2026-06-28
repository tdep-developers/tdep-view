"""Example 2 — convert a TDEP trajectory to standard formats.

Writes an Extended XYZ file (OVITO/ASE) and a LAMMPS/OVITO dump, using a stride
to keep the output small. No visualization dependencies required.

Run:  python examples/02_export_formats.py
"""

from __future__ import annotations

from pathlib import Path

from _data import example_prefix

from tdep.visualization import Trajectory

OUT = Path(__file__).resolve().parent / "_output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    traj = Trajectory.from_prefix(example_prefix())

    stride = max(1, traj.nframes // 20)  # ~20 frames regardless of run length

    xyz = OUT / "traj.xyz"
    traj.export(xyz, "extxyz", stride=stride)
    print(f"wrote {xyz}  (forces included: {traj.forces is not None})")

    dump = OUT / "traj.dump"
    traj.export(dump, "ovito", stride=stride)
    print(f"wrote {dump}  (open in OVITO)")

    nframes = len(range(0, traj.nframes, stride))
    print(f"\nexported {nframes} frames (stride={stride}) of {traj.nframes}")


if __name__ == "__main__":
    main()

"""Example 3 — displacement analysis (no rendering needed).

Unwraps the trajectory, then compares the time-averaged structure against the
ssposcar reference to quantify static distortion, per species.

Run:  python examples/03_displacement_analysis.py
"""

from __future__ import annotations

import numpy as np

from _data import example_prefix

from tdep.visualization import Trajectory
from tdep.visualization.analysis import vibration_rms


def main() -> None:
    traj = Trajectory.from_prefix(example_prefix())

    # Per-atom RMS vibration amplitude about each atom's own mean position.
    rms = vibration_rms(traj.scaled_positions, traj.cell.lattice)
    print("per-atom RMS vibration amplitude:")
    print(f"  mean {rms.mean():.4f} Å, max {rms.max():.4f} Å "
          f"(atom {int(np.argmax(rms))}, {traj.symbols[int(np.argmax(rms))]})")

    # Time-averaged structure vs the ideal reference lattice.
    cmp = traj.compare_average_to_reference()
    print("\naveraged structure vs ssposcar reference:")
    print(f"  overall RMS deviation: {cmp.rms:.4f} Å")
    for el, s in cmp.per_species().items():
        print(f"  {el:<3s} rms={s['rms']:.4f} Å  max={s['max']:.4f} Å  (n={s['count']})")
    if not cmp.bijective:
        print("  note: site matching was not one-to-one (large distortion)")


if __name__ == "__main__":
    main()

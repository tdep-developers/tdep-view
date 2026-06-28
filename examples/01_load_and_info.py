"""Example 1 — load a trajectory and inspect it.

Run:  python examples/01_load_and_info.py
"""

from __future__ import annotations

from _data import example_prefix

from tdep.visualization import Trajectory


def main() -> None:
    traj = Trajectory.from_prefix(example_prefix())

    print(f"atoms      : {traj.natoms}")
    print(f"frames     : {traj.nframes}")
    print(f"timestep   : {traj.dt_fs} fs")
    print(f"temperature: {traj.temperature} K")
    print(f"has forces : {traj.forces is not None}")

    # Composition (first-appearance order).
    counts: dict[str, int] = {}
    for s in traj.symbols:
        counts[s] = counts.get(s, 0) + 1
    print("composition:", ", ".join(f"{el}={n}" for el, n in counts.items()))

    # A single frame is a cheap view; coordinates are available both ways.
    frame = traj.frame(0)
    print("\nframe 0, atom 0:")
    print("  fractional:", frame.scaled_positions[0])
    print("  cartesian :", frame.cartesian_positions[0], "(Å)")


if __name__ == "__main__":
    main()

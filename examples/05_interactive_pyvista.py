"""Example 5 — interactive PyVista viewer (opens a live window).

Unlike example 4 (which renders off-screen to files), this opens an interactive
3-D window: an isometric view with the periodic box drawn and a **frame slider**
to scrub through the trajectory. Requires the visualization extra and a display:

    pip install ".[viz]"

Run:  python examples/05_interactive_pyvista.py
      python examples/05_interactive_pyvista.py --deviation   # static-distortion view

Controls (PyVista defaults): left-drag rotate, scroll zoom, right-drag pan,
drag the "frame" slider to step through time.
"""

from __future__ import annotations

import argparse

from _data import example_prefix

from tdep.visualization import Trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deviation",
        action="store_true",
        help="show arrows from each ssposcar site to the time-averaged position "
        "(static distortion) instead of the animated trajectory",
    )
    parser.add_argument(
        "--color-by",
        choices=("species", "displacement"),
        default="displacement",
        help="atom colouring for the animated view (default: displacement)",
    )
    args = parser.parse_args()

    # Unwrap so atoms follow continuous paths instead of jumping across the box.
    traj = Trajectory.from_prefix(example_prefix()).unwrapped()

    try:
        if args.deviation:
            # Needs infile.ssposcar (loaded automatically by from_prefix).
            traj.view_average_deviation(arrow_scale=1.0)
        else:
            traj.view(
                backend="pyvista",
                color_by=args.color_by,
                show_forces=traj.forces is not None,
                force_scale=1.0,
            )
    except ImportError as exc:
        raise SystemExit(f"install the [viz] extra to open the viewer: {exc}")


if __name__ == "__main__":
    main()

"""Example 4 — screenshots and a movie (requires the [viz] extra).

    pip install ".[viz]"          # PyVista + VTK
    pip install ".[viz,movie]"    # + imageio for GIF/MP4

Renders off-screen, so it works headless (no display needed).

Run:  python examples/04_visualize.py
"""

from __future__ import annotations

from pathlib import Path

from _data import example_prefix

from tdep.visualization import Trajectory

OUT = Path(__file__).resolve().parent / "_output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # Unwrap so atoms don't jump across the periodic box mid-animation.
    traj = Trajectory.from_prefix(example_prefix()).unwrapped()

    shot = OUT / "frame0.png"
    traj.screenshot(shot, index=0, color_by="displacement", show_forces=True)
    print(f"wrote {shot}")

    stride = max(1, traj.nframes // 60)  # ~60-frame movie
    gif = OUT / "traj.gif"
    traj.to_movie(gif, stride=stride, fps=20, color_by="displacement")
    print(f"wrote {gif}")

    # For an interactive window instead, run:
    #   traj.view(backend="pyvista", color_by="displacement", show_forces=True)


if __name__ == "__main__":
    main()

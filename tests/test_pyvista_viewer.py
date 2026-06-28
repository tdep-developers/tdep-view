"""Off-screen smoke tests for the PyVista viewer.

These only exercise scene construction + a single off-screen render (no GUI):
the heavy geometry logic is covered separately in ``test_scene.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from tdep.visualization import Cell, Trajectory  # noqa: E402
from tdep.visualization.viewers.pyvista_viewer import PyVistaViewer  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def _small_traj(nframes: int = 3, forces: bool = True) -> Trajectory:
    rng = np.random.default_rng(0)
    return Trajectory(
        cell=Cell(lattice=np.diag([6.0, 6.0, 6.0])),
        symbols=["Pb", "I", "Cs"],
        scaled_positions=rng.random((nframes, 3, 3)),
        forces=rng.standard_normal((nframes, 3, 3)) if forces else None,
        reference_scaled=rng.random((3, 3)),
    )


def _nonblank(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] == 3 and int(np.ptp(img)) > 0


def test_screenshot_species(tmp_path: Path) -> None:
    out = tmp_path / "s.png"
    img = _small_traj().screenshot(out, index=0)
    assert out.exists()
    assert _nonblank(img)


def test_screenshot_displacement(tmp_path: Path) -> None:
    viewer = PyVistaViewer(_small_traj(), color_by="displacement")
    assert viewer._disp_clim is not None and viewer._disp_clim[0] == 0.0
    assert viewer._disp_clim[1] > 0.0
    assert _nonblank(viewer.screenshot(tmp_path / "d.png", index=1))


def test_screenshot_forces(tmp_path: Path) -> None:
    img = _small_traj().screenshot(
        tmp_path / "f.png", index=0, show_forces=True, force_scale=1.5
    )
    assert _nonblank(img)


def test_invalid_color_by() -> None:
    with pytest.raises(ValueError):
        PyVistaViewer(_small_traj(), color_by="rainbow")


def test_forces_required() -> None:
    with pytest.raises(ValueError):
        PyVistaViewer(_small_traj(forces=False), show_forces=True)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError):
        _small_traj().view(backend="vmd")


@pytest.mark.skipif(
    not (DATA / "infile.positions").exists(), reason="real fixtures absent"
)
def test_real_data_screenshot(tmp_path: Path) -> None:
    traj = Trajectory.from_prefix(DATA / "infile")
    img = traj.screenshot(tmp_path / "real.png", index=0)
    assert _nonblank(img)


def _traj_with_reference(nframes: int = 5) -> Trajectory:
    rng = np.random.default_rng(2)
    ref = rng.random((4, 3))
    # Atoms vibrate (and are slightly off-centred) around the reference sites.
    scaled = (ref[None] + 0.02 * rng.standard_normal((nframes, 4, 3))) % 1.0
    scaled[:, :, 0] += 0.03  # systematic off-centre -> nonzero deviation
    return Trajectory(
        cell=Cell(lattice=np.diag([6.0, 6.0, 6.0])),
        symbols=["Pb", "Pb", "I", "I"],
        scaled_positions=scaled % 1.0,
        reference_scaled=ref,
    )


def test_view_average_deviation_screenshot(tmp_path: Path) -> None:
    out = tmp_path / "dev.png"
    # The screenshot path renders off-screen and closes its own plotter.
    _traj_with_reference().view_average_deviation(off_screen=True, screenshot=out)
    assert out.exists() and out.stat().st_size > 0


def test_view_average_deviation_needs_reference() -> None:
    traj = Trajectory(  # no reference_scaled
        cell=Cell(lattice=np.diag([6.0, 6.0, 6.0])),
        symbols=["Pb", "I"],
        scaled_positions=np.random.default_rng(0).random((3, 2, 3)),
    )
    assert traj.reference_scaled is None
    with pytest.raises(ValueError):
        traj.view_average_deviation(off_screen=True)

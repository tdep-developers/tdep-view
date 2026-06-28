"""Off-screen movie-export smoke tests (GIF works without ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from tdep.visualization import Cell, Trajectory  # noqa: E402


def _small_traj(nframes: int = 4) -> Trajectory:
    rng = np.random.default_rng(0)
    return Trajectory(
        cell=Cell(lattice=np.diag([6.0, 6.0, 6.0])),
        symbols=["Pb", "I", "Cs"],
        scaled_positions=rng.random((nframes, 3, 3)),
    )


def test_movie_gif(tmp_path: Path) -> None:
    out = tmp_path / "traj.gif"
    result = _small_traj().to_movie(out, stride=2, fps=5)
    assert Path(result).exists()
    assert out.stat().st_size > 0


def test_movie_mp4_requires_ffmpeg(tmp_path: Path) -> None:
    # If the ffmpeg plugin is missing, we must raise a clear ImportError, not a
    # cryptic internal failure. If it is present, the file is written instead.
    try:
        import imageio_ffmpeg  # noqa: F401

        have_ffmpeg = True
    except ImportError:
        have_ffmpeg = False

    out = tmp_path / "traj.mp4"
    if have_ffmpeg:
        _small_traj().to_movie(out, fps=5)
        assert out.exists()
    else:
        with pytest.raises(ImportError, match="ffmpeg"):
            _small_traj().to_movie(out, fps=5)

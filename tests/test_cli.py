"""Tests for the tdep-view CLI.

A tiny synthetic ``infile.*`` set is written per test so these do not depend on
the bundled real fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tdep.visualization.cli import main


def _make_fixture(tmp_path: Path, with_forces: bool = True) -> Path:
    """Write a minimal 2-atom, 3-frame trajectory; return the prefix path."""
    (tmp_path / "infile.meta").write_text("2\n3\n1.0\n300.0\n")
    (tmp_path / "infile.ssposcar").write_text(
        "comment\n1.0\n"
        "5.0 0.0 0.0\n0.0 5.0 0.0\n0.0 0.0 5.0\n"
        "H He\n1 1\nDirect\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
    )
    rows = "\n".join(f"{i * 0.1:.3f} {i * 0.1:.3f} {i * 0.1:.3f}" for i in range(6))
    (tmp_path / "infile.positions").write_text(rows + "\n")
    if with_forces:
        (tmp_path / "infile.forces").write_text(rows + "\n")
    return tmp_path / "infile"


def test_info_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path))])
    out = capsys.readouterr().out
    assert rc == 0
    assert "atoms      : 2" in out
    assert "frames     : 3" in out
    assert "H=1" in out and "He=1" in out
    assert "forces     : yes" in out


def test_info_no_forces(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path, with_forces=False)), "--info"])
    assert rc == 0
    assert "forces     : no" in capsys.readouterr().out


def test_export_default_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prefix = _make_fixture(tmp_path)
    rc = main([str(prefix), "--export", "extxyz"])
    assert rc == 0
    out_file = tmp_path / "infile.xyz"
    assert out_file.exists()
    assert out_file.read_text().count("Lattice=") == 3  # one header per frame
    assert "wrote" in capsys.readouterr().out


def test_export_lammps_default_extension(tmp_path: Path) -> None:
    prefix = _make_fixture(tmp_path)
    assert main([str(prefix), "--export", "lammps"]) == 0
    assert (tmp_path / "infile.dump").exists()


def test_export_output_override(tmp_path: Path) -> None:
    prefix = _make_fixture(tmp_path)
    target = tmp_path / "custom.xyz"
    assert main([str(prefix), "--export", "xyz", "-o", str(target)]) == 0
    assert target.exists()


def test_export_stride(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prefix = _make_fixture(tmp_path)
    rc = main([str(prefix), "--export", "extxyz", "--stride", "2"])
    assert rc == 0
    # frames 0 and 2 -> 2 blocks.
    assert (tmp_path / "infile.xyz").read_text().count("Lattice=") == 2
    assert "2 frames" in capsys.readouterr().out


def test_unknown_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path)), "--export", "bogus"])
    assert rc == 2
    assert "unknown export format" in capsys.readouterr().err


def test_missing_prefix(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(tmp_path / "does_not_exist")])
    assert rc == 1
    assert "failed to load" in capsys.readouterr().err


def test_malformed_poscar_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # counts say 2 atoms but only 1 coordinate line -> IndexError in the reader,
    # which the CLI must turn into a clean error, not a traceback.
    (tmp_path / "infile.meta").write_text("2\n1\n")
    (tmp_path / "infile.ssposcar").write_text(
        "comment\n1.0\n5 0 0\n0 5 0\n0 0 5\nH He\n1 1\nDirect\n0 0 0\n"
    )
    (tmp_path / "infile.positions").write_text("0 0 0\n0 0 0\n")
    rc = main([str(tmp_path / "infile"), "--info"])
    assert rc == 1
    assert "failed to load" in capsys.readouterr().err


def test_bad_stride(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path)), "--export", "xyz", "--stride", "0"])
    assert rc == 2
    assert "stride" in capsys.readouterr().err


def test_unknown_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path)), "--backend", "vmd"])
    assert rc == 1
    assert "viewer failed" in capsys.readouterr().err


def test_average_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([str(_make_fixture(tmp_path)), "--average"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RMS deviation" in out
    assert "per species" in out


def test_export_unwrap(tmp_path: Path) -> None:
    # --unwrap should run without error and still produce an export.
    prefix = _make_fixture(tmp_path)
    rc = main([str(prefix), "--unwrap", "--export", "extxyz", "-o", str(tmp_path / "u.xyz")])
    assert rc == 0
    assert (tmp_path / "u.xyz").exists()


def test_movie_gif(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("pyvista")
    import pyvista as pv

    pv.OFF_SCREEN = True
    prefix = _make_fixture(tmp_path)
    out = tmp_path / "m.gif"
    rc = main([str(prefix), "--movie", str(out), "--stride", "2", "--fps", "5"])
    assert rc == 0
    assert out.exists()
    assert "wrote" in capsys.readouterr().out

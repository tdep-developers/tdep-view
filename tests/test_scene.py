"""Tests for the numpy-only scene geometry (no rendering)."""

from __future__ import annotations

import numpy as np
import pytest

from tdep.visualization.viewers import scene


def test_radii_and_colors_cspbi() -> None:
    radii = scene.atom_radii(["Pb", "Cs", "I"], scale=1.0)
    assert np.allclose(radii, [1.46, 2.44, 1.39], atol=0.01)
    colors = scene.atom_colors(["I"])
    assert colors.shape == (1, 3)
    assert np.allclose(colors[0], [0.58, 0.0, 0.58], atol=0.01)


def test_radius_scale() -> None:
    assert np.allclose(
        scene.atom_radii(["Pb"], scale=0.5), scene.atom_radii(["Pb"], scale=1.0) / 2
    )


def test_unknown_element_fallback() -> None:
    # A nonsense symbol must not crash; it gets the fallback radius/colour.
    assert scene.covalent_radius("Xx") == scene._FALLBACK_RADIUS
    assert scene.cpk_color("Xx") == scene._FALLBACK_COLOR


def test_cell_corners_and_edges() -> None:
    lattice = np.diag([2.0, 3.0, 4.0])
    corners, edges = scene.cell_edges(lattice)
    assert corners.shape == (8, 3)
    assert len(edges) == 12
    # Origin and the far corner are present.
    assert np.any(np.all(corners == [0, 0, 0], axis=1))
    assert np.any(np.all(corners == [2, 3, 4], axis=1))
    # Every edge length equals one lattice-vector length for an orthogonal cell.
    lengths = sorted({
        round(float(np.linalg.norm(corners[b] - corners[a])), 6) for a, b in edges
    })
    assert lengths == [2.0, 3.0, 4.0]


def test_min_image_displacement_wraps() -> None:
    lattice = np.eye(3) * 10.0
    # Atom near 0.98 vs reference 0.02: raw diff 0.96, min image -0.04 -> -0.4 A.
    scaled = np.array([[0.98, 0.0, 0.0]])
    ref = np.array([[0.02, 0.0, 0.0]])
    d = scene.min_image_displacement(scaled, ref, lattice)
    assert np.allclose(d, [[-0.4, 0.0, 0.0]])
    mag = scene.displacement_magnitude(scaled, ref, lattice)
    assert np.allclose(mag, [0.4])


def test_displacement_zero_for_reference() -> None:
    lattice = np.eye(3) * 5.0
    ref = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert np.allclose(scene.displacement_magnitude(ref, ref, lattice), [0.0, 0.0])

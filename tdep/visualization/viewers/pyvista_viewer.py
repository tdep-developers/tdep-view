"""Interactive PyVista backend.

Thin rendering layer over :mod:`tdep.visualization.viewers.scene`. PyVista is
imported lazily so the package works without it; importing this module's
:class:`PyVistaViewer` without PyVista installed raises a clear error.

Atoms are sphere glyphs (radius by element, colour by species or displacement),
the cell is a wireframe, and forces are optional arrow glyphs. ``show()`` adds a
frame slider; ``screenshot()`` renders a single frame off-screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import scene

if TYPE_CHECKING:
    from ..models import Trajectory


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the PyVista viewer requires the 'viz' extra: pip install 'tdep-view[viz]'"
        ) from exc
    return pv


class PyVistaViewer:
    def __init__(
        self,
        trajectory: "Trajectory",
        *,
        color_by: str = "species",
        show_forces: bool = False,
        force_scale: float = 1.0,
        radius_scale: float = 0.5,
        cmap: str = "viridis",
        show_box: bool = True,
    ) -> None:
        if color_by not in ("species", "displacement"):
            raise ValueError(
                f"color_by must be 'species' or 'displacement', got {color_by!r}"
            )
        if show_forces and trajectory.forces is None:
            raise ValueError("show_forces=True but the trajectory has no forces")

        self.pv = _import_pyvista()
        self.traj = trajectory
        self.color_by = color_by
        self.show_forces = show_forces
        self.force_scale = force_scale
        self.cmap = cmap
        self.show_box = show_box

        self._radii = scene.atom_radii(trajectory.symbols, scale=radius_scale)
        self._rgb = scene.atom_colors(trajectory.symbols)
        # Displacement colouring is the per-frame deviation from each atom's own
        # trajectory mean (a robust 'vibration amplitude' that does not assume the
        # positions share the ssposcar ordering). Computed once for all frames.
        if color_by == "displacement":
            from .. import analysis

            self._disp = analysis.displacement_from_mean(
                trajectory.scaled_positions, trajectory.cell.lattice
            )  # (T, N)
            hi = float(self._disp.max())
            self._disp_clim = (0.0, hi if hi > 0.0 else 1.0)
        else:
            self._disp = None
            self._disp_clim = None

    def _atoms_mesh(self, index: int):
        pv = self.pv
        cart = self.traj.cell.to_cartesian(self.traj.scaled_positions[index])
        poly = pv.PolyData(cart)
        poly["radius"] = self._radii
        if self.color_by == "species":
            poly["rgb"] = self._rgb
        else:
            poly["displacement"] = self._disp[index]
        glyph = poly.glyph(scale="radius", orient=False, geom=pv.Sphere(radius=1.0))
        return glyph

    def _forces_mesh(self, index: int):
        pv = self.pv
        cart = self.traj.cell.to_cartesian(self.traj.scaled_positions[index])
        poly = pv.PolyData(cart)
        poly["vectors"] = self.traj.forces[index]
        poly.set_active_vectors("vectors")
        return poly.glyph(orient="vectors", scale="vectors", factor=self.force_scale,
                          geom=pv.Arrow())

    def _box_mesh(self):
        pv = self.pv
        corners, edges = scene.cell_edges(self.traj.cell.lattice)
        lines = np.hstack([[2, a, b] for a, b in edges])
        return pv.PolyData(corners, lines=lines)

    # -- rendering ------------------------------------------------------- #
    def _add_frame(self, plotter, index: int) -> None:
        """Add/replace the per-frame actors (atoms, forces) on ``plotter``."""
        atoms = self._atoms_mesh(index)
        if self.color_by == "species":
            plotter.add_mesh(atoms, scalars="rgb", rgb=True, name="atoms")
        else:
            plotter.add_mesh(
                atoms, scalars="displacement", cmap=self.cmap, clim=self._disp_clim,
                name="atoms", scalar_bar_args={"title": "vibration (A)"},
            )
        if self.show_forces:
            plotter.add_mesh(self._forces_mesh(index), color="red", name="forces")

    def _new_plotter(self, off_screen: bool, window_size):
        plotter = self.pv.Plotter(off_screen=off_screen, window_size=list(window_size))
        if self.show_box:
            plotter.add_mesh(self._box_mesh(), color="black", line_width=2, name="box")
        return plotter

    def screenshot(
        self,
        path: str | Path,
        index: int = 0,
        window_size: tuple[int, int] = (1024, 768),
    ) -> np.ndarray:
        """Render frame ``index`` off-screen to ``path``; return the image array."""
        plotter = self._new_plotter(off_screen=True, window_size=window_size)
        self._add_frame(plotter, index)
        plotter.camera_position = "iso"
        img = plotter.screenshot(str(path))
        plotter.close()
        return img

    def write_movie(
        self,
        path: str | Path,
        *,
        stride: int = 1,
        fps: int = 20,
        window_size: tuple[int, int] = (1024, 768),
    ) -> Path:
        """Render every ``stride``-th frame to a movie (``.gif`` or ``.mp4``)."""
        path = Path(path)
        is_gif = path.suffix.lower() == ".gif"
        if not is_gif:
            # mp4/avi go through imageio's ffmpeg plugin; fail early and clearly.
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    f"writing {path.suffix} movies needs the ffmpeg plugin: "
                    "pip install imageio[ffmpeg]  (or use a .gif path)"
                ) from exc

        plotter = self._new_plotter(off_screen=True, window_size=window_size)
        self._add_frame(plotter, 0)
        plotter.camera_position = "iso"
        if is_gif:
            plotter.open_gif(str(path), fps=fps)
        else:
            plotter.open_movie(str(path), framerate=fps)
        try:
            for i in range(0, self.traj.nframes, stride):
                self._add_frame(plotter, i)
                plotter.write_frame()
        finally:
            plotter.close()
        return path

    def show(self, window_size: tuple[int, int] = (1024, 768)):
        """Open an interactive window with a frame slider."""
        plotter = self._new_plotter(off_screen=False, window_size=window_size)
        self._add_frame(plotter, 0)
        plotter.camera_position = "iso"

        if self.traj.nframes > 1:
            def _on_frame(value: float) -> None:
                self._add_frame(plotter, int(round(value)))

            plotter.add_slider_widget(
                _on_frame, [0, self.traj.nframes - 1], value=0, title="frame",
                fmt="%.0f",
            )
        plotter.show()
        return plotter


def plot_average_deviation(
    trajectory,
    *,
    arrow_scale: float = 1.0,
    cmap: str = "plasma",
    off_screen: bool = False,
    screenshot=None,
    window_size: tuple[int, int] = (1024, 768),
):
    """Arrows from each matched ssposcar site to the trajectory-averaged position
    (drawn at true length by default; raise ``arrow_scale`` to exaggerate the
    ~0.1-1 A deviations), coloured by deviation."""
    pv = _import_pyvista()
    cmp = trajectory.compare_average_to_reference()
    lattice = trajectory.cell.lattice
    start_frac = trajectory.reference_scaled[cmp.matched_index]
    d = cmp.average_scaled - start_frac
    d -= np.round(d)                                  # min-image vector
    start_cart = start_frac @ lattice
    vec_cart = d @ lattice

    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    corners, edges = scene.cell_edges(lattice)
    plotter.add_mesh(
        pv.PolyData(corners, lines=np.hstack([[2, a, b] for a, b in edges])),
        color="black", line_width=2,
    )
    sites = pv.PolyData(start_cart)
    sites["deviation"] = cmp.deviation
    plotter.add_mesh(
        sites.glyph(scale=False, orient=False, geom=pv.Sphere(radius=0.4)),
        scalars="deviation", cmap=cmap,
        clim=(0.0, float(cmp.deviation.max()) or 1.0),
        scalar_bar_args={"title": "deviation (A)"},
    )
    arrows = pv.PolyData(start_cart)
    arrows["vec"] = vec_cart * arrow_scale
    arrows["deviation"] = cmp.deviation
    arrows.set_active_vectors("vec")
    plotter.add_mesh(
        arrows.glyph(orient="vec", scale="vec", geom=pv.Arrow()),
        scalars="deviation", cmap=cmap, show_scalar_bar=False,
    )
    plotter.camera_position = "iso"
    if screenshot is not None:
        plotter.screenshot(str(screenshot))
        plotter.close()            # off-screen render: release the context
    elif not off_screen:
        plotter.show()
    return plotter

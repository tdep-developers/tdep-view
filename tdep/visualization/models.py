"""Core data model for tdep-view.

Positions and forces are stored columnar as ``(T, N, 3)`` arrays so that
derived quantities (Cartesian coordinates, displacements, unwrapping) stay
vectorized. A :class:`Frame` is a lightweight view into a :class:`Trajectory`,
it owns no arrays. No visualization dependencies live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import read_frames, read_meta, read_poscar


@dataclass
class Cell:
    lattice: np.ndarray                      # (3,3), rows = lattice vectors (Å)
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def to_cartesian(self, scaled: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates (..., 3) to Cartesian (..., 3)."""
        return scaled @ self.lattice


@dataclass
class Trajectory:
    cell: Cell
    symbols: list[str]                       # length N
    scaled_positions: np.ndarray             # (T, N, 3) fractional
    forces: np.ndarray | None = None         # (T, N, 3) eV/Å, or None
    reference_scaled: np.ndarray | None = None  # (N, 3) equilibrium reference
    dt_fs: float | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        pos = self.scaled_positions
        if pos.ndim != 3 or pos.shape[2] != 3:
            raise ValueError(
                f"scaled_positions must have shape (T, N, 3), got {pos.shape}"
            )
        if len(self.symbols) != pos.shape[1]:
            raise ValueError(
                f"symbols length {len(self.symbols)} != natoms {pos.shape[1]}"
            )
        if self.forces is not None and self.forces.shape != pos.shape:
            raise ValueError(
                f"forces shape {self.forces.shape} != positions {pos.shape}"
            )

    @property
    def nframes(self) -> int:
        return self.scaled_positions.shape[0]

    @property
    def natoms(self) -> int:
        return self.scaled_positions.shape[1]

    def frame(self, index: int) -> "Frame":
        """Return a lightweight view onto frame ``index``."""
        if not -self.nframes <= index < self.nframes:
            raise IndexError(index)
        return Frame(self, index % self.nframes)

    def export(
        self,
        path: str | Path,
        format: str = "extxyz",
        *,
        stride: int = 1,
    ) -> None:
        """Write the trajectory to ``path`` in ``format``.

        Supported formats: ``extxyz``/``xyz``, ``lammps``/``ovito``/``dump``.
        ``stride`` exports every Nth frame.
        """
        # Imported lazily to keep the export layer optional and avoid a cycle.
        from .exporters import get_exporter

        get_exporter(format).write(self, path, stride=stride)

    def view(self, backend: str = "pyvista", **kwargs):
        """Open an interactive viewer. Currently only ``backend='pyvista'``.

        Keyword arguments (``color_by``, ``show_forces``, ``force_scale``, ...)
        are forwarded to the backend viewer.
        """
        viewer = self._make_viewer(backend, kwargs)
        return viewer.show()

    def screenshot(self, path: str | Path, index: int = 0, *, backend: str = "pyvista", **kwargs):
        """Render a single frame to an image file (off-screen)."""
        viewer = self._make_viewer(backend, kwargs)
        return viewer.screenshot(path, index=index)

    def view_average_deviation(self, **kwargs):
        """Visualize deviation of the trajectory-averaged structure from the
        ssposcar reference: arrows from each matched site to the mean position,
        coloured by deviation magnitude. Keyword args (``arrow_scale``, ``cmap``,
        ``off_screen``, ``screenshot``, ``window_size``) go to the backend.
        """
        from .viewers.pyvista_viewer import plot_average_deviation

        return plot_average_deviation(self, **kwargs)

    def to_movie(
        self,
        path: str | Path,
        backend: str = "pyvista",
        *,
        stride: int = 1,
        fps: int = 20,
        **kwargs,
    ):
        """Render the trajectory to a movie file (``.gif`` or ``.mp4``)."""
        viewer = self._make_viewer(backend, kwargs)
        return viewer.write_movie(path, stride=stride, fps=fps)

    def _make_viewer(self, backend: str, kwargs: dict):
        if backend != "pyvista":
            raise ValueError(
                f"unknown viewer backend {backend!r}; only 'pyvista' is available"
            )
        from .viewers.pyvista_viewer import PyVistaViewer

        return PyVistaViewer(self, **kwargs)

    # -- analysis -------------------------------------------------------- #
    def unwrapped(self) -> "Trajectory":
        """Return a copy with positions unwrapped across periodic boundaries."""
        from .analysis import unwrap_scaled

        return Trajectory(
            cell=self.cell,
            symbols=self.symbols,
            scaled_positions=unwrap_scaled(self.scaled_positions),
            forces=self.forces,
            reference_scaled=self.reference_scaled,
            dt_fs=self.dt_fs,
            temperature=self.temperature,
        )

    def average_positions(self) -> np.ndarray:
        """Trajectory-averaged fractional positions (N,3), minimum-image safe.

        Self-referential (centred on each atom's first frame); no dependence on
        the ssposcar ordering.
        """
        from .analysis import average_scaled

        return average_scaled(self.scaled_positions)

    def compare_average_to_reference(self, reference: np.ndarray | None = None):
        """Compare averaged positions to a reference (ssposcar equilibrium).

        Averaged atoms are matched to the nearest reference site per species, so
        a mismatch between the positions and ssposcar ordering is tolerated.
        Returns an :class:`~tdep.visualization.analysis.AverageComparison`.
        """
        from .analysis import compare_to_reference

        ref = reference if reference is not None else self.reference_scaled
        if ref is None:
            raise ValueError("no reference positions available (need infile.ssposcar)")
        return compare_to_reference(
            self.scaled_positions, ref, self.cell.lattice, self.symbols
        )

    @classmethod
    def from_prefix(cls, prefix: str | Path = "infile") -> "Trajectory":
        """Load a trajectory from a file prefix (e.g. ``"infile"``).

        Reads ``<prefix>.meta`` (if present), ``<prefix>.ssposcar``,
        ``<prefix>.positions`` and ``<prefix>.forces`` (if present).
        """
        prefix = Path(prefix)
        base = prefix.name

        def sibling(ext: str) -> Path:
            return prefix.with_name(f"{base}.{ext}")

        poscar = read_poscar(sibling("ssposcar"))
        natoms = len(poscar.symbols)

        meta_path = sibling("meta")
        if meta_path.exists():
            meta = read_meta(meta_path)
            if meta.natoms != natoms:
                raise ValueError(
                    f"infile.meta natoms ({meta.natoms}) disagrees with "
                    f"ssposcar ({natoms})"
                )
            nsteps: int | None = meta.nsteps
            dt_fs = meta.dt_fs
            temperature = meta.temperature
        else:
            nsteps = None
            dt_fs = None
            temperature = None

        scaled = read_frames(sibling("positions"), natoms, nsteps)

        forces_path = sibling("forces")
        forces = (
            read_frames(forces_path, natoms, scaled.shape[0])
            if forces_path.exists()
            else None
        )

        return cls(
            cell=Cell(lattice=poscar.lattice),
            symbols=poscar.symbols,
            scaled_positions=scaled,
            forces=forces,
            reference_scaled=poscar.scaled_positions,
            dt_fs=dt_fs,
            temperature=temperature,
        )


@dataclass
class Frame:
    """A read-only view onto a single frame of a :class:`Trajectory`."""

    trajectory: Trajectory = field(repr=False)
    index: int

    @property
    def scaled_positions(self) -> np.ndarray:
        return self.trajectory.scaled_positions[self.index]

    @property
    def cartesian_positions(self) -> np.ndarray:
        return self.trajectory.cell.to_cartesian(self.scaled_positions)

    @property
    def forces(self) -> np.ndarray | None:
        f = self.trajectory.forces
        return None if f is None else f[self.index]

    @property
    def time(self) -> float | None:
        dt = self.trajectory.dt_fs
        return None if dt is None else self.index * dt

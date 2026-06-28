# tdep-view

Trajectory visualization and conversion for **TDEP**-style atomistic MD workflows.

`tdep-view` reads the native TDEP `infile.*` files (reduced coordinates, periodic
cell, optional forces), exposes a clean NumPy data model, and offers:

- **First-class periodic boundary conditions** — reduced coordinates throughout,
  with optional trajectory *unwrapping* to follow diffusion across cell faces.
- **Displacement analysis** — per-atom RMS vibration amplitude and comparison of
  the time-averaged structure against the `ssposcar` reference (off-centering,
  tilting, drift), with per-species statistics.
- **Exporters** — Extended XYZ and LAMMPS/OVITO dump, with no rendering
  dependencies.
- **Interactive viewer & movies** — a PyVista backend for screenshots, animated
  GIF/MP4, force-arrow overlays, and colouring by species or displacement.

The I/O, data-model, and export layers are pure NumPy; visualization
dependencies are optional extras, so the package stays light for headless use
(clusters, CI, batch conversion).

## Installation

```bash
pip install .                  # core (NumPy only): I/O, model, exporters
pip install ".[viz]"           # + PyVista interactive viewer / movies
pip install ".[notebook]"      # + ASE / nglview for Jupyter
pip install ".[movie]"         # + imageio (GIF/MP4 writing)
pip install ".[dev]"           # + pytest
```

Requires Python ≥ 3.10.

## Quickstart

### Command line

```bash
# Summary of a trajectory (default action)
tdep-view infile

# Convert to OVITO/LAMMPS dump, every 10th frame
tdep-view infile --export ovito --stride 10 -o traj.dump

# Time-averaged structure vs the ssposcar reference
tdep-view infile --average

# Render an animated GIF, unwrapped, coloured by displacement, with force arrows
tdep-view infile --movie traj.gif --unwrap --color-by displacement --forces
```

### Python API

```python
from tdep.visualization import Trajectory

traj = Trajectory.from_prefix("infile")      # reads infile.{meta,ssposcar,positions,forces}

print(traj.natoms, traj.nframes, traj.symbols[:3])
frame = traj.frame(0)
xyz = frame.cartesian_positions               # (N, 3) Cartesian, Å

traj.export("traj.xyz", "extxyz", stride=10)  # convert
cmp = traj.compare_average_to_reference()     # displacement analysis
print(cmp.rms, cmp.per_species())

traj.unwrapped().screenshot("frame0.png")     # needs the [viz] extra
```

## Documentation & examples

- **[docs/manual.md](docs/manual.md)** — the full user manual (render to PDF or a
  web page with `pandoc`/MkDocs; see the top of that file).
- **[examples/](examples/)** — short runnable scripts for each core capability and
  a Jupyter notebook (`tdep-view-demo.ipynb`).

A small real CsPbI₃ trajectory (40 atoms) ships under
[tests/data/](tests/data/) and is used both by the test suite and the examples.

## Development

```bash
pip install ".[dev,viz]"
pytest
```

The test suite runs headless: the PyVista tests render off-screen, and the
exporter/I/O integration tests use the bundled `tests/data/infile.*` fixtures.

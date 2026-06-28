---
title: "tdep-view — User Manual"
author: "TDEP-VIEW"
date: "2026"
---

<!--
Render this manual to a standalone document:

  # PDF (needs pandoc + a LaTeX engine):
  pandoc docs/manual.md -o manual.pdf

  # Self-contained HTML page:
  pandoc docs/manual.md -s -o manual.html

  # Or serve the whole docs/ folder as a website with MkDocs:
  pip install mkdocs && mkdocs serve
-->

# tdep-view — User Manual

`tdep-view` reads native **TDEP** `infile.*` trajectory files into a small,
vectorized NumPy data model and provides analysis, format conversion, and
interactive 3-D visualization. It is built specifically for periodic atomistic
MD in TDEP workflows: reduced coordinates, periodic boundary conditions, and
optional force fields are first-class throughout.

## Contents

1. [Concepts and data model](#1-concepts-and-data-model)
2. [Input files](#2-input-files)
3. [Loading a trajectory](#3-loading-a-trajectory)
4. [Analysis](#4-analysis)
5. [Exporting](#5-exporting)
6. [Visualization](#6-visualization)
7. [Command-line reference](#7-command-line-reference)
8. [Installation and extras](#8-installation-and-extras)

---

## 1. Concepts and data model

The package is layered so that the I/O, data-model, and export layers carry **no
visualization dependencies** — only the viewer needs PyVista/VTK.

| Object | Module | Meaning |
| --- | --- | --- |
| `Cell` | `tdep.visualization.models` | Lattice `(3,3)` (rows = vectors, Å) + PBC flags. |
| `Trajectory` | `tdep.visualization.models` | The whole run: `(T, N, 3)` fractional positions, optional forces, reference, metadata. |
| `Frame` | `tdep.visualization.models` | A lightweight read-only view onto one frame (owns no arrays). |

Positions and forces are stored **columnar** as `(T, N, 3)` arrays so derived
quantities (Cartesian coordinates, displacements, unwrapping) stay vectorized.

> **Atom ordering caveat.** TDEP's `infile.positions` is *not* guaranteed to
> share the per-atom ordering of `infile.ssposcar`. `tdep-view` therefore
> computes averages *self-referentially* (each atom centred on its own first
> frame) and only compares to the reference via nearest-site matching per
> species. You never have to pre-sort your files.

## 2. Input files

`Trajectory.from_prefix("infile")` reads the following siblings of the prefix:

| File | Required | Contents |
| --- | --- | --- |
| `infile.ssposcar` | **yes** | Supercell reference structure (POSCAR): lattice, species, equilibrium reduced positions. |
| `infile.positions` | **yes** | `T·N` rows of fractional positions, one atom per line. |
| `infile.meta` | no | `N atoms`, `N timesteps`, timestep (fs), temperature (K). |
| `infile.forces` | no | `T·N` rows of forces (eV/Å); enables force arrows. |

If `infile.meta` is absent, the number of steps is inferred from the row count,
and timestep/temperature are left unset.

## 3. Loading a trajectory

```python
from tdep.visualization import Trajectory

traj = Trajectory.from_prefix("infile")

traj.natoms          # e.g. 40
traj.nframes         # e.g. 8000
traj.symbols         # ['Pb', 'Cs', 'I', 'I', 'I', ...]
traj.dt_fs           # 1.0 (or None)
traj.temperature     # 300.0 (or None)
traj.cell.lattice    # (3, 3) ndarray, Å

frame = traj.frame(0)            # negative indices allowed
frame.scaled_positions           # (N, 3) fractional
frame.cartesian_positions        # (N, 3) Cartesian, Å
frame.forces                     # (N, 3) or None
frame.time                       # index * dt_fs, or None
```

## 4. Analysis

All analysis is pure NumPy and reference-free unless you ask otherwise.

### Unwrapping periodic boundaries

```python
uw = traj.unwrapped()    # minimum-image step accumulation; first frame unchanged
```

Removes the artificial jumps when atoms cross a cell face — essential for seeing
diffusion as continuous paths.

### Trajectory-averaged structure

```python
avg = traj.average_positions()          # (N, 3) wrapped fractional mean
```

Each atom is centred on its own first frame before averaging, so atoms that
straddle a boundary are averaged correctly.

### Deviation from the reference

```python
cmp = traj.compare_average_to_reference()   # needs infile.ssposcar
cmp.rms                 # overall RMS deviation (Å)
cmp.deviation           # (N,) per-atom distance to matched site
cmp.per_species()       # {'Pb': {'rms':..., 'max':..., 'count':...}, ...}
cmp.bijective           # False => site matching collided (large distortion)
```

A large deviation flags a **static distortion** of the time-averaged structure
relative to the ideal lattice (off-centering, tilting, drift). On the command
line this is `tdep-view infile --average`.

## 5. Exporting

```python
traj.export("traj.xyz",  "extxyz", stride=1)    # Extended XYZ (OVITO, ASE)
traj.export("traj.dump", "ovito",  stride=10)   # LAMMPS / OVITO dump
```

Format aliases:

| Format | Aliases | Notes |
| --- | --- | --- |
| Extended XYZ | `extxyz`, `xyz` | Cartesian positions; forces included when present; read natively by OVITO and ASE. |
| LAMMPS dump | `lammps`, `ovito`, `dump` | OVITO-friendly atomic dump. |

`stride=N` writes every Nth frame.

## 6. Visualization

Visualization needs the `[viz]` extra (`pip install ".[viz]"`). The viewer
renders off-screen for screenshots/movies (works headless) or opens an
interactive window.

### Screenshots

```python
traj.screenshot("frame0.png", index=0,
                color_by="displacement",     # or "species"
                show_forces=True, force_scale=1.5)
```

### Movies

```python
traj.to_movie("traj.gif", stride=5, fps=20, color_by="species")
traj.to_movie("traj.mp4", fps=30)            # MP4 needs imageio-ffmpeg
```

### Interactive window

```python
traj.unwrapped().view(backend="pyvista", color_by="displacement", show_forces=True)
```

Opens an isometric 3-D view with the periodic box drawn and a **frame slider**
to scrub through time. Colour options:

- `color_by="species"` — one colour per element.
- `color_by="displacement"` — per-atom instantaneous displacement from its mean
  position (blue → small motion, towards the colormap max → large motion),
  highlighting diffusion, melting, and defect motion at a glance.

### Average-deviation view

```python
traj.view_average_deviation(arrow_scale=1.0, cmap="plasma",
                            off_screen=True, screenshot="deviation.png")
```

Draws an arrow from each reference site to the matched time-averaged position,
coloured by deviation magnitude — a direct picture of static distortion.

## 7. Command-line reference

```
tdep-view PREFIX [options]
```

| Option | Effect |
| --- | --- |
| *(none)* / `--info` | Print a summary of the trajectory (default). |
| `--average` | Report averaged positions vs the `ssposcar` reference. |
| `--export FORMAT` | Export: `extxyz`, `xyz`, `lammps`, `ovito`, `dump`. |
| `-o, --output PATH` | Output path for `--export` (default `<prefix>.<ext>`). |
| `--stride N` | Use every Nth frame for movie/export. |
| `--unwrap` | Unwrap across periodic boundaries before view/movie/export. |
| `--backend NAME` | Launch the interactive viewer (e.g. `pyvista`). |
| `--movie PATH` | Render a movie to `.gif`/`.mp4`. |
| `--color-by {species,displacement}` | Atom colouring (default `species`). |
| `--forces` | Draw force arrows. |
| `--force-scale F` | Force-arrow scale (default 1.0). |
| `--fps N` | Movie frame rate (default 20). |

Examples:

```bash
tdep-view infile
tdep-view infile --average
tdep-view infile --export ovito --stride 10 -o traj.dump
tdep-view infile --movie traj.gif --unwrap --color-by displacement --forces
tdep-view infile --backend pyvista --color-by displacement
```

## 8. Installation and extras

```bash
pip install .                  # core: NumPy only
pip install ".[viz]"           # PyVista + VTK (viewer, screenshots, movies)
pip install ".[notebook]"      # ASE + nglview + ipywidgets (Jupyter)
pip install ".[movie]"         # imageio (GIF/MP4 writers)
pip install ".[dev]"           # pytest
```

Python ≥ 3.10 is required. The core stack imports without any visualization
dependency, so headless conversion and analysis work on clusters and in CI.

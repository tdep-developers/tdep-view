# Examples

Short, runnable scripts for each core capability. They run out of the box
against the small CsPbI₃ trajectory bundled in [`../tests/data`](../tests/data);
set `TDEP_VIEW_PREFIX=/path/to/your/infile` to run them on your own data.

Run from the package root:

```bash
python examples/01_load_and_info.py        # load + inspect a trajectory
python examples/02_export_formats.py       # convert to Extended XYZ + LAMMPS dump
python examples/03_displacement_analysis.py  # vibration amplitudes + distortion
python examples/04_visualize.py            # screenshots + movie  (needs [viz])
python examples/05_interactive_pyvista.py  # live 3-D window      (needs [viz])
```

Scripts 1–3 need only NumPy. Scripts 4–5 need the visualization extra
(script 5 also needs a display, since it opens an interactive window):

```bash
pip install ".[viz,movie]"
```

Example 5 takes a couple of flags:

```bash
python examples/05_interactive_pyvista.py --color-by species
python examples/05_interactive_pyvista.py --deviation   # static-distortion arrows
```

Outputs from scripts 2 and 4 are written to `examples/_output/` (git-ignored).

There is also a Jupyter notebook, **[`tdep-view-demo.ipynb`](tdep-view-demo.ipynb)**,
that walks through loading, analysis, and rendering an inline movie.

"""Viewer layer: rendering backends.

`scene` is numpy-only (no rendering dependency). `pyvista_viewer` imports
PyVista lazily so the package imports fine without it installed.
"""

from . import scene

__all__ = ["scene"]

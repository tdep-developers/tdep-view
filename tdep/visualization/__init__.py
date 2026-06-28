"""tdep-view: trajectory visualization for TDEP-style workflows.

Public entry points live here; the I/O and data-model layers carry no
visualization dependencies (see design_tdep-view.md, section 2.1).
"""

from .models import Cell, Frame, Trajectory

__all__ = ["Cell", "Frame", "Trajectory"]

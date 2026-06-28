"""Shared helper: locate the bundled CsPbI3 example trajectory.

The examples run out of the box against the small real trajectory that ships
under ``tests/data``. Point ``TDEP_VIEW_PREFIX`` at your own ``infile`` prefix
to run them on your data instead.
"""

from __future__ import annotations

import os
from pathlib import Path

_BUNDLED = Path(__file__).resolve().parent.parent / "tests" / "data" / "infile"


def example_prefix() -> Path:
    """Return the trajectory prefix to use (env override or bundled data)."""
    env = os.environ.get("TDEP_VIEW_PREFIX")
    return Path(env) if env else _BUNDLED

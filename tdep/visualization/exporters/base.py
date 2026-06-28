"""Base exporter interface (design section 12.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Trajectory


class BaseExporter(ABC):
    """Write a :class:`~tdep.visualization.models.Trajectory` to a file.

    Subclasses implement :meth:`write`. A ``stride`` selects every Nth frame.
    """

    @abstractmethod
    def write(
        self,
        trajectory: "Trajectory",
        path: str | Path,
        *,
        stride: int = 1,
    ) -> None: ...

    @staticmethod
    def _frame_indices(nframes: int, stride: int) -> range:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        return range(0, nframes, stride)

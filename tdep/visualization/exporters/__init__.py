"""Export layer: trajectory -> standard formats. No visualization dependencies.

Numpy-only writers (design section 2.2 keeps the export layer dependency-free).
"""

from .base import BaseExporter
from .extxyz import ExtxyzExporter
from .lammps import LammpsExporter, lattice_to_lammps

EXPORTERS: dict[str, type[BaseExporter]] = {
    "extxyz": ExtxyzExporter,
    "xyz": ExtxyzExporter,
    "lammps": LammpsExporter,
    "ovito": LammpsExporter,
    "dump": LammpsExporter,
}


def get_exporter(fmt: str) -> BaseExporter:
    """Return an exporter instance for a format name (case-insensitive)."""
    try:
        return EXPORTERS[fmt.lower()]()
    except KeyError:
        raise ValueError(
            f"unknown export format {fmt!r}; known: {sorted(set(EXPORTERS))}"
        ) from None


__all__ = [
    "BaseExporter",
    "ExtxyzExporter",
    "LammpsExporter",
    "lattice_to_lammps",
    "EXPORTERS",
    "get_exporter",
]

"""Command-line interface for tdep-view.

Currently wires the parser and exporters together; viewer/movie backends are
recognized but not yet implemented. Entry point: ``tdep-view`` (see pyproject).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .exporters import EXPORTERS
from .models import Trajectory

# Default output extension per export format. The set of *valid* formats is the
# exporter registry (EXPORTERS); this map only supplies the default extension.
_DEFAULT_EXT = {
    "extxyz": "xyz",
    "xyz": "xyz",
    "lammps": "dump",
    "ovito": "dump",
    "dump": "dump",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdep-view",
        description="Visualize / convert TDEP-style MD trajectories.",
    )
    p.add_argument(
        "prefix",
        help="trajectory file prefix, e.g. 'infile' (reads infile.meta, "
        "infile.ssposcar, infile.positions, infile.forces)",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="print a summary of the trajectory and exit (default action)",
    )
    p.add_argument(
        "--export",
        metavar="FORMAT",
        help=f"export format: {', '.join(sorted(EXPORTERS))}",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="output path for --export (default: <prefix>.<ext>)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="use every Nth frame for movie/export (default: 1)",
    )
    p.add_argument(
        "--average",
        action="store_true",
        help="report trajectory-averaged positions vs the ssposcar reference",
    )
    p.add_argument(
        "--unwrap",
        action="store_true",
        help="unwrap positions across periodic boundaries before view/movie/export",
    )
    p.add_argument(
        "--backend",
        metavar="NAME",
        help="launch interactive viewer with this backend (e.g. pyvista)",
    )
    p.add_argument("--movie", metavar="PATH", help="render a movie to this path (.gif/.mp4)")

    view = p.add_argument_group("view / movie options")
    view.add_argument(
        "--color-by",
        choices=("species", "displacement"),
        default="species",
        help="atom colouring (default: species)",
    )
    view.add_argument("--forces", action="store_true", help="draw force arrows")
    view.add_argument(
        "--force-scale", type=float, default=1.0, metavar="F", help="force arrow scale"
    )
    view.add_argument("--fps", type=int, default=20, metavar="N", help="movie frame rate")
    return p


def _viewer_kwargs(args: argparse.Namespace) -> dict:
    return {
        "color_by": args.color_by,
        "show_forces": args.forces,
        "force_scale": args.force_scale,
    }


def _print_average(traj: Trajectory) -> int:
    if traj.reference_scaled is None:
        print(
            "tdep-view: --average needs reference positions from infile.ssposcar",
            file=sys.stderr,
        )
        return 1
    cmp = traj.compare_average_to_reference()
    i = cmp.max_atom
    print("Trajectory-averaged positions vs ssposcar reference")
    print("(averaged atoms matched to nearest reference site per species):")
    print(f"  overall RMS deviation : {cmp.rms:.4f} A")
    print(f"  max deviation         : {cmp.deviation[i]:.4f} A "
          f"(atom {i}, {cmp.symbols[i]})")
    print("  per species           :")
    for el, s in cmp.per_species().items():
        print(f"    {el:<3s} rms={s['rms']:.4f} A  max={s['max']:.4f} A  (n={s['count']})")
    if not cmp.bijective:
        print("  warning: site matching was not one-to-one (large distortion or "
              "ambiguous sites)")
    return 0


def _print_info(traj: Trajectory) -> None:
    species: dict[str, int] = {}
    for s in traj.symbols:
        species[s] = species.get(s, 0) + 1
    comp = ", ".join(f"{el}={n}" for el, n in species.items())
    lat = traj.cell.lattice

    print(f"atoms      : {traj.natoms}")
    print(f"frames     : {traj.nframes}")
    print(f"composition: {comp}")
    if traj.dt_fs is not None:
        print(f"timestep   : {traj.dt_fs} fs")
    if traj.temperature is not None:
        print(f"temperature: {traj.temperature} K")
    print(f"forces     : {'yes' if traj.forces is not None else 'no'}")
    print("cell (A)   :")
    for row in lat:
        print(f"             {row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.stride < 1:
        print(f"tdep-view: --stride must be >= 1, got {args.stride}", file=sys.stderr)
        return 2

    try:
        traj = Trajectory.from_prefix(args.prefix)
    except (OSError, ValueError, IndexError) as exc:
        print(f"tdep-view: failed to load {args.prefix!r}: {exc}", file=sys.stderr)
        return 1

    if args.average:
        return _print_average(traj)

    if args.unwrap:
        traj = traj.unwrapped()

    if args.movie is not None:
        try:
            out = traj.to_movie(
                args.movie, stride=args.stride, fps=args.fps, **_viewer_kwargs(args)
            )
        except (OSError, ValueError, ImportError) as exc:
            print(f"tdep-view: movie export failed: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {out}")
        return 0

    if args.backend is not None:
        try:
            traj.view(backend=args.backend, **_viewer_kwargs(args))
        except (ValueError, ImportError) as exc:
            print(f"tdep-view: viewer failed: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.export is not None:
        fmt = args.export.lower()
        if fmt not in EXPORTERS:
            print(
                f"tdep-view: unknown export format {args.export!r}; "
                f"choose from {', '.join(sorted(EXPORTERS))}",
                file=sys.stderr,
            )
            return 2
        out = (
            Path(args.output)
            if args.output is not None
            else Path(f"{args.prefix}.{_DEFAULT_EXT.get(fmt, 'out')}")
        )
        try:
            traj.export(out, fmt, stride=args.stride)
        except (OSError, ValueError) as exc:
            print(f"tdep-view: export failed: {exc}", file=sys.stderr)
            return 1
        nframes = len(range(0, traj.nframes, args.stride))
        print(f"wrote {out} ({nframes} frames, format={fmt})")
        return 0

    # Default action: print info.
    _print_info(traj)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rename legacy pNNN/mNNN scan directories using measured coordinate values.

The default mode is a non-mutating dry run.  Use ``--apply`` only after reading
the complete plan.  Destination collisions, duplicate rounded labels, malformed
geometries, and inconsistent paired GS/TDDFT geometries are fatal.

Scientific output files are not rewritten: the utility renames the GS and/or
TDDFT job directories and writes an auditable CSV mapping.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LEGACY_LABEL = re.compile(r"^[pm]\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class Bond:
    coefficient: float
    atom_i: int
    atom_j: int
    label: str


@dataclass(frozen=True)
class Rename:
    kind: str
    legacy_label: str
    coordinate: float
    source: Path
    destination: Path
    geometry: Path


def read_xyz(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: incomplete XYZ file")
    natoms = int(lines[0].strip())
    if len(lines) < natoms + 2:
        raise ValueError(f"{path}: expected {natoms} atoms")
    coordinates = []
    for line_number, line in enumerate(lines[2 : 2 + natoms], 3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: malformed XYZ row")
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return coordinates


def read_bonds(path: Path, zero_based: bool) -> list[Bond]:
    bonds = []
    for line_number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(
                f"{path}:{line_number}: expected coefficient atom_i atom_j [label]"
            )
        coefficient = float(fields[0])
        atom_i, atom_j = int(fields[1]), int(fields[2])
        if not zero_based:
            atom_i -= 1
            atom_j -= 1
        label = " ".join(fields[3:]) or f"{atom_i + 1}-{atom_j + 1}"
        bonds.append(Bond(coefficient, atom_i, atom_j, label))
    if not bonds:
        raise ValueError(f"{path}: no coordinate definitions found")
    return bonds


def coordinate_value(
    coordinates: list[tuple[float, float, float]], bonds: Iterable[Bond]
) -> float:
    total = 0.0
    for bond in bonds:
        try:
            left, right = coordinates[bond.atom_i], coordinates[bond.atom_j]
        except IndexError as exc:
            raise ValueError(f"Bond {bond.label!r} refers outside the XYZ geometry") from exc
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
        total += bond.coefficient * distance
    return total


def geometry_for(kind: str, directory: Path) -> Path:
    candidates = (
        ("geom.xyz",)
        if kind == "tddft"
        else (
            "optimized.xyz",
            "job.xyz",
            "optimized_from_output.xyz",
            "start_constrained.xyz",
        )
    )
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{directory}: could not locate an XYZ geometry ({', '.join(candidates)})"
    )


def discover_kind(
    *,
    kind: str,
    root: Path,
    prefix: str,
    bonds: list[Bond],
    decimals: int,
) -> list[Rename]:
    renames = []
    if not root.is_dir():
        return renames
    for source in sorted(path for path in root.iterdir() if path.is_dir()):
        if not source.name.startswith(prefix):
            continue
        legacy_label = source.name[len(prefix) :]
        if not LEGACY_LABEL.fullmatch(legacy_label):
            continue
        geometry = geometry_for(kind, source)
        coordinate = coordinate_value(read_xyz(geometry), bonds)
        new_label = f"{coordinate:.{decimals}f}"
        destination = root / f"{prefix}{new_label}"
        renames.append(
            Rename(kind, legacy_label, coordinate, source, destination, geometry)
        )
    return renames


def validate_plan(renames: list[Rename], pair_tolerance: float) -> None:
    destinations: dict[Path, Path] = {}
    sources = {rename.source for rename in renames}
    for rename in renames:
        if rename.destination in destinations:
            raise ValueError(
                f"Duplicate destination {rename.destination}: "
                f"{destinations[rename.destination]} and {rename.source}"
            )
        destinations[rename.destination] = rename.source
        if rename.destination.exists() and rename.destination not in sources:
            raise FileExistsError(
                f"Destination already exists: {rename.destination}. "
                "No existing data will be merged or overwritten."
            )

    by_legacy: dict[str, list[Rename]] = {}
    for rename in renames:
        by_legacy.setdefault(rename.legacy_label.lower(), []).append(rename)
    for legacy_label, paired in by_legacy.items():
        if len(paired) < 2:
            continue
        values = [rename.coordinate for rename in paired]
        spread = max(values) - min(values)
        if spread > pair_tolerance:
            details = ", ".join(
                f"{rename.kind}={rename.coordinate:.12f}" for rename in paired
            )
            raise ValueError(
                f"Legacy pair {legacy_label} disagrees by {spread:.3g} A "
                f"(tolerance {pair_tolerance:g} A): {details}"
            )


def write_manifest(path: Path, renames: list[Rename], status: str) -> None:
    fields = [
        "kind",
        "legacy_label",
        "coordinate_A",
        "old_path",
        "new_path",
        "geometry_used",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rename in renames:
            writer.writerow(
                {
                    "kind": rename.kind,
                    "legacy_label": rename.legacy_label,
                    "coordinate_A": f"{rename.coordinate:.12f}",
                    "old_path": str(rename.source.resolve()),
                    "new_path": str(rename.destination.resolve()),
                    "geometry_used": str(rename.geometry.resolve()),
                    "status": status,
                }
            )


def apply_plan(renames: list[Rename]) -> None:
    # A temporary phase makes the operation safe even if a future naming scheme
    # permits a destination to equal another source.
    staged: list[tuple[Rename, Path]] = []
    try:
        for rename in renames:
            temporary = rename.source.with_name(
                f".rename-coordinate-{uuid.uuid4().hex}-{rename.source.name}"
            )
            rename.source.rename(temporary)
            staged.append((rename, temporary))
        for rename, temporary in staged:
            temporary.rename(rename.destination)
    except Exception:
        # Best-effort rollback.  Never overwrite an occupied original path.
        for rename, temporary in reversed(staged):
            if temporary.exists() and not rename.source.exists():
                temporary.rename(rename.source)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--workdir", type=Path, default=Path("gs_scan_then_tddft"))
    parser.add_argument("--bonds-file", required=True, type=Path)
    parser.add_argument("--zero-based", action="store_true")
    parser.add_argument("--label-decimals", type=int, default=4)
    parser.add_argument("--pair-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--kind", choices=("both", "gs", "tddft"), default="both")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Mapping CSV (defaults to WORKDIR/scan_directory_rename_manifest.csv)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename directories; without this flag the command is a dry run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = args.workdir.resolve()
    bonds = read_bonds(args.bonds_file, args.zero_based)
    renames: list[Rename] = []
    if args.kind in {"both", "gs"}:
        renames.extend(
            discover_kind(
                kind="gs",
                root=workdir / "gs_scan",
                prefix="gs_scan_point_",
                bonds=bonds,
                decimals=args.label_decimals,
            )
        )
    if args.kind in {"both", "tddft"}:
        renames.extend(
            discover_kind(
                kind="tddft",
                root=workdir / "tddft",
                prefix="tddft_step_",
                bonds=bonds,
                decimals=args.label_decimals,
            )
        )
    if not renames:
        print("No legacy pNNN/mNNN scan directories were found.")
        return 0

    validate_plan(renames, args.pair_tolerance)
    renames.sort(key=lambda item: (item.coordinate, item.kind, item.source.name))
    action = "RENAME" if args.apply else "WOULD RENAME"
    for rename in renames:
        print(
            f"{action:12s} {rename.kind:5s} {rename.source.name} -> "
            f"{rename.destination.name}  (Q={rename.coordinate:.12f} A)"
        )

    manifest = (
        args.manifest.resolve()
        if args.manifest
        else workdir / "scan_directory_rename_manifest.csv"
    )
    if args.apply:
        # Record the validated intent before changing names, then mark completion.
        write_manifest(manifest, renames, "planned")
        apply_plan(renames)
        write_manifest(manifest, renames, "renamed")
        print(f"Renamed {len(renames)} directories. Mapping: {manifest}")
    else:
        print(
            f"Dry run only: {len(renames)} directories passed validation. "
            "Re-run with --apply to rename them."
        )
        print(f"When applied, the mapping will be written to {manifest}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

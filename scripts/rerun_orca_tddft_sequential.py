#!/usr/bin/env python3
"""Run ORCA TDDFT single points sequentially along an existing scan.

The geometries are ordered by their *measured* collective coordinate.  The
first point reads the matching ground-state ``job.gbw`` as its SCF guess.  At
every later point, the converged TDDFT ``job.gbw`` from the preceding point is
copied to ``previous.gbw`` and used both as the SCF MORead/CMatrix guess and as
the ``%tddft Restart`` source for the response amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Iterable


@dataclass(frozen=True)
class Bond:
    coefficient: float
    atom_i: int
    atom_j: int
    label: str


@dataclass(frozen=True)
class ScanPoint:
    source_label: str
    coordinate: float
    label: str
    geometry: Path


def read_xyz(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: incomplete XYZ file")
    natoms = int(lines[0].strip())
    if len(lines) < natoms + 2:
        raise ValueError(f"{path}: expected {natoms} atoms")
    atoms: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(lines[2 : 2 + natoms], 3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: malformed XYZ row")
        atoms.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return atoms, coordinates


def read_bonds(path: Path, zero_based: bool) -> list[Bond]:
    bonds: list[Bond] = []
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
        raise ValueError(f"{path}: no collective-coordinate bonds found")
    return bonds


def collective_coordinate(
    coordinates: list[tuple[float, float, float]], bonds: Iterable[Bond]
) -> float:
    value = 0.0
    for bond in bonds:
        try:
            xyz_i, xyz_j = coordinates[bond.atom_i], coordinates[bond.atom_j]
        except IndexError as exc:
            raise ValueError(
                f"Bond {bond.label!r} refers to an atom outside the XYZ geometry"
            ) from exc
        distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(xyz_i, xyz_j)))
        value += bond.coefficient * distance
    return value


def source_rows(jobs_csv: Path | None, source_tddft_dir: Path) -> list[tuple[str, Path]]:
    if jobs_csv is None:
        rows = []
        for directory in sorted(source_tddft_dir.glob("tddft_step_*")):
            if directory.is_dir() and (directory / "geom.xyz").is_file():
                rows.append((directory.name.removeprefix("tddft_step_"), directory / "geom.xyz"))
        if not rows:
            raise ValueError(f"No tddft_step_*/geom.xyz files found in {source_tddft_dir}")
        return rows

    rows = []
    with jobs_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(f"{jobs_csv}: expected a CSV column named 'label'")
        for line_number, row in enumerate(reader, 2):
            label = (row.get("label") or "").strip()
            if not label:
                raise ValueError(f"{jobs_csv}:{line_number}: empty label")
            # Rebase by label so that a manifest copied from another computer does
            # not retain unusable absolute paths.
            rebased = source_tddft_dir / f"tddft_step_{label}" / "geom.xyz"
            original = Path((row.get("geom") or "").strip()).expanduser()
            geometry = rebased if rebased.is_file() else original
            if not geometry.is_file():
                raise FileNotFoundError(
                    f"{jobs_csv}:{line_number}: geometry not found as either "
                    f"{rebased} or {original}"
                )
            rows.append((label, geometry))
    if not rows:
        raise ValueError(f"{jobs_csv}: no scan points found")
    return rows


def build_points(
    rows: list[tuple[str, Path]], bonds: list[Bond], decimals: int
) -> list[ScanPoint]:
    points: list[ScanPoint] = []
    reference_atoms: list[str] | None = None
    labels: dict[str, Path] = {}
    for source_label, geometry in rows:
        atoms, coordinates = read_xyz(geometry)
        if reference_atoms is None:
            reference_atoms = atoms
        elif atoms != reference_atoms:
            raise ValueError(f"{geometry}: atom list/order differs from the first geometry")
        coordinate = collective_coordinate(coordinates, bonds)
        label = f"{coordinate:.{decimals}f}"
        if label in labels:
            raise ValueError(
                f"Coordinate-label collision at {label}: {labels[label]} and {geometry}. "
                "Increase --label-decimals or correct duplicate input points."
            )
        labels[label] = geometry
        points.append(ScanPoint(source_label, coordinate, label, geometry.resolve()))
    return sorted(points, key=lambda point: point.coordinate)


def strip_orca_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def validate_template(text: str) -> None:
    clean = strip_orca_comments(text)
    if "${guess_block}" not in text:
        raise ValueError("Template must contain the ${guess_block} placeholder")
    if "${tddft_restart}" not in text:
        raise ValueError("Template must contain the ${tddft_restart} placeholder")
    for placeholder in ("${charge}", "${mult}", "${xyzfile}"):
        if placeholder not in text:
            raise ValueError(f"Template must contain the {placeholder} placeholder")
    if not re.search(r"(?im)^\s*%tddft\b", clean):
        raise ValueError("Template has no %tddft block")
    if not re.search(r"(?im)^\s*triplets\s+true\b", clean):
        raise ValueError("Restricted-reference template must set 'triplets true'")
    if re.search(r"(?im)^\s*!\s*.*\b(?:UKS|UHF)\b", clean):
        raise ValueError("Restricted-reference template must not request UKS or UHF")
    if re.search(r"(?im)^\s*!\s*.*\b(?:MORead|AutoStart)\b", clean):
        raise ValueError("Put no guess keyword in the template; ${guess_block} controls it")
    if re.search(r"(?im)^\s*%moinp\b|\bGuess\s+MORead\b", clean):
        raise ValueError("Put no MORead directive in the template; ${guess_block} controls it")
    if re.search(r"(?im)^\s*restart\s+[\"']", clean):
        raise ValueError(
            "Put no fixed TDDFT Restart directive in the template; "
            "${tddft_restart} controls it"
        )
    multiplicity = re.search(r"(?im)^\s*irootmult\s+(\S+)", clean)
    if multiplicity and multiplicity.group(1).lower() not in {"3", "triplet"}:
        raise ValueError("IRootMult, when present, must be Triplet (or 3)")


def guess_block() -> str:
    return """%scf
  Guess MORead
  MOInp "previous.gbw"
  GuessMode CMatrix
  AutoStart false
end"""


def tddft_restart(enabled: bool) -> str:
    return '  Restart "previous.gbw"' if enabled else ""


def render_input(
    template_text: str,
    *,
    charge: int,
    geometry: Path,
    restart_tddft: bool,
    point: ScanPoint,
) -> str:
    return Template(template_text).substitute(
        charge=str(charge),
        mult="1",
        xyzfile=str(geometry.resolve()),
        guess_block=guess_block(),
        tddft_restart=tddft_restart(restart_tddft),
        scan_coordinate=f"{point.coordinate:.12f}",
        scan_label=point.label,
    )


def xyzfile_charge_and_multiplicity(input_path: Path) -> tuple[int, int] | None:
    """Read the final ``* xyzfile charge multiplicity ...`` directive."""
    matches = re.findall(
        r"(?im)^\s*\*\s+xyzfile\s+([+-]?\d+)\s+(\d+)\s+\S+",
        strip_orca_comments(input_path.read_text(errors="replace")),
    )
    if not matches:
        return None
    charge, multiplicity = matches[-1]
    return int(charge), int(multiplicity)


def max_geometry_displacement(left: Path, right: Path) -> float:
    left_atoms, left_coordinates = read_xyz(left)
    right_atoms, right_coordinates = read_xyz(right)
    if left_atoms != right_atoms:
        raise ValueError(f"GS seed geometry {left} has a different atom list/order from {right}")
    return max(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(left_xyz, right_xyz)))
        for left_xyz, right_xyz in zip(left_coordinates, right_coordinates)
    )


def find_gs_seed(
    point: ScanPoint,
    source_gs_dir: Path,
    *,
    charge: int,
    geometry_tolerance: float,
) -> Path:
    """Locate and validate the GS GBW corresponding to the first TDDFT point."""
    labels = list(dict.fromkeys((point.source_label, point.label)))
    candidates = [source_gs_dir / f"gs_scan_point_{label}" for label in labels]
    jobdir = next((candidate for candidate in candidates if (candidate / "job.gbw").is_file()), None)
    if jobdir is None:
        searched = ", ".join(str(candidate / "job.gbw") for candidate in candidates)
        raise FileNotFoundError(f"Could not find the first-point GS seed; searched {searched}")

    gbw = (jobdir / "job.gbw").resolve()
    if gbw.stat().st_size == 0:
        raise ValueError(f"Ground-state seed is empty: {gbw}")
    output = jobdir / "job.out"
    if not terminated_normally(output):
        raise ValueError(f"Ground-state seed did not terminate normally: {output}")

    input_path = jobdir / "job.inp"
    if input_path.is_file():
        identity = xyzfile_charge_and_multiplicity(input_path)
        if identity is not None:
            seed_charge, seed_multiplicity = identity
            if seed_charge != charge or seed_multiplicity != 1:
                raise ValueError(
                    f"GS seed {input_path} records charge/multiplicity "
                    f"{seed_charge}/{seed_multiplicity}, but this restricted-singlet "
                    f"TDDFT run requests {charge}/1"
                )

    gs_geometry = next(
        (
            jobdir / name
            for name in ("optimized.xyz", "job.xyz", "optimized_from_output.xyz")
            if (jobdir / name).is_file()
        ),
        None,
    )
    if gs_geometry is not None:
        displacement = max_geometry_displacement(gs_geometry, point.geometry)
        if displacement > geometry_tolerance:
            raise ValueError(
                f"First GS seed geometry differs from the first TDDFT geometry by up to "
                f"{displacement:.3g} A (tolerance {geometry_tolerance:g} A)"
            )
    return gbw


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" in value:
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ORCA executable not found: {resolved}")
        return resolved
    found = shutil.which(value)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {value!r} in PATH; pass --orca /absolute/path/to/orca"
        )
    return Path(found).resolve()


def terminated_normally(output: Path) -> bool:
    return output.is_file() and "ORCA TERMINATED NORMALLY" in output.read_text(errors="replace")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "order",
        "source_label",
        "coordinate_A",
        "label",
        "source_geometry",
        "jobdir",
        "scf_guess_source",
        "tddft_restart_source",
        "status",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source-tddft-dir", required=True, type=Path)
    parser.add_argument(
        "--source-gs-dir",
        required=True,
        type=Path,
        help="GS scan directory containing gs_scan_point_LABEL/job.gbw",
    )
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        help="Optional allow-list/order-independent manifest; its label column selects points",
    )
    parser.add_argument("--bonds-file", required=True, type=Path)
    parser.add_argument("--zero-based", action="store_true")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--charge", required=True, type=int)
    parser.add_argument("--orca", default="orca")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label-decimals", type=int, default=4)
    parser.add_argument(
        "--gs-geometry-tolerance",
        type=float,
        default=1.0e-5,
        help="Maximum Cartesian mismatch between first GS and TDDFT geometries, in Angstrom",
    )
    parser.add_argument("--timeout", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only jobs with normal termination and a nonempty job.gbw",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the ordered plan without creating or running jobs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template_text = args.template.read_text(errors="replace")
    validate_template(template_text)
    bonds = read_bonds(args.bonds_file, args.zero_based)
    rows = source_rows(args.jobs_csv, args.source_tddft_dir)
    points = build_points(rows, bonds, args.label_decimals)
    gs_seed = find_gs_seed(
        points[0],
        args.source_gs_dir,
        charge=args.charge,
        geometry_tolerance=args.gs_geometry_tolerance,
    )

    print(f"Validated restricted singlet-reference template: {args.template}")
    print(f"Selected {len(points)} unique geometries; execution order is increasing Q.")
    print(f"First-point SCF guess: {gs_seed}")
    print("Later points: preceding TDDFT GBW supplies both SCF orbitals and TDDFT amplitudes.")
    for index, point in enumerate(points):
        print(
            f"{index:4d}  Q={point.coordinate:.10f} A  "
            f"{point.source_label!r} -> tddft_step_{point.label}"
        )
    if args.dry_run:
        print("Dry run complete; no directories were created and ORCA was not launched.")
        return 0

    orca = resolve_executable(args.orca)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sequential_tddft_manifest.csv"
    manifest_rows: list[dict[str, str]] = []
    preceding_gbw = gs_seed

    for index, point in enumerate(points):
        restart_tddft = index > 0
        jobdir = args.output_dir / f"tddft_step_{point.label}"
        output = jobdir / "job.out"
        gbw = jobdir / "job.gbw"
        if jobdir.exists() and any(jobdir.iterdir()) and not args.resume:
            raise FileExistsError(
                f"{jobdir} is nonempty. Use a fresh --output-dir or explicitly use --resume."
            )
        jobdir.mkdir(parents=True, exist_ok=True)

        if args.resume and terminated_normally(output) and gbw.is_file() and gbw.stat().st_size:
            status = "reused-normal"
            print(f"[{index + 1}/{len(points)}] Reusing {jobdir}")
        else:
            local_geometry = jobdir / "geom.xyz"
            shutil.copy2(point.geometry, local_geometry)
            local_guess = jobdir / "previous.gbw"
            shutil.copy2(preceding_gbw, local_guess)
            rendered = render_input(
                template_text,
                charge=args.charge,
                geometry=local_geometry,
                restart_tddft=restart_tddft,
                point=point,
            )
            (jobdir / "job.inp").write_text(rendered)
            print(
                f"[{index + 1}/{len(points)}] Running Q={point.coordinate:.10f} A; "
                f"SCF guess={preceding_gbw}; "
                f"TDDFT restart={preceding_gbw if restart_tddft else 'none (GS seed)'}"
            )
            with output.open("w") as handle:
                result = subprocess.run(
                    [str(orca), "job.inp"],
                    cwd=jobdir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ORCA returned {result.returncode} at Q={point.coordinate:.10f}; "
                    f"see {output}"
                )
            if not terminated_normally(output):
                raise RuntimeError(
                    f"ORCA did not terminate normally at Q={point.coordinate:.10f}; "
                    f"the GBW will not be propagated. See {output}"
                )
            if not gbw.is_file() or gbw.stat().st_size == 0:
                raise RuntimeError(f"Normal ORCA job did not produce a nonempty {gbw}")
            status = "completed-normal"

        manifest_rows.append(
            {
                "order": str(index),
                "source_label": point.source_label,
                "coordinate_A": f"{point.coordinate:.12f}",
                "label": point.label,
                "source_geometry": str(point.geometry),
                "jobdir": str(jobdir.resolve()),
                "scf_guess_source": str(preceding_gbw),
                "tddft_restart_source": str(preceding_gbw) if restart_tddft else "",
                "status": status,
            }
        )
        write_manifest(manifest_path, manifest_rows)
        preceding_gbw = gbw.resolve()

    print(f"Completed {len(points)} sequential TDDFT jobs.")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Continue a relaxed ORCA GS scan and sequential RKS-triplet TDDFT chain.

For every new scan coordinate this driver:

1. projects the preceding converged GS geometry onto the new constraint;
2. runs a constrained GS optimization using the preceding GS GBW;
3. runs RKS-reference triplet TDDFT on the optimized geometry using the
   preceding TDDFT GBW for both the SCF guess and response-amplitude restart;
4. propagates outputs only after strict normal-termination and artifact checks.

The GS and TDDFT restart chains remain separate.  Resume decisions include
SHA-256 provenance for geometry, templates, and upstream GBW files, so changing
or recomputing an upstream point invalidates incompatible downstream jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from string import Template
from typing import Iterable


ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
)
XYZFILE_RE = re.compile(
    r"(?im)^\s*\*\s+xyzfile\s+([+-]?\d+)\s+(\d+)\s+\S+"
)
NROOTS_RE = re.compile(r"(?im)^\s*nroots\s+(\d+)\b")
STATE_RE = re.compile(
    r"^\s*STATE\s+\d+:\s+E=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+au"
)


@dataclass(frozen=True)
class Bond:
    coefficient: float
    atom_i: int
    atom_j: int
    label: str


@dataclass(frozen=True)
class GeometryData:
    atoms: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Seed:
    jobdir: Path
    gbw: Path
    geometry_path: Path
    geometry: GeometryData
    coordinate: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed-gs-job", required=True, type=Path)
    parser.add_argument("--seed-tddft-job", required=True, type=Path)
    parser.add_argument("--bonds-file", required=True, type=Path)
    parser.add_argument("--zero-based", action="store_true")
    parser.add_argument("--gs-template", required=True, type=Path)
    parser.add_argument("--tddft-template", required=True, type=Path)
    parser.add_argument("--gs-charge", required=True, type=int)
    parser.add_argument("--gs-mult", type=int, default=1)
    parser.add_argument("--tddft-charge", required=True, type=int)
    parser.add_argument("--stop-coordinate", required=True, type=float)
    parser.add_argument("--coordinate-step", type=float, default=0.01)
    parser.add_argument("--label-decimals", type=int, default=4)
    parser.add_argument("--orca", default="orca")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--geom-maxiter", type=int)
    parser.add_argument("--gs-timeout", type=int)
    parser.add_argument("--tddft-timeout", type=int)
    parser.add_argument("--constraint-tolerance", type=float, default=5.0e-5)
    parser.add_argument("--seed-geometry-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only jobs whose outputs and SHA-256 provenance still validate",
    )
    parser.add_argument(
        "--skip-tddft",
        action="store_true",
        help="Extend only the constrained GS scan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the continuation plan without writing files",
    )
    return parser.parse_args(argv)


def strip_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def read_xyz(path: Path) -> GeometryData:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: incomplete XYZ file")
    atom_count = int(lines[0].strip())
    if len(lines) < atom_count + 2:
        raise ValueError(f"{path}: expected {atom_count} atoms")
    atoms: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(lines[2 : 2 + atom_count], 3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{path}:{line_number}: malformed XYZ row")
        atoms.append(fields[0])
        coordinates.append(tuple(float(value) for value in fields[1:4]))
    return GeometryData(tuple(atoms), tuple(coordinates))


def xyz_text(geometry: GeometryData, comment: str) -> str:
    lines = [str(len(geometry.atoms)), comment]
    for atom, (x, y, z) in zip(geometry.atoms, geometry.coordinates):
        lines.append(f"{atom:<3s} {x:18.10f} {y:18.10f} {z:18.10f}")
    return "\n".join(lines) + "\n"


def write_xyz(path: Path, geometry: GeometryData, comment: str) -> None:
    path.write_text(xyz_text(geometry, comment))


def read_last_xyz_frame(path: Path) -> GeometryData | None:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    frames: list[GeometryData] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        try:
            atom_count = int(lines[index].strip())
        except ValueError:
            break
        end = index + atom_count + 2
        if end > len(lines):
            break
        frame_lines = lines[index:end]
        try:
            atoms = []
            coordinates = []
            for row in frame_lines[2:]:
                fields = row.split()
                atoms.append(fields[0])
                coordinates.append(tuple(float(value) for value in fields[1:4]))
            if len(atoms) == atom_count:
                frames.append(GeometryData(tuple(atoms), tuple(coordinates)))
        except (IndexError, ValueError):
            pass
        index = end
    return frames[-1] if frames else None


def optimized_geometry(jobdir: Path) -> tuple[Path, GeometryData]:
    for name in ("optimized.xyz", "job.xyz", "job_trj.xyz"):
        path = jobdir / name
        geometry = read_last_xyz_frame(path) if path.is_file() else None
        if geometry is not None:
            return path.resolve(), geometry
    raise FileNotFoundError(
        f"No readable optimized.xyz, job.xyz, or job_trj.xyz in {jobdir}"
    )


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
        raise ValueError(f"{path}: no scan bonds found")
    return bonds


def validate_bonds(bonds: Iterable[Bond], atom_count: int) -> None:
    for bond in bonds:
        if (
            bond.atom_i == bond.atom_j
            or bond.atom_i < 0
            or bond.atom_j < 0
            or bond.atom_i >= atom_count
            or bond.atom_j >= atom_count
        ):
            raise ValueError(
                f"Bond {bond.label!r} has invalid atoms "
                f"{bond.atom_i + 1}-{bond.atom_j + 1}"
            )


def distance(left, right) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def bond_distances(geometry: GeometryData, bonds: Iterable[Bond]) -> list[float]:
    return [
        distance(
            geometry.coordinates[bond.atom_i],
            geometry.coordinates[bond.atom_j],
        )
        for bond in bonds
    ]


def collective_coordinate(geometry: GeometryData, bonds: Iterable[Bond]) -> float:
    return sum(
        bond.coefficient * value
        for bond, value in zip(bonds, bond_distances(geometry, bonds))
    )


def target_distances(
    geometry: GeometryData, bonds: list[Bond], target_coordinate: float
) -> list[float]:
    current_distances = bond_distances(geometry, bonds)
    current_coordinate = sum(
        bond.coefficient * value
        for bond, value in zip(bonds, current_distances)
    )
    coefficient_norm = sum(bond.coefficient**2 for bond in bonds)
    if coefficient_norm <= 0.0:
        raise ValueError("The collective-coordinate coefficient norm is zero")
    scale = (target_coordinate - current_coordinate) / coefficient_norm
    targets = [
        value + scale * bond.coefficient
        for bond, value in zip(bonds, current_distances)
    ]
    if any(value <= 0.05 for value in targets):
        raise ValueError(f"Nonphysical target bond distances: {targets}")
    return targets


def project_to_targets(
    geometry: GeometryData,
    bonds: list[Bond],
    targets: list[float],
    *,
    tolerance: float = 1.0e-9,
    max_cycles: int = 500,
    damping: float = 0.5,
) -> GeometryData:
    coordinates = [list(row) for row in geometry.coordinates]
    for _ in range(max_cycles):
        max_error = 0.0
        for bond, target in zip(bonds, targets):
            left = coordinates[bond.atom_i]
            right = coordinates[bond.atom_j]
            vector = [a - b for a, b in zip(left, right)]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm < 1.0e-12:
                raise ValueError(f"Zero-length bond for {bond.label}")
            error = norm - target
            max_error = max(max_error, abs(error))
            correction = [damping * error * value / norm for value in vector]
            for axis in range(3):
                left[axis] -= 0.5 * correction[axis]
                right[axis] += 0.5 * correction[axis]
        if max_error < tolerance:
            break
    else:
        raise RuntimeError("Projection onto scan constraints did not converge")
    return GeometryData(
        geometry.atoms,
        tuple(tuple(float(value) for value in row) for row in coordinates),
    )


def max_displacement(left: GeometryData, right: GeometryData) -> float:
    if left.atoms != right.atoms:
        raise ValueError("Geometry atom identities/order differ")
    return max(
        distance(a, b) for a, b in zip(left.coordinates, right.coordinates)
    )


def terminated_normally(path: Path) -> bool:
    return path.is_file() and "ORCA TERMINATED NORMALLY" in path.read_text(
        errors="replace"
    )


def optimization_converged(path: Path) -> bool:
    return path.is_file() and bool(
        re.search(
            r"THE\s+OPTIMIZATION\s+HAS\s+CONVERGED",
            path.read_text(errors="replace"),
            re.IGNORECASE,
        )
    )


def parse_energy(path: Path) -> float | None:
    matches = ENERGY_RE.findall(path.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def parse_xyzfile_identity(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    matches = XYZFILE_RE.findall(strip_comments(path.read_text(errors="replace")))
    if not matches:
        return None
    charge, multiplicity = matches[-1]
    return int(charge), int(multiplicity)


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def validate_seed_gs(
    jobdir: Path, bonds: list[Bond], charge: int, multiplicity: int
) -> Seed:
    jobdir = jobdir.resolve()
    output = jobdir / "job.out"
    gbw = jobdir / "job.gbw"
    if not terminated_normally(output) or not optimization_converged(output):
        raise ValueError(f"GS seed is not a normally converged optimization: {output}")
    if not nonempty(gbw):
        raise ValueError(f"GS seed has no nonempty GBW: {gbw}")
    identity = parse_xyzfile_identity(jobdir / "job.inp")
    if identity is not None and identity != (charge, multiplicity):
        raise ValueError(
            f"GS seed charge/multiplicity is {identity}, expected "
            f"{(charge, multiplicity)}"
        )
    geometry_path, geometry = optimized_geometry(jobdir)
    validate_bonds(bonds, len(geometry.atoms))
    return Seed(
        jobdir,
        gbw.resolve(),
        geometry_path,
        geometry,
        collective_coordinate(geometry, bonds),
    )


def triplet_root_count(output: Path) -> int:
    in_triplets = False
    count = 0
    for line in output.read_text(errors="replace").splitlines():
        if "TD-DFT/TDA EXCITED STATES (TRIPLETS)" in line:
            in_triplets = True
            continue
        if in_triplets and (
            "ABSORPTION SPECTRUM" in line
            or "TD-DFT/TDA EXCITED STATES (SINGLETS)" in line
        ):
            in_triplets = False
        if in_triplets and STATE_RE.match(line):
            count += 1
    return count


def validate_seed_tddft(
    jobdir: Path,
    bonds: list[Bond],
    charge: int,
    expected_roots: int,
) -> Seed:
    jobdir = jobdir.resolve()
    output = jobdir / "job.out"
    gbw = jobdir / "job.gbw"
    cis = jobdir / "job.cis"
    geometry_path = jobdir / "geom.xyz"
    if not terminated_normally(output):
        raise ValueError(f"TDDFT seed did not terminate normally: {output}")
    if not nonempty(gbw) or not nonempty(cis):
        raise ValueError(f"TDDFT seed lacks a nonempty GBW or CIS file: {jobdir}")
    if triplet_root_count(output) != expected_roots:
        raise ValueError(
            f"TDDFT seed contains {triplet_root_count(output)} parsed triplet roots; "
            f"expected {expected_roots}"
        )
    identity = parse_xyzfile_identity(jobdir / "job.inp")
    if identity is not None and identity != (charge, 1):
        raise ValueError(
            f"TDDFT seed charge/multiplicity is {identity}, expected {(charge, 1)}"
        )
    geometry = read_xyz(geometry_path)
    validate_bonds(bonds, len(geometry.atoms))
    return Seed(
        jobdir,
        gbw.resolve(),
        geometry_path.resolve(),
        geometry,
        collective_coordinate(geometry, bonds),
    )


def validate_gs_template(text: str) -> None:
    for placeholder in (
        "${guess_block}",
        "${constraints_block}",
        "${charge}",
        "${mult}",
        "${xyzfile}",
    ):
        if placeholder not in text:
            raise ValueError(f"GS template must contain {placeholder}")
    clean = strip_comments(text)
    if not re.search(r"(?im)^\s*!\s*.*\bOpt\b", clean):
        raise ValueError("GS template must request an ORCA optimization")
    if re.search(r"(?im)^\s*%moinp\b|\bGuess\s+MORead\b", clean):
        raise ValueError(
            "GS template must not contain a fixed orbital guess; use ${guess_block}"
        )


def validate_tddft_template(text: str) -> int:
    for placeholder in (
        "${guess_block}",
        "${tddft_restart}",
        "${charge}",
        "${mult}",
        "${xyzfile}",
    ):
        if placeholder not in text:
            raise ValueError(f"TDDFT template must contain {placeholder}")
    clean = strip_comments(text)
    if not re.search(r"(?im)^\s*!\s*.*\bRKS\b", clean):
        raise ValueError("TDDFT template must explicitly request RKS")
    if re.search(r"(?im)^\s*!\s*.*\b(?:UKS|UHF)\b", clean):
        raise ValueError("TDDFT template must not request UKS or UHF")
    if not re.search(r"(?im)^\s*triplets\s+true\b", clean):
        raise ValueError("TDDFT template must set 'triplets true'")
    if re.search(r"(?im)^\s*restart\s+[\"']", clean):
        raise ValueError(
            "TDDFT template must not contain a fixed restart; "
            "use ${tddft_restart}"
        )
    nroots = NROOTS_RE.search(clean)
    if nroots is None:
        raise ValueError("TDDFT template must define nroots")
    return int(nroots.group(1))


def guess_block() -> str:
    return """%scf
  Guess MORead
  MOInp "previous.gbw"
  GuessMode CMatrix
  AutoStart false
end"""


def constraints_block(bonds: list[Bond], maxiter: int | None) -> str:
    lines = ["%geom"]
    if maxiter is not None:
        lines.append(f"  MaxIter {int(maxiter)}")
    lines.append("  Constraints")
    for bond in bonds:
        lines.append(
            f"    {{ B {bond.atom_i} {bond.atom_j} C }}  # {bond.label}"
        )
    lines.extend(("  end", "end"))
    return "\n".join(lines)


def render_gs_input(
    template_text: str,
    geometry: Path,
    charge: int,
    multiplicity: int,
    bonds: list[Bond],
    maxiter: int | None,
    target: float,
) -> str:
    return Template(template_text).substitute(
        guess_block=guess_block(),
        constraints_block=constraints_block(bonds, maxiter),
        charge=str(charge),
        mult=str(multiplicity),
        xyzfile=str(geometry.resolve()),
        scan_coordinate=f"{target:.12f}",
        scan_label=f"{target:.4f}",
    )


def render_tddft_input(
    template_text: str,
    geometry: Path,
    charge: int,
    target: float,
) -> str:
    return Template(template_text).substitute(
        guess_block=guess_block(),
        tddft_restart='  Restart "previous.gbw"',
        charge=str(charge),
        mult="1",
        xyzfile=str(geometry.resolve()),
        scan_coordinate=f"{target:.12f}",
        scan_label=f"{target:.4f}",
    )


def generate_targets(seed: float, stop: float, spacing: float) -> list[float]:
    if spacing <= 0.0:
        raise ValueError("--coordinate-step must be positive")
    if stop <= seed:
        raise ValueError(
            f"--stop-coordinate ({stop}) must exceed seed coordinate ({seed})"
        )
    current = Decimal(str(seed))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(spacing))
    targets: list[Decimal] = []
    candidate = current + step_decimal
    while candidate < stop_decimal:
        targets.append(candidate)
        candidate += step_decimal
    if not targets or targets[-1] != stop_decimal:
        targets.append(stop_decimal)
    return [float(value) for value in targets]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def provenance_matches(path: Path, expected: dict) -> bool:
    try:
        return json.loads(path.read_text()) == expected
    except (OSError, ValueError):
        return False


def write_json_atomic(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def archive_stale(jobdir: Path) -> None:
    if not jobdir.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = jobdir.with_name(f"{jobdir.name}.stale-{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = jobdir.with_name(
            f"{jobdir.name}.stale-{timestamp}-{counter:02d}"
        )
        counter += 1
    jobdir.rename(candidate)
    print(f"Archived incompatible/incomplete job as {candidate}")


def run_orca(
    orca: Path, jobdir: Path, timeout: int | None
) -> subprocess.CompletedProcess:
    output = jobdir / "job.out"
    with output.open("w") as handle:
        return subprocess.run(
            [str(orca), "job.inp"],
            cwd=jobdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )


def valid_gs_job(
    jobdir: Path,
    expected_provenance: dict,
    bonds: list[Bond],
    target: float,
    tolerance: float,
) -> tuple[Path, GeometryData] | None:
    if not provenance_matches(jobdir / "provenance.json", expected_provenance):
        return None
    if (
        not terminated_normally(jobdir / "job.out")
        or not optimization_converged(jobdir / "job.out")
        or not nonempty(jobdir / "job.gbw")
    ):
        return None
    try:
        geometry_path, geometry = optimized_geometry(jobdir)
    except (OSError, ValueError):
        return None
    if abs(collective_coordinate(geometry, bonds) - target) > tolerance:
        return None
    return geometry_path, geometry


def valid_tddft_job(
    jobdir: Path, expected_provenance: dict, expected_roots: int
) -> bool:
    return (
        provenance_matches(jobdir / "provenance.json", expected_provenance)
        and terminated_normally(jobdir / "job.out")
        and nonempty(jobdir / "job.gbw")
        and nonempty(jobdir / "job.cis")
        and triplet_root_count(jobdir / "job.out") == expected_roots
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "order",
        "target_coordinate_A",
        "label",
        "gs_status",
        "gs_final_coordinate_A",
        "gs_energy_Eh",
        "gs_jobdir",
        "gs_seed_gbw_source",
        "gs_seed_gbw_sha256",
        "tddft_status",
        "tddft_jobdir",
        "tddft_seed_gbw_source",
        "tddft_seed_gbw_sha256",
        "triplet_roots",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bonds = read_bonds(args.bonds_file, args.zero_based)
    gs_template_text = args.gs_template.read_text(errors="replace")
    td_template_text = args.tddft_template.read_text(errors="replace")
    validate_gs_template(gs_template_text)
    expected_roots = validate_tddft_template(td_template_text)

    gs_seed = validate_seed_gs(
        args.seed_gs_job, bonds, args.gs_charge, args.gs_mult
    )
    td_seed = validate_seed_tddft(
        args.seed_tddft_job, bonds, args.tddft_charge, expected_roots
    )
    if gs_seed.geometry.atoms != td_seed.geometry.atoms:
        raise ValueError("GS and TDDFT seed atom identities/order differ")
    seed_displacement = max_displacement(gs_seed.geometry, td_seed.geometry)
    if seed_displacement > args.seed_geometry_tolerance:
        raise ValueError(
            f"GS and TDDFT seed geometries differ by {seed_displacement:.3g} A; "
            f"tolerance is {args.seed_geometry_tolerance:g} A"
        )

    targets = generate_targets(
        gs_seed.coordinate, args.stop_coordinate, args.coordinate_step
    )
    labels = [f"{target:.{args.label_decimals}f}" for target in targets]
    if len(labels) != len(set(labels)):
        raise ValueError(
            "Rounded scan labels collide; increase --label-decimals or "
            "increase --coordinate-step"
        )

    print(f"GS seed: {gs_seed.jobdir}")
    print(f"TDDFT seed: {td_seed.jobdir}")
    print(f"Validated common seed coordinate: {gs_seed.coordinate:.12f} A")
    print(
        f"Planned {len(targets)} new points through "
        f"{args.stop_coordinate:.12f} A:"
    )
    for index, (target, label) in enumerate(zip(targets, labels), 1):
        print(f"{index:4d}  Q={target:.12f} A  label={label}")
    if args.dry_run:
        print("Dry run complete; no directories or ORCA jobs were created.")
        return 0

    orca = resolve_executable(args.orca)
    workdir = args.workdir.resolve()
    if workdir.exists() and any(workdir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{workdir} is nonempty. Use a fresh --workdir or --resume."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    gs_root = workdir / "gs_scan"
    td_root = workdir / "tddft_rks_triplets_sequential"
    gs_root.mkdir(exist_ok=True)
    if not args.skip_tddft:
        td_root.mkdir(exist_ok=True)
    shutil.copy2(args.bonds_file, workdir / "input_bonds.dat")
    shutil.copy2(args.gs_template, workdir / "input_gs_template.inp")
    shutil.copy2(args.tddft_template, workdir / "input_tddft_template.inp")
    write_json_atomic(
        workdir / "continuation_plan.json",
        {
            "seed_gs_job": str(gs_seed.jobdir),
            "seed_tddft_job": str(td_seed.jobdir),
            "seed_coordinate_A": gs_seed.coordinate,
            "stop_coordinate_A": args.stop_coordinate,
            "coordinate_step_A": args.coordinate_step,
            "targets_A": targets,
            "orca": str(orca),
            "expected_triplet_roots": expected_roots,
        },
    )

    previous_gs_geometry = gs_seed.geometry
    previous_gs_gbw = gs_seed.gbw
    previous_td_gbw = td_seed.gbw
    manifest_rows: list[dict[str, str]] = []
    manifest_path = workdir / "continuation_manifest.csv"

    for index, (target, label) in enumerate(zip(targets, labels)):
        print(f"\n[{index + 1}/{len(targets)}] GS Q={target:.12f} A")
        targets_for_bonds = target_distances(
            previous_gs_geometry, bonds, target
        )
        projected = project_to_targets(
            previous_gs_geometry, bonds, targets_for_bonds
        )
        gs_jobdir = gs_root / f"gs_scan_point_{label}"
        projected_text = xyz_text(
            projected, f"Projected GS start; target Q={target:.12f} A"
        )
        gs_seed_source = str(previous_gs_gbw)
        gs_seed_hash = sha256_file(previous_gs_gbw)
        gs_provenance = {
            "stage": "gs",
            "target_coordinate_A": target,
            "geometry_sha256": sha256_text(projected_text),
            "seed_gbw_sha256": gs_seed_hash,
            "template_sha256": sha256_text(gs_template_text),
            "bonds_sha256": sha256_file(args.bonds_file),
            "charge": args.gs_charge,
            "multiplicity": args.gs_mult,
        }
        reused_gs = (
            valid_gs_job(
                gs_jobdir,
                gs_provenance,
                bonds,
                target,
                args.constraint_tolerance,
            )
            if args.resume
            else None
        )
        if reused_gs is not None:
            gs_geometry_path, gs_geometry = reused_gs
            gs_status = "reused-validated"
            print(f"Reusing validated GS job {gs_jobdir}")
        else:
            if gs_jobdir.exists() and any(gs_jobdir.iterdir()):
                archive_stale(gs_jobdir)
            gs_jobdir.mkdir(parents=True, exist_ok=True)
            start_geometry = gs_jobdir / "start_constrained.xyz"
            start_geometry.write_text(projected_text)
            shutil.copy2(previous_gs_gbw, gs_jobdir / "previous.gbw")
            (gs_jobdir / "job.inp").write_text(
                render_gs_input(
                    gs_template_text,
                    start_geometry,
                    args.gs_charge,
                    args.gs_mult,
                    bonds,
                    args.geom_maxiter,
                    target,
                )
            )
            write_json_atomic(gs_jobdir / "provenance.json", gs_provenance)
            result = run_orca(orca, gs_jobdir, args.gs_timeout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ORCA GS returned {result.returncode}; see "
                    f"{gs_jobdir / 'job.out'}"
                )
            if (
                not terminated_normally(gs_jobdir / "job.out")
                or not optimization_converged(gs_jobdir / "job.out")
                or not nonempty(gs_jobdir / "job.gbw")
            ):
                raise RuntimeError(
                    f"GS point did not converge normally or lacks job.gbw: "
                    f"{gs_jobdir}"
                )
            _, gs_geometry = optimized_geometry(gs_jobdir)
            final_coordinate = collective_coordinate(gs_geometry, bonds)
            if abs(final_coordinate - target) > args.constraint_tolerance:
                raise RuntimeError(
                    f"GS final coordinate {final_coordinate:.12f} differs from "
                    f"target {target:.12f} by more than "
                    f"{args.constraint_tolerance:g} A"
                )
            gs_geometry_path = gs_jobdir / "optimized.xyz"
            write_xyz(
                gs_geometry_path,
                gs_geometry,
                f"Converged GS; target Q={target:.12f} A; "
                f"final Q={final_coordinate:.12f} A",
            )
            gs_status = "completed-normal"

        final_coordinate = collective_coordinate(gs_geometry, bonds)
        gs_energy = parse_energy(gs_jobdir / "job.out")
        previous_gs_geometry = gs_geometry
        previous_gs_gbw = (gs_jobdir / "job.gbw").resolve()

        td_status = "skipped"
        td_jobdir = td_root / f"tddft_step_{label}"
        td_seed_source = str(previous_td_gbw)
        td_seed_hash = sha256_file(previous_td_gbw)
        if not args.skip_tddft:
            print(f"[{index + 1}/{len(targets)}] TDDFT Q={target:.12f} A")
            td_geometry_text = xyz_text(
                gs_geometry, f"TDDFT on converged GS; Q={final_coordinate:.12f} A"
            )
            td_provenance = {
                "stage": "tddft",
                "target_coordinate_A": target,
                "geometry_sha256": sha256_text(td_geometry_text),
                "seed_gbw_sha256": td_seed_hash,
                "template_sha256": sha256_text(td_template_text),
                "charge": args.tddft_charge,
                "multiplicity": 1,
                "expected_triplet_roots": expected_roots,
            }
            reuse_td = args.resume and valid_tddft_job(
                td_jobdir, td_provenance, expected_roots
            )
            if reuse_td:
                td_status = "reused-validated"
                print(f"Reusing validated TDDFT job {td_jobdir}")
            else:
                if td_jobdir.exists() and any(td_jobdir.iterdir()):
                    archive_stale(td_jobdir)
                td_jobdir.mkdir(parents=True, exist_ok=True)
                td_geometry = td_jobdir / "geom.xyz"
                td_geometry.write_text(td_geometry_text)
                shutil.copy2(previous_td_gbw, td_jobdir / "previous.gbw")
                (td_jobdir / "job.inp").write_text(
                    render_tddft_input(
                        td_template_text,
                        td_geometry,
                        args.tddft_charge,
                        target,
                    )
                )
                write_json_atomic(
                    td_jobdir / "provenance.json", td_provenance
                )
                result = run_orca(orca, td_jobdir, args.tddft_timeout)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ORCA TDDFT returned {result.returncode}; see "
                        f"{td_jobdir / 'job.out'}"
                    )
                if not valid_tddft_job(
                    td_jobdir, td_provenance, expected_roots
                ):
                    raise RuntimeError(
                        f"TDDFT point failed validation; propagation stopped at "
                        f"{td_jobdir}"
                    )
                td_status = "completed-normal"
            previous_td_gbw = (td_jobdir / "job.gbw").resolve()

        manifest_rows.append(
            {
                "order": str(index),
                "target_coordinate_A": f"{target:.12f}",
                "label": label,
                "gs_status": gs_status,
                "gs_final_coordinate_A": f"{final_coordinate:.12f}",
                "gs_energy_Eh": "" if gs_energy is None else f"{gs_energy:.12f}",
                "gs_jobdir": str(gs_jobdir.resolve()),
                "gs_seed_gbw_source": gs_seed_source,
                "gs_seed_gbw_sha256": gs_seed_hash,
                "tddft_status": td_status,
                "tddft_jobdir": (
                    "" if args.skip_tddft else str(td_jobdir.resolve())
                ),
                "tddft_seed_gbw_source": (
                    "" if args.skip_tddft else td_seed_source
                ),
                "tddft_seed_gbw_sha256": (
                    "" if args.skip_tddft else td_seed_hash
                ),
                "triplet_roots": (
                    "" if args.skip_tddft else str(expected_roots)
                ),
            }
        )
        write_manifest(manifest_path, manifest_rows)

    print(f"\nCompleted {len(targets)} continuation points.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

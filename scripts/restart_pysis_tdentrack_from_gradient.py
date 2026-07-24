#!/usr/bin/env python3
"""Restart the Ru-CO TDenTrack optimization from a retained ORCA gradient.

This deliberately starts a fresh RFOptimizer instead of restoring a pysisyphus
``restart_*.yaml`` file.  It retains electronic continuity by auditing the
accepted all-root survey that produced the seed gradient, then bootstraps a new
transactional TDenTrack session from the exact BSON/CIS/EnGrad artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from pysisyphus.Geometry import Geometry
from pysisyphus.calculators import ORCA
from pysisyphus.calculators.ORCA6StateSurvey import (
    GLOBAL_STATE_ROOTS,
    bootstrap_tdentrack_snapshot,
)
from pysisyphus.calculators.TDenTrackORCA import TDenTrackORCA
from pysisyphus.intcoords.PrimTypes import PrimTypes
from pysisyphus.optimizers.RFOptimizer import RFOptimizer


NROOTS = 15
ROOTS = tuple(range(1, NROOTS + 1))
PAL = 14
MEM_PER_CORE_MB = 2000
VALIDATION_TOLERANCE = 1.0e-6
# The N(5)-Ru(0)-N(37) angle is 174.96 degrees at the retained seed.
# Represent it by the two nonsingular linear-bend components instead of one
# ordinary bend that becomes invalid as soon as it crosses 175 degrees.
DEFAULT_LINEARIZED_BENDS = ((5, 0, 37),)

KEYWORDS = (
    "libxc(wb97x-d3bj) UKS def2-SVP def2/J RIJCOSX "
    "CPCM(Acetonitrile) VeryTightSCF KeepDens UNO DEFGRID3"
)

BLOCKS = f"""%basis
  NewGTO Ru "def2-TZVP" end
end

%tddft
  nroots {NROOTS}
  tda true
  CPCMEQ true
  UPop true
  dotrans all
end

%output
  Print[P_SpinDensity] 1
  Print[P_UNO_OccNum] 1
  Print[P_UNO_AtPopMO_L] 1
  Print[P_UNO_ReducedOrbPopMO_L] 1
end"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a fresh transactional RF optimization from a completed "
            "ORCA root gradient and its accepted TDenTrack survey."
        )
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(
            os.environ.get(
                "TDENTRACK_WORKDIR",
                "/scratch/wb97xdbj_def2tzvpp_def2svp/pysis_test_optimizations",
            )
        ),
    )
    parser.add_argument("--seed-dir", default="opt_job_retry7")
    parser.add_argument(
        "--seed-stem", default="calculator_000.006.orca"
    )
    parser.add_argument("--run-dir", default="opt_job_retry8")
    parser.add_argument(
        "--manifest-glob",
        default="tdentrack_surveys/survey-r0006-*/state_survey.json",
    )
    parser.add_argument("--previous-root", type=int, default=2)
    parser.add_argument("--initial-root", type=int, default=2)
    parser.add_argument("--accepted-factor", type=float, default=0.5)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument(
        "--coord-type",
        choices=("redund", "tric", "cart"),
        default="redund",
        help=(
            "Nuclear optimization coordinates. Standard redundant internals are "
            "the restart default: unlike TRIC they contain no fragment rotation "
            "primitives, whose implicit rebuild cannot be committed safely by "
            "the transactional state-following controller."
        ),
    )
    parser.add_argument(
        "--linearize-bend",
        nargs=3,
        type=int,
        action="append",
        metavar=("I", "J", "K"),
        help=(
            "For redundant coordinates, replace ordinary bend I-J-K by its "
            "linear-bend pair. May be repeated. The Ru-CO restart defaults to "
            "the near-linear N(5)-Ru(0)-N(37) bend."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the retained seed without creating the new run directory.",
    )
    return parser.parse_args()


def normalized_atoms(atoms):
    return tuple(str(atom).strip().lower() for atom in atoms)


def exact_engrad(path: Path) -> dict[str, object]:
    """Read one explicitly named ORCA EnGrad file.

    ORCA.parse_engrad(directory) is intentionally not used because a restart
    seed directory can contain several earlier ``*.engrad`` files.
    """

    values = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(stripped.replace("D", "E").replace("d", "e"))
    if len(values) < 2:
        raise RuntimeError(f"Truncated EnGrad file: {path}")
    atom_count = int(values[0])
    energy = float(values[1])
    gradients = np.asarray(
        [float(value) for value in values[2 : 2 + 3 * atom_count]],
        dtype=float,
    )
    if gradients.size != 3 * atom_count or not np.all(np.isfinite(gradients)):
        raise RuntimeError(f"Invalid gradient vector in {path}")
    return {"energy": energy, "forces": -gradients}


def build_geometry(
    atoms, coordinates: np.ndarray, coord_type: str, linearize_bends
) -> Geometry:
    if coord_type != "redund":
        return Geometry(atoms, coordinates, coord_type=coord_type)

    probe = Geometry(atoms, coordinates, coord_type="redund")
    typed_prims = list(probe.internal.typed_prims)
    for requested in linearize_bends:
        i, j, k = map(int, requested)
        matches = [
            primitive
            for primitive in typed_prims
            if (
                primitive[0] == PrimTypes.BEND
                and primitive[2] == j
                and {primitive[1], primitive[3]} == {i, k}
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one ordinary bend matching {i}-{j}-{k}; "
                f"found {matches}."
            )
        ordinary = matches[0]
        typed_prims.remove(ordinary)
        indices = tuple(map(int, ordinary[1:]))
        typed_prims.extend(
            (
                (PrimTypes.LINEAR_BEND, *indices),
                (PrimTypes.LINEAR_BEND_COMPLEMENT, *indices),
            )
        )
        print(
            "Replaced near-linear redundant primitive "
            f"Bend({indices}) by a linear-bend/complement pair."
        )

    return Geometry(
        atoms,
        coordinates,
        coord_type="redund",
        coord_kwargs={"typed_prims": typed_prims},
    )


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    seed_dir = (workdir / args.seed_dir).resolve()
    run_dir = (workdir / args.run_dir).resolve()

    def artifact(kind: str) -> Path:
        path = seed_dir / f"{args.seed_stem}.{kind}"
        if not path.is_file():
            raise FileNotFoundError(f"Required seed artifact is absent: {path}")
        return path

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {run_dir}. Choose a new --run-dir."
        )

    # ORCA.parse_atoms_coords returns bohr, as required by Geometry.
    seed_atoms, seed_coords = ORCA.parse_atoms_coords(artifact("out"))
    linearize_bends = (
        tuple(map(tuple, args.linearize_bend))
        if args.linearize_bend is not None
        else DEFAULT_LINEARIZED_BENDS
    )
    geom = build_geometry(
        seed_atoms, seed_coords, args.coord_type, linearize_bends
    )
    cart_dim = geom.cart_coords.size
    opt_dim = geom.coords.size
    print(
        f"Geometry dimensions: Cartesian={cart_dim}, "
        f"{args.coord_type.upper()}={opt_dim}"
    )
    if opt_dim > 10 * cart_dim:
        raise RuntimeError(
            f"Implausible coordinate expansion ({opt_dim} versus {cart_dim}); "
            "check geometry units and connectivity."
        )

    seed_calc = ORCA(
        keywords=KEYWORDS,
        blocks=BLOCKS,
        root=args.initial_root,
        nroots=NROOTS,
        charge=2,
        mult=3,
        pal=PAL,
        mem=MEM_PER_CORE_MB,
        out_dir=str(seed_dir),
        base_name="calculator",
    )
    seed_calc.cis = artifact("cis")
    seed_calc.bson = artifact("bson")
    seed_calc.gbw = artifact("gbw")
    seed_calc.out = artifact("out")

    manifest_paths = sorted(seed_dir.glob(args.manifest_glob))
    if len(manifest_paths) != 1:
        raise RuntimeError(
            "Expected exactly one accepted seed-survey manifest matching "
            f"{args.manifest_glob!r}, found {manifest_paths}."
        )
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text())
    factor = float(manifest["factor"])
    reference_root = int(manifest["reference_root"])
    if (
        not math.isclose(factor, args.accepted_factor, abs_tol=1.0e-12)
        or reference_root != args.previous_root
    ):
        raise RuntimeError(
            "Seed manifest does not have the requested factor/reference root: "
            f"factor={factor}, reference_root={reference_root}."
        )
    if normalized_atoms(manifest["atoms"]) != normalized_atoms(geom.atoms):
        raise RuntimeError("Seed survey atom identities/order differ from its gradient.")
    survey_coords = np.asarray(manifest["coordinates_bohr"], dtype=float)
    max_coord_error = float(np.max(np.abs(survey_coords - geom.cart_coords)))
    if max_coord_error > 2.0e-5:
        raise RuntimeError(
            "Seed survey and gradient geometries differ by "
            f"{max_coord_error:.3e} bohr."
        )

    reference_roots = [int(root) for root in manifest["reference_roots"]]
    candidate_roots = [int(root) for root in manifest["candidate_roots"]]
    row = reference_roots.index(args.previous_root)
    overlaps = np.asarray(manifest["signed_overlap_matrix"], dtype=float)[row]
    reference_norm = float(np.asarray(manifest["reference_gram"])[row, row])
    candidate_norms = np.diag(
        np.asarray(manifest["candidate_gram"], dtype=float)
    )
    scores = np.abs(overlaps) / np.sqrt(reference_norm * candidate_norms)
    ranking = sorted(zip(scores, candidate_roots), reverse=True)
    best_score, best_root = ranking[0]
    second_score = ranking[1][0]
    if (
        best_root != args.initial_root
        or best_score < 0.95
        or best_score - second_score < 0.50
    ):
        raise RuntimeError(f"Seed root-identity link is not decisive: {ranking[:3]}.")
    print(
        f"Validated identity link: root {args.previous_root} -> root {best_root}, "
        f"factor={factor:g}, normalized overlap={best_score:.8f}, "
        f"margin={best_score - second_score:.8f}."
    )

    seed_root, _ = seed_calc.parse_engrad_info(seed_calc.out)
    if seed_root != args.initial_root:
        raise RuntimeError(
            f"Seed gradient is for root {seed_root!r}, "
            f"expected {args.initial_root}."
        )

    print("Bootstrapping TDenTrack from the validated gradient...")
    initial_snapshot = bootstrap_tdentrack_snapshot(
        calculator=seed_calc,
        atoms=geom.atoms,
        coordinates=geom.cart_coords,
        selected_root=args.initial_root,
        requested_roots=ROOTS,
        tda=True,
        validation_tolerance=VALIDATION_TOLERANCE,
    )
    if initial_snapshot.metadata.get("root_numbering") != GLOBAL_STATE_ROOTS:
        raise RuntimeError(
            "This unrestricted calculation must use global ORCA STATE/IRoot "
            "numbering."
        )

    manifest_energy = float(
        manifest["candidate_energies_eh"][str(args.initial_root)]
    )
    snapshot_energy = float(initial_snapshot.energies_eh[args.initial_root])
    if abs(manifest_energy - snapshot_energy) > 5.0e-5:
        raise RuntimeError(
            "Seed survey and gradient bootstrap disagree on selected-state energy."
        )

    cached_gradient = exact_engrad(artifact("engrad"))
    raw_engrad_energy = float(cached_gradient["energy"])
    final_anchor_energy = float(
        initial_snapshot.metadata["final_single_point_energy_eh"]
    )
    if abs(raw_engrad_energy - final_anchor_energy) > 5.0e-6:
        raise RuntimeError(
            "Cached EnGrad energy disagrees with the audited final energy."
        )
    cached_gradient["energy"] = snapshot_energy
    print(f"Raw cached EnGrad energy: {raw_engrad_energy:.12f} Eh")
    print(f"Bootstrap selected-state energy: {snapshot_energy:.12f} Eh")
    print(f"Validated manifest: {manifest_path}")

    if args.preflight_only or os.environ.get("TDENTRACK_PREFLIGHT_ONLY") == "1":
        print("Preflight complete; no run directory or ORCA job was created.")
        return

    track_calc = TDenTrackORCA(
        initial_snapshot=initial_snapshot,
        enable_default_survey=True,
        default_survey_options={
            "requested_roots": ROOTS,
            "expected_orca_version": "6.1.1",
            "validation_tolerance": VALIDATION_TOLERANCE,
            "tda": True,
        },
        keywords=KEYWORDS,
        blocks=BLOCKS,
        root=args.initial_root,
        nroots=NROOTS,
        charge=2,
        mult=3,
        pal=PAL,
        mem=MEM_PER_CORE_MB,
        out_dir=str(run_dir),
        gbw=str(initial_snapshot.artifacts["gbw"]),
    )
    geom.set_calculator(track_calc)
    track_calc.last_orca_engrad_energy = raw_engrad_energy
    geom.set_results(cached_gradient)
    print("Loaded the validated cycle-0 energy and gradient.")

    step_controller = {
        "type": "state_aware",
        "factors": (0.5, 0.75, 1.0, 1.25, 1.5),
        "primary_factor": 1.0,
        "fallback_only": True,
        "require_descent": True,
    }
    print(f"Starting fresh TDenTrack-aware optimization in {run_dir}...")
    optimizer = RFOptimizer(
        geom,
        max_cycles=args.max_cycles,
        dump=True,
        dump_restart=1,
        out_dir=str(run_dir),
        step_controller=step_controller,
    )
    optimizer.run()
    print("Optimization finished.")


if __name__ == "__main__":
    main()

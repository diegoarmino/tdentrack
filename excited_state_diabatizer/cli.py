from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .basis_overlap import MoldenCache, get_cross_overlap, load_same_geometry_overlap, validate_orbital_orthonormality
from .cis_io import (
    validate_cis_against_output,
    write_cis_manifest,
    write_orca_2json_full_config_template,
    write_orca_2json_mo_only_config,
)
from .json_io import parse_orca_json, require_nto_json
from .models import (
    CISAmplitudeCheck,
    DiabatizationError,
    ExtractionStatus,
    MissingDataError,
    NTOIndexMapping,
    NTOOrbitalPair,
    NTOStateVector,
    NTOVisualizationRecord,
    OrthonormalityError,
    OrthonormalityResult,
    SelfNTOReconstructionRecord,
    SelfNTOWeightRecord,
    StepData,
)
from .molden_io import build_nto_state_from_molden, coefficients_for_validation, parse_molden, select_nto_contributions
from .nto_from_tden import derive_self_ntos_from_tden_steps
from .orca_auxiliary import orca_auxiliary_cross_overlap
from .orca_io import filter_states, find_workdir_jobs, nto_path_for_job, parse_int_range, parse_tddft_output, read_jobs_csv
from .overlap import cis_transition_density_overlap_matrix, nto_similarity_matrix
from .report import ensure_output_dirs, write_outputs
from .self_nto_cubes import render_self_nto_visualizations
from .segments import build_track_segments, make_reversed_matrix_provider
from .tracking import TrackingConfig, run_tracking


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="TDenTrack: numerical ORCA TDDFT excited-state tracking/diabatization post-processor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--workdir", default=".", help="Workflow directory containing tddft/tddft_step_* jobs.")
    ap.add_argument("--jobs-csv", help="Generic jobs CSV with order,label,step,out,geom,gbw,uno,nto_pattern columns.")
    ap.add_argument("--roots", default="1-15", help="Root list/range, e.g. 1-15 or 1,3,5.")
    ap.add_argument("--engine", choices=["tden-json", "nto-json", "nto-molden"], default="tden-json")
    ap.add_argument("--assignment", choices=["hungarian"], default="hungarian")
    ap.add_argument("--adaptive-reference", action="store_true", help="Shortcut for --reference-mode adaptive.")
    ap.add_argument("--bidirectional", action="store_true", help="Run forward/backward adjacent assignment consistency checks.")
    ap.add_argument("--reverse-tracking", action="store_true", help="Run a full right-to-left tracking pass and write reverse diagnostics.")
    ap.add_argument(
        "--consensus-tracking",
        action="store_true",
        help="Run full reverse tracking and report locally stable forward/reverse/two-sided track segments.",
    )
    ap.add_argument("--subspace-detection", dest="subspace_detection", action="store_true", default=True)
    ap.add_argument("--no-subspace-detection", dest="subspace_detection", action="store_false")
    ap.add_argument("--outdir", help="Output directory. Defaults to WORKDIR/state_tracking.")
    ap.add_argument("--html", action="store_true", help="Write state_tracking_report.html.")
    ap.add_argument("--orca-2json", default="orca_2json")
    ap.add_argument("--orca", default=None, help="ORCA executable for --ao-overlap-mode orca-auxiliary. Defaults to sibling of --orca-2json or PATH.")
    ap.add_argument("--orca-2mkl", default="orca_2mkl")
    ap.add_argument("--force-json", action="store_true")
    ap.add_argument("--force-molden", action="store_true")
    ap.add_argument("--similarity-threshold-good", type=float, default=0.70)
    ap.add_argument("--similarity-threshold-low", type=float, default=0.45)
    ap.add_argument("--assignment-margin-threshold", type=float, default=0.10)
    ap.add_argument("--subspace-gap-ev", type=float, default=0.10)
    ap.add_argument("--subspace-ratio-threshold", type=float, default=0.85)
    ap.add_argument("--max-subspace-size", type=int, default=3)
    ap.add_argument("--reference-mode", choices=["previous", "first", "adaptive", "hybrid"], default="previous")
    ap.add_argument("--anchor-weight", type=float, default=0.25)
    ap.add_argument("--previous-weight", type=float, default=0.65)
    ap.add_argument("--predictor-weight", type=float, default=0.10)
    ap.add_argument("--energy-penalty-weight", type=float, default=0.00)
    ap.add_argument("--s2-penalty-weight", type=float, default=0.00)
    ap.add_argument(
        "--nto-purity-guard",
        dest="nto_purity_guard",
        action="store_true",
        default=True,
        help=(
            "For tden-json, treat roots whose dominant self-derived NTO pair carries less than "
            "--nto-purity-threshold of the selected NTO norm as mixed/provisional. With "
            "--ambiguity-rescue enabled, subsequent assignments are compared to the last unmixed reference."
        ),
    )
    ap.add_argument("--no-nto-purity-guard", dest="nto_purity_guard", action="store_false")
    ap.add_argument("--nto-purity-threshold", type=float, default=0.80)
    ap.add_argument(
        "--ambiguity-rescue",
        dest="ambiguity_rescue",
        action="store_true",
        default=True,
        help="When a track becomes provisional, compare the next step to the last stable anchor instead of the mixed previous root.",
    )
    ap.add_argument("--no-ambiguity-rescue", dest="ambiguity_rescue", action="store_false")
    ap.add_argument("--nto-weight-min", type=float, default=0.01)
    ap.add_argument("--nto-cumulative", type=float, default=0.995)
    ap.add_argument("--segment-min-purity", type=float, default=0.90, help="Minimum dominant self-derived NTO fraction for local branch segments.")
    ap.add_argument("--segment-min-adjacent-similarity", type=float, default=0.85, help="Minimum adjacent TD overlap for local branch segments.")
    ap.add_argument("--segment-min-length", type=int, default=2, help="Minimum number of scan points in a reported local branch segment.")
    ap.add_argument(
        "--segment-seed-only-pure-roots",
        dest="segment_seed_only_pure_roots",
        action="store_true",
        default=True,
        help="Only report branch segments whose seed/root endpoints satisfy --segment-min-purity.",
    )
    ap.add_argument("--no-segment-seed-only-pure-roots", dest="segment_seed_only_pure_roots", action="store_false")
    ap.add_argument("--nto-index-shift", type=int, default=0)
    ap.add_argument(
        "--nto-pair-selection",
        choices=["all", "dominant-per-spin", "dominant-total"],
        default="all",
        help="Select printed NTO pairs used in the NTO overlap. 'all' uses weight/cumulative thresholds; "
        "'dominant-per-spin' keeps only the largest same-spin pair in each spin channel.",
    )
    ap.add_argument(
        "--nto-json-index-window",
        type=int,
        default=0,
        help="Experimental diagnostic for ORCA .nto JSON indexing: compare a one-sided local candidate window around printed NTO indices. "
        "Holes use [index-window, index], particles use [index, index+window]. Use with "
        "--nto-pair-selection dominant-per-spin to avoid double-counting overlapping windows. "
        "This is not a proof of the ORCA index convention and should not replace tden-json/job.cis tracking.",
    )
    ap.add_argument(
        "--ao-overlap-mode",
        choices=["auto", "json-cross", "pyscf-cross", "orca-auxiliary"],
        default="auto",
        help=(
            "Cross-geometry AO overlap backend. auto uses ORCA auxiliary cross-overlap jobs for "
            "tden-json/nto-json and PySCF cross integrals for nto-molden."
        ),
    )
    ap.add_argument("--require-orthonormality-check", action="store_true", help="Retained for API clarity; checks are mandatory.")
    ap.add_argument("--allow-failed-orthonormality", action="store_true")
    ap.add_argument(
        "--render-self-nto-cubes",
        action="store_true",
        help="For tden-json, write Gaussian cube files for self-derived CIS/TDA NTOs and render PNGs with Jmol. Requires PySCF.",
    )
    ap.add_argument("--self-nto-plot-roots", help="Optional root subset for self-derived NTO visualization, e.g. 5-9.")
    ap.add_argument("--self-nto-plot-steps", help="Optional step subset for self-derived NTO visualization, e.g. p004,p005 or 4-6.")
    ap.add_argument("--self-nto-plot-weight-min", type=float, default=0.10, help="Minimum self-derived NTO weight rendered as a cube.")
    ap.add_argument("--self-nto-plot-max-pairs", type=int, default=2, help="Maximum self-derived NTO pairs rendered per root.")
    ap.add_argument("--self-nto-cube-grid", type=int, nargs=3, default=[64, 64, 64], help="Grid dimensions for self-derived NTO cube files.")
    ap.add_argument("--self-nto-cube-margin-ang", type=float, default=3.0, help="Cube padding around the molecule in Angstrom.")
    ap.add_argument(
        "--self-nto-cube-overlap-tol",
        type=float,
        default=1.0e-6,
        help="Required max absolute difference between PySCF and ORCA same-geometry AO overlap before cube generation.",
    )
    ap.add_argument("--self-nto-no-render-png", action="store_true", help="Write cubes only; do not call Jmol.")
    ap.add_argument("--force-cubes", action="store_true", help="Regenerate self-derived NTO cube and PNG files even if cached.")
    ap.add_argument("--jmol", default="jmol", help="Jmol executable for self-derived NTO cube rendering.")
    ap.add_argument("--jmol-cutoff", type=float, default=0.05, help="Jmol isosurface cutoff for self-derived NTO PNGs.")
    ap.add_argument("--jmol-rotate-y", type=float, default=-90.0)
    ap.add_argument("--jmol-rotate-x", type=float, default=-50.0)
    ap.add_argument("--jmol-zoom", type=int, default=150)
    ap.add_argument("--jmol-width", type=int, default=420)
    ap.add_argument("--jmol-height", type=int, default=315)
    ap.add_argument("--jmol-timeout", type=int, default=120)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.adaptive_reference:
        args.reference_mode = "adaptive"
    if args.consensus_tracking:
        args.reverse_tracking = True
    if args.segment_min_length < 2:
        parser.error("--segment-min-length must be at least 2.")
    if args.nto_json_index_window < 0:
        parser.error("--nto-json-index-window must be non-negative.")
    if args.nto_json_index_window > 0 and args.nto_pair_selection == "all":
        parser.error(
            "--nto-json-index-window > 0 requires --nto-pair-selection dominant-per-spin "
            "or dominant-total; windowed all-pair NTO overlaps double-count overlapping local "
            "NTO neighborhoods."
        )
    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else workdir / "state_tracking"
    ensure_output_dirs(outdir)
    roots = parse_int_range(args.roots)
    extraction_statuses: List[ExtractionStatus] = []
    orth_results = []
    nto_index_mappings: List[NTOIndexMapping] = []
    cis_amplitude_checks: List[CISAmplitudeCheck] = []
    self_nto_weights: List[SelfNTOWeightRecord] = []
    self_nto_reconstruction: List[SelfNTOReconstructionRecord] = []
    self_nto_visualizations: List[NTOVisualizationRecord] = []
    steps: List[StepData] = []
    result = None
    reverse_result = None
    track_segments = []
    try:
        jobs = read_jobs_csv(Path(args.jobs_csv)) if args.jobs_csv else find_workdir_jobs(workdir)
        if not jobs:
            raise MissingDataError("No TDDFT jobs were found.")
        _probe_utilities(args, outdir, extraction_statuses)
        steps = _load_step_metadata(jobs, roots, extraction_statuses)
        same_overlaps = _load_same_overlaps(jobs, outdir, args, extraction_statuses)
        if args.engine == "tden-json":
            _load_tden_json_or_fail(steps, roots, outdir, args, extraction_statuses, cis_amplitude_checks, orth_results, same_overlaps)
            self_nto_weights, self_nto_reconstruction, self_nto_statuses = derive_self_ntos_from_tden_steps(
                steps,
                roots,
                weight_min=args.nto_weight_min,
                cumulative=args.nto_cumulative,
                cache_dir=outdir / "json_cache" / "self_nto_cache",
                force=args.force_json or args.render_self_nto_cubes or args.nto_purity_guard,
            )
            extraction_statuses.extend(self_nto_statuses)
        elif args.engine == "nto-json":
            _load_nto_json_vectors(steps, roots, outdir, args, extraction_statuses, orth_results, same_overlaps, nto_index_mappings)
        elif args.engine == "nto-molden":
            _load_nto_molden_vectors(
                steps,
                roots,
                outdir,
                args,
                extraction_statuses,
                orth_results,
                same_overlaps,
                nto_index_mappings,
            )
        _enforce_orthonormality(orth_results, args)
        matrix_provider = _make_matrix_provider(steps, roots, same_overlaps, args, outdir, extraction_statuses)
        config = TrackingConfig(
            assignment=args.assignment,
            reference_mode=args.reference_mode,
            similarity_threshold_good=args.similarity_threshold_good,
            similarity_threshold_low=args.similarity_threshold_low,
            assignment_margin_threshold=args.assignment_margin_threshold,
            subspace_detection=args.subspace_detection,
            subspace_gap_ev=args.subspace_gap_ev,
            subspace_ratio_threshold=args.subspace_ratio_threshold,
            max_subspace_size=args.max_subspace_size,
            anchor_weight=args.anchor_weight,
            previous_weight=args.previous_weight,
            predictor_weight=args.predictor_weight,
            energy_penalty_weight=args.energy_penalty_weight,
            s2_penalty_weight=args.s2_penalty_weight,
            nto_purity_guard=args.nto_purity_guard,
            nto_purity_threshold=args.nto_purity_threshold,
            ambiguity_rescue=args.ambiguity_rescue,
        )
        result = run_tracking(steps, roots, matrix_provider, config, args.engine)
        if not args.bidirectional:
            result.bidirectional_disagreements = []
        if args.reverse_tracking:
            reverse_provider = make_reversed_matrix_provider(matrix_provider, len(steps))
            reverse_result = run_tracking(list(reversed(steps)), roots, reverse_provider, config, f"{args.engine}-reverse")
            if not args.bidirectional:
                reverse_result.bidirectional_disagreements = []
        if args.consensus_tracking or args.reverse_tracking:
            track_segments = build_track_segments(
                result,
                reverse_result,
                steps,
                min_purity=args.segment_min_purity,
                min_adjacent_similarity=args.segment_min_adjacent_similarity,
                min_length=args.segment_min_length,
                seed_only_pure_roots=args.segment_seed_only_pure_roots,
            )
        if args.render_self_nto_cubes:
            if args.engine != "tden-json":
                extraction_statuses.append(
                    ExtractionStatus(
                        "",
                        None,
                        "self-nto-cubes",
                        status="skipped",
                        message="Self-derived NTO cubes are only available for --engine tden-json.",
                    )
                )
            else:
                self_nto_visualizations = render_self_nto_visualizations(steps, roots, outdir, args, extraction_statuses)
        write_outputs(
            outdir,
            result,
            steps,
            extraction_statuses,
            orth_results,
            nto_index_mappings=nto_index_mappings,
            cis_amplitude_checks=cis_amplitude_checks,
            self_nto_weights=self_nto_weights,
            self_nto_reconstruction=self_nto_reconstruction,
            self_nto_visualizations=self_nto_visualizations,
            write_html=args.html,
            selected_roots=roots,
            engine=args.engine,
            nto_purity_threshold=args.nto_purity_threshold,
            nto_purity_guard=args.nto_purity_guard,
            reverse_result=reverse_result,
            track_segments=track_segments,
        )
        print(f"Wrote state tracking outputs to {outdir}")
        return 0
    except DiabatizationError as exc:
        msg = str(exc)
        write_outputs(
            outdir,
            result,
            steps,
            extraction_statuses,
            orth_results,
            nto_index_mappings=nto_index_mappings,
            cis_amplitude_checks=cis_amplitude_checks,
            self_nto_weights=self_nto_weights,
            self_nto_reconstruction=self_nto_reconstruction,
            self_nto_visualizations=self_nto_visualizations,
            write_html=args.html,
            failure_message=msg,
            selected_roots=roots,
            engine=args.engine,
            nto_purity_threshold=args.nto_purity_threshold,
            nto_purity_guard=args.nto_purity_guard,
            reverse_result=reverse_result,
            track_segments=track_segments,
        )
        print(f"orca_diabatize failed: {msg}", file=sys.stderr)
        print(f"Partial diagnostics were written to {outdir}", file=sys.stderr)
        return 2


def _probe_utilities(args, outdir: Path, statuses: List[ExtractionStatus]) -> None:
    if args.engine in {"tden-json", "nto-json"}:
        exe = _resolve_executable(args.orca_2json)
        if exe is None:
            statuses.append(
                ExtractionStatus("", None, "orca_2json", status="missing", message=f"{args.orca_2json!r} was not found on PATH.")
            )
        else:
            _capture_help(exe, outdir / "utility_logs" / "orca_2json_help.txt")
            statuses.append(ExtractionStatus("", None, "orca_2json", source_file=Path(exe), status="available", message="help/version output captured"))
    if args.engine in {"tden-json", "nto-json"} and args.ao_overlap_mode in {"auto", "orca-auxiliary"}:
        exe = _resolve_orca_executable(args)
        if exe is None:
            statuses.append(ExtractionStatus("", None, "orca", status="missing", message="ORCA executable was not found for auxiliary cross-overlap jobs."))
        else:
            statuses.append(ExtractionStatus("", None, "orca", source_file=Path(exe), status="available", message="ORCA executable resolved for auxiliary cross-overlap jobs."))
    if args.engine == "nto-molden":
        exe = _resolve_executable(args.orca_2mkl)
        if exe is None:
            statuses.append(
                ExtractionStatus("", None, "orca_2mkl", status="missing", message=f"{args.orca_2mkl!r} was not found on PATH.")
            )
        else:
            _capture_help(exe, outdir / "utility_logs" / "orca_2mkl_help.txt")
            statuses.append(ExtractionStatus("", None, "orca_2mkl", source_file=Path(exe), status="available", message="help/version output captured"))


def _load_step_metadata(jobs, roots, statuses: List[ExtractionStatus]) -> List[StepData]:
    steps: List[StepData] = []
    for job in jobs:
        _, states, normal = parse_tddft_output(job.out_path, job.label, job.order, job.scan_step)
        states = filter_states(states, roots)
        status = "ok" if normal and states else "warning" if states else "failed"
        msg = "ORCA terminated normally and STATE/NTO data were parsed." if normal and states else "STATE lines parsed but normal termination was not detected." if states else "No requested STATE lines were parsed."
        statuses.append(ExtractionStatus(job.label, None, "orca-output", source_file=job.out_path, status=status, message=msg))
        missing = [r for r in roots if r not in states]
        for root in missing:
            statuses.append(ExtractionStatus(job.label, root, "orca-output", source_file=job.out_path, status="missing", message="Requested root was not parsed from job.out."))
        steps.append(StepData(job=job, states=states))
    return steps


def _load_same_overlaps(jobs, outdir: Path, args, statuses: List[ExtractionStatus]) -> Dict[int, np.ndarray]:
    same: Dict[int, np.ndarray] = {}
    for job in jobs:
        src = job.step_dir / "job.json"
        cache = outdir / "json_cache" / f"{job.label}_job.json"
        if src.exists() and (args.force_json or not cache.exists()):
            shutil.copy2(src, cache)
        use = cache if cache.exists() else src
        if (args.force_json or not use.exists()) and args.engine in {"tden-json", "nto-json"}:
            generated = _try_generate_rich_orca_json(job, cache, outdir, args, statuses)
            if generated is not None:
                use = generated
        if not use.exists():
            statuses.append(ExtractionStatus(job.label, None, "orca-json-overlap", source_file=src, json_file=cache, status="missing", message="No job.json S-Matrix export is available for orthonormality validation."))
            continue
        try:
            same[job.order] = load_same_geometry_overlap(use)
            statuses.append(ExtractionStatus(job.label, None, "orca-json-overlap", source_file=src, json_file=use, status="ok", message="same-geometry AO S-Matrix loaded"))
        except DiabatizationError as exc:
            statuses.append(ExtractionStatus(job.label, None, "orca-json-overlap", source_file=src, json_file=use, status="failed", message=str(exc)))
    return same


def _load_tden_json_or_fail(
    steps: Sequence[StepData],
    roots: Sequence[int],
    outdir: Path,
    args,
    statuses: List[ExtractionStatus],
    cis_checks: List[CISAmplitudeCheck],
    orth_results,
    same_overlaps,
) -> None:
    config_template = outdir / "json_cache" / "orca_2json_full_export_template.json.conf"
    write_orca_2json_full_config_template(config_template)
    statuses.append(
        ExtractionStatus(
            "",
            None,
            "orca-json-config-template",
            json_file=config_template,
            status="ok",
            message="Template written for a richer orca_2json export with MOCoefficients=true, Basisset=true, and S integrals.",
        )
    )
    blockers = []
    for step in steps:
        job = step.job
        json_file = outdir / "json_cache" / f"{job.label}_job.json"
        if args.force_json:
            generated = _try_generate_rich_orca_json(job, json_file, outdir, args, statuses)
            if generated is not None:
                json_file = generated
        if not json_file.exists() and (job.step_dir / "job.json").exists():
            shutil.copy2(job.step_dir / "job.json", json_file)
        parsed = None
        has_mo_coefficients = False
        if not json_file.exists():
            statuses.append(ExtractionStatus(job.label, None, "tden-json", source_file=job.gbw_path, json_file=json_file, status="missing", message="No cached JSON export."))
        else:
            parsed = parse_orca_json(json_file)
            has_mo_coefficients = bool(parsed.mo_coefficients)
            if not has_mo_coefficients:
                generated = _try_generate_rich_orca_json(job, json_file, outdir, args, statuses)
                if generated is not None:
                    json_file = generated
                    parsed = parse_orca_json(json_file)
                    has_mo_coefficients = bool(parsed.mo_coefficients)
            if parsed is not None and parsed.overlap is not None and job.order not in same_overlaps:
                same_overlaps[job.order] = parsed.overlap
            if parsed.tden_states and has_mo_coefficients:
                step.tden_vectors.update({root: rec for root, rec in parsed.tden_states.items() if root in roots})
                step.mo_coefficients = parsed.mo_coefficients
                step.json_file = json_file
                statuses.append(
                    ExtractionStatus(
                        job.label,
                        None,
                        "tden-json",
                        source_file=job.gbw_path,
                        json_file=json_file,
                        status="ok",
                        message="JSON TDDFT transition-density/amplitude data and MO coefficients were found.",
                    )
                )
            else:
                parts = []
                if not parsed.tden_states:
                    parts.append("no JSON TDDFT transition-density/amplitude records")
                if not has_mo_coefficients:
                    parts.append("no MO coefficients")
                statuses.append(
                    ExtractionStatus(
                        job.label,
                        None,
                        "tden-json",
                        source_file=job.gbw_path,
                        json_file=json_file,
                        status="warning",
                        message="; ".join(parts) + ". Trying ORCA job.cis amplitudes where available.",
                    )
                )
            if has_mo_coefficients and parsed is not None:
                step.mo_coefficients = parsed.mo_coefficients
                step.json_file = json_file
                if job.order in same_overlaps:
                    for spin, coeffs in parsed.mo_coefficients.items():
                        orth_results.append(
                            validate_orbital_orthonormality(
                                coeffs,
                                same_overlaps[job.order],
                                job.label,
                                json_file,
                                f"tden-json-mo-{spin}",
                            )
                        )

        cis_path = job.step_dir / "job.cis"
        if cis_path.exists():
            try:
                header, amplitudes, rows = validate_cis_against_output(cis_path, job.out_path, job.label, roots)
                cis_checks.extend(rows)
                failed_rows = [row for row in rows if not row.passed]
                manifest = outdir / "json_cache" / f"{job.label}_cis_tden_manifest.json"
                write_cis_manifest(
                    manifest,
                    cis_path,
                    header,
                    list(amplitudes),
                    rows,
                    amplitudes=amplitudes,
                )
                if failed_rows:
                    statuses.append(
                        ExtractionStatus(
                            job.label,
                            None,
                            "cis-binary",
                            source_file=cis_path,
                            json_file=manifest,
                            status="failed",
                            message=f"Parsed .cis amplitudes, but {len(failed_rows)} printed coefficient checks failed.",
                        )
                    )
                    blockers.append(f"{job.label}: .cis amplitude validation failed; see diagnostics/cis_amplitude_checks.csv")
                else:
                    step.tden_vectors.update(amplitudes)
                    checked_msg = f"{len(rows)} printed coefficients validated" if rows else "no printed c= coefficients were available for validation"
                    statuses.append(
                        ExtractionStatus(
                            job.label,
                            None,
                            "cis-binary",
                            source_file=cis_path,
                            json_file=manifest,
                            status="ok" if rows else "warning",
                            message=(
                                f"Loaded ORCA .cis amplitudes for {len(amplitudes)} roots; {checked_msg}. "
                                f"Active alpha shape {header.alpha_nocc}x{header.alpha_nvirt}, beta shape {header.beta_nocc}x{header.beta_nvirt}."
                            ),
                        )
                    )
            except Exception as exc:
                statuses.append(ExtractionStatus(job.label, None, "cis-binary", source_file=cis_path, status="failed", message=str(exc)))
                blockers.append(f"{job.label}: {exc}")
        else:
            statuses.append(ExtractionStatus(job.label, None, "cis-binary", source_file=cis_path, status="missing", message="No ORCA job.cis file found."))
            blockers.append(f"{job.label}: no job.cis")

        if step.tden_vectors and not has_mo_coefficients:
            blockers.append(
                f"{job.label}: .cis amplitudes are available, but {json_file} lacks MO coefficients. "
                "The local job.json.conf has likely disabled MOCoefficients; regenerate JSON with the template in json_cache."
            )
        if not step.tden_vectors:
            blockers.append(f"{job.label}: no transition-density amplitudes were loaded from JSON or job.cis.")
        if step.tden_vectors and not step.mo_coefficients:
            blockers.append(f"{job.label}: transition-density amplitudes are available, but MO coefficients are missing.")
    if blockers:
        raise MissingDataError(
            "tden-json found ORCA .cis amplitudes where available, but cannot yet construct trustworthy cross-geometry "
            "transition-density overlaps without validated MO coefficients and AO cross-overlaps. First blocker: "
            + blockers[0]
        )


def _try_generate_rich_orca_json(job, cache: Path, outdir: Path, args, statuses: List[ExtractionStatus]) -> Optional[Path]:
    exe = _resolve_executable(args.orca_2json)
    if exe is None:
        statuses.append(
            ExtractionStatus(
                job.label,
                None,
                "orca_2json-rich",
                source_file=job.gbw_path,
                json_file=cache,
                status="missing",
                message=f"{args.orca_2json!r} was not found on PATH; cannot regenerate rich JSON.",
            )
        )
        return None
    if job.gbw_path is None or not Path(job.gbw_path).exists():
        statuses.append(
            ExtractionStatus(
                job.label,
                None,
                "orca_2json-rich",
                source_file=job.gbw_path,
                json_file=cache,
                status="missing",
                message="No GBW file is available for rich orca_2json export.",
            )
        )
        return None
    tmp_parent = outdir / "json_cache"
    log = outdir / "utility_logs" / f"orca_2json_{job.label}_rich.log"
    saved_config = outdir / "json_cache" / f"{job.label}_job.full.json.conf"
    write_orca_2json_full_config_template(saved_config)
    with tempfile.TemporaryDirectory(prefix=f"{job.label}_orca_2json_", dir=tmp_parent) as td:
        tmp = Path(td)
        tmp_gbw = tmp / "job.gbw"
        try:
            os.symlink(Path(job.gbw_path), tmp_gbw)
        except Exception:
            shutil.copy2(job.gbw_path, tmp_gbw)
        write_orca_2json_full_config_template(tmp / "job.json.conf")
        cmd = [exe, "job.gbw"]
        try:
            proc = subprocess.run(cmd, cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as exc:
            log.write_text("COMMAND: " + " ".join(cmd) + f"\nERROR: {exc}\n")
            statuses.append(
                ExtractionStatus(
                    job.label,
                    None,
                    "orca_2json-rich",
                    source_file=job.gbw_path,
                    json_file=cache,
                    status="failed",
                    message=f"orca_2json execution failed; see {log}.",
                )
            )
            return None
        log.write_text(
            "COMMAND: "
            + " ".join(cmd)
            + "\nCONFIG:\n"
            + (tmp / "job.json.conf").read_text()
            + f"\nreturncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
        produced = _first_existing([tmp / "job.json", tmp / "job.gbw.json"])
        if proc.returncode == 0 and produced is not None:
            shutil.copy2(produced, cache)
            statuses.append(
                ExtractionStatus(
                    job.label,
                    None,
                    "orca_2json-rich",
                    source_file=job.gbw_path,
                    json_file=cache,
                    status="ok",
                    message=f"Regenerated cached JSON with full export config; utility log: {log}.",
                )
            )
            return cache
        statuses.append(
            ExtractionStatus(
                job.label,
                None,
                "orca_2json-rich",
                source_file=job.gbw_path,
                json_file=cache,
                status="failed",
                message=f"orca_2json did not produce a recognized JSON file; see {log}.",
            )
        )
        return None


def _load_nto_json_vectors(steps, roots, outdir: Path, args, statuses, orth_results, same_overlaps, nto_index_mappings) -> None:
    failures = []
    for step in steps:
        for root in roots:
            source = nto_path_for_job(step.job, root)
            json_file = outdir / "json_cache" / f"{step.job.label}_s{root}.nto.json"
            if not json_file.exists() or args.force_json:
                for adjacent in (source.with_suffix(source.suffix + ".json"), source.with_suffix(".json")):
                    if adjacent.exists():
                        shutil.copy2(adjacent, json_file)
                        break
            if (not json_file.exists() or args.force_json) and source.exists():
                generated = _try_generate_nto_json(source, json_file, outdir, args, step.job.label, root, statuses)
                if generated is not None:
                    json_file = generated
            if not json_file.exists():
                msg = "No JSON NTO export found. orca_2json automatic .nto conversion is version-dependent; provide a cached .nto JSON export or use --engine nto-molden."
                statuses.append(ExtractionStatus(step.job.label, root, "nto-json", source_file=source, json_file=json_file, status="missing", message=msg))
                failures.append(f"{step.job.label} root {root}: {msg}")
                continue
            try:
                parsed = parse_orca_json(json_file)
                if parsed.nto_states:
                    require_nto_json(parsed)
                    vec = _build_nto_state_from_json(parsed.nto_states[root], step.job.label, step.job.order, root, json_file, args)
                elif parsed.mo_coefficients:
                    state = step.states.get(root)
                    if state is None or not state.contribs:
                        raise MissingDataError("NTO JSON contains orbital coefficients, but no printed NTO contribution block was parsed from job.out.")
                    vec = _build_nto_state_from_orca_json(parsed.mo_coefficients, state.contribs, step.job.label, step.job.order, root, json_file, args)
                else:
                    require_nto_json(parsed)
                step.nto_vectors[root] = vec
                statuses.append(ExtractionStatus(step.job.label, root, "nto-json", source_file=source, json_file=json_file, status="ok", message=f"{len(vec.pairs)} NTO pairs loaded"))
                validation = _validate_vector(vec, step.job.order, same_overlaps, orth_results)
                _append_mapping_rows(nto_index_mappings, vec, validation)
            except Exception as exc:
                statuses.append(ExtractionStatus(step.job.label, root, "nto-json", source_file=source, json_file=json_file, status="failed", message=str(exc)))
                failures.append(f"{step.job.label} root {root}: {exc}")
    if failures:
        raise MissingDataError("nto-json could not load required NTO coefficients. First failure: " + failures[0])


def _try_generate_nto_json(source: Path, cache: Path, outdir: Path, args, label: str, root: int, statuses) -> Optional[Path]:
    exe = _resolve_executable(args.orca_2json)
    if exe is None:
        statuses.append(ExtractionStatus(label, root, "orca_2json-nto", source_file=source, json_file=cache, status="missing", message="orca_2json was not found."))
        return None
    saved_config = cache.with_suffix(cache.suffix + ".conf")
    write_orca_2json_mo_only_config(saved_config)
    with tempfile.TemporaryDirectory(prefix=f"{label}_s{root}_nto_json_", dir=outdir / "json_cache") as td:
        work = Path(td)
        local = work / source.name
        try:
            os.symlink(source, local)
        except Exception:
            shutil.copy2(source, local)
        shutil.copy2(saved_config, work / (source.stem + ".json.conf"))
        shutil.copy2(saved_config, work / "orca.json.conf")
        cmd = [exe, source.name, "-json"]
        try:
            proc = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as exc:
            log = outdir / "utility_logs" / f"orca_2json_nto_{label}_s{root}.log"
            log.write_text("COMMAND: " + " ".join(cmd) + f"\nCONFIG: {saved_config}\nERROR: {exc}\n")
            statuses.append(ExtractionStatus(label, root, "orca_2json-nto", source_file=source, json_file=cache, status="failed", message=f"orca_2json .nto conversion failed; see {log}."))
            return None
        log = outdir / "utility_logs" / f"orca_2json_nto_{label}_s{root}.log"
        log.write_text(
            "COMMAND: "
            + " ".join(cmd)
            + f"\nCONFIG: {saved_config}"
            + f"\nreturncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
        produced = _first_existing([work / (source.stem + ".json"), work / (source.name + ".json"), work / "job.json"])
        if proc.returncode == 0 and produced is not None:
            shutil.copy2(produced, cache)
            statuses.append(ExtractionStatus(label, root, "orca_2json-nto", source_file=source, json_file=cache, status="ok", message="orca_2json produced JSON from .nto source."))
            return cache
        statuses.append(
            ExtractionStatus(
                label,
                root,
                "orca_2json-nto",
                source_file=source,
                json_file=cache,
                status="failed",
                message=f"orca_2json did not produce NTO JSON; return code {proc.returncode}. See {log}.",
            )
        )
        return None


def _load_nto_molden_vectors(steps, roots, outdir: Path, args, statuses, orth_results, same_overlaps, nto_index_mappings) -> None:
    failures = []
    for step in steps:
        for root in roots:
            source = nto_path_for_job(step.job, root)
            molden_file = _ensure_molden(source, step.job.label, root, outdir, args, statuses)
            if molden_file is None:
                failures.append(f"{step.job.label} root {root}: no Molden file")
                continue
            state = step.states.get(root)
            if state is None or not state.contribs:
                msg = "No printed NTO contribution block was parsed from job.out for this root."
                statuses.append(ExtractionStatus(step.job.label, root, "nto-molden", source_file=source, molden_file=molden_file, status="failed", message=msg))
                failures.append(f"{step.job.label} root {root}: {msg}")
                continue
            try:
                molden = parse_molden(molden_file)
                vec = build_nto_state_from_molden(
                    molden,
                    step.job.label,
                    step.job.order,
                    root,
                    state.contribs,
                    nto_index_shift=args.nto_index_shift,
                    weight_min=args.nto_weight_min,
                    cumulative=args.nto_cumulative,
                    pair_selection=args.nto_pair_selection,
                )
                step.nto_vectors[root] = vec
                statuses.append(
                    ExtractionStatus(
                        step.job.label,
                        root,
                        "nto-molden",
                        source_file=source,
                        molden_file=molden_file,
                        status="ok",
                        message=f"{len(vec.pairs)} NTO pairs loaded; " + " | ".join(vec.diagnostics[:3]),
                    )
                )
                validation = _validate_vector(vec, step.job.order, same_overlaps, orth_results)
                _append_mapping_rows(nto_index_mappings, vec, validation)
            except Exception as exc:
                statuses.append(ExtractionStatus(step.job.label, root, "nto-molden", source_file=source, molden_file=molden_file, status="failed", message=str(exc)))
                failures.append(f"{step.job.label} root {root}: {exc}")
    if failures:
        raise MissingDataError("nto-molden could not load required NTO coefficients. First failure: " + failures[0])


def _ensure_molden(source: Path, label: str, root: int, outdir: Path, args, statuses) -> Optional[Path]:
    cache = outdir / "molden_cache" / f"{label}_s{root}.nto.molden"
    adjacent = source.with_suffix(source.suffix + ".molden")
    if adjacent.exists() and (args.force_molden or not cache.exists()):
        shutil.copy2(adjacent, cache)
    if cache.exists() and not args.force_molden:
        return cache
    if not source.exists():
        statuses.append(ExtractionStatus(label, root, "orca_2mkl", source_file=source, molden_file=cache, status="missing", message="Source .nto file is missing."))
        return None
    exe = _resolve_executable(args.orca_2mkl)
    if exe is None:
        statuses.append(ExtractionStatus(label, root, "orca_2mkl", source_file=source, molden_file=cache, status="missing", message="orca_2mkl is not available and no cached Molden file exists."))
        return None
    work = source.parent
    tmp_name = cache.name
    cmd = [exe, source.name, tmp_name, "-molden", "-anyorbs"]
    proc = subprocess.run(cmd, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    log = outdir / "utility_logs" / f"orca_2mkl_{label}_s{root}.log"
    log.write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\n\nSTDERR:\n" + proc.stderr)
    produced = work / tmp_name
    if proc.returncode == 0 and produced.exists():
        shutil.move(str(produced), cache)
        statuses.append(ExtractionStatus(label, root, "orca_2mkl", source_file=source, molden_file=cache, status="ok", message="Molden fallback conversion completed."))
        return cache
    statuses.append(ExtractionStatus(label, root, "orca_2mkl", source_file=source, molden_file=cache, status="failed", message=f"orca_2mkl failed; see {log}"))
    return None


def _build_nto_state_from_json(rec: dict, step_label: str, step_order: int, root: int, source: Path, args) -> NTOStateVector:
    pairs = rec.get("pairs") or rec.get("nto_pairs") or rec.get("NTO Pairs") or []
    vec = NTOStateVector(step_label=step_label, step_order=step_order, root=root, source_file=source, source_type="nto-json")
    ordered = sorted(pairs, key=lambda p: float(p.get("weight", p.get("n", 0.0))), reverse=True)
    total = sum(max(0.0, float(p.get("weight", p.get("n", 0.0)))) for p in ordered)
    running = 0.0
    for p in ordered:
        weight = float(p.get("weight", p.get("n", 0.0)))
        if weight < args.nto_weight_min and (total <= 0 or running / total >= args.nto_cumulative):
            continue
        donor = np.asarray(p.get("donor_coeff") or p.get("donor") or p.get("hole"), dtype=float)
        acceptor = np.asarray(p.get("acceptor_coeff") or p.get("acceptor") or p.get("particle"), dtype=float)
        spin = p.get("spin") or p.get("donor_spin") or "alpha"
        vec.pairs.append(
            NTOOrbitalPair(
                spin="alpha" if str(spin).lower().startswith("a") else "beta",
                donor_index=int(p.get("donor_index", 0)),
                acceptor_index=int(p.get("acceptor_index", 0)),
                weight=weight,
                donor_coeff=donor,
                acceptor_coeff=acceptor,
                source_file=source,
            )
        )
        running += max(0.0, weight)
        if total > 0 and running / total >= args.nto_cumulative and weight < args.nto_weight_min:
            break
    return vec


def _build_nto_state_from_orca_json(
    mo_coefficients: Dict[str, np.ndarray],
    contribs,
    step_label: str,
    step_order: int,
    root: int,
    source: Path,
    args,
) -> NTOStateVector:
    selected = _select_nto_contributions_for_tracking(contribs, args)
    vec = NTOStateVector(step_label=step_label, step_order=step_order, root=root, source_file=source, source_type="nto-json")
    for c in selected:
        if c.donor_spin.lower()[:1] != c.acceptor_spin.lower()[:1]:
            vec.diagnostics.append(
                f"Skipped mixed-spin NTO pair {c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin}; "
                "same-spin alpha/alpha and beta/beta pairs are compared separately."
            )
            continue
        spin = "alpha" if c.donor_spin.lower().startswith("a") else "beta"
        coeffs = mo_coefficients.get(spin)
        if coeffs is None:
            raise MissingDataError(f"NTO JSON lacks {spin} coefficient block required for {c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin}.")
        donor_idx = int(c.donor) + int(args.nto_index_shift) - 1
        acc_idx = int(c.acceptor) + int(args.nto_index_shift) - 1
        if donor_idx < 0 or donor_idx >= coeffs.shape[1] or acc_idx < 0 or acc_idx >= coeffs.shape[1]:
            raise MissingDataError(
                f"Printed ORCA NTO pair {c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin} maps to "
                f"vector indices {donor_idx}/{acc_idx}, but {spin} JSON has {coeffs.shape[1]} vectors."
            )
        donor_candidate_indices = _one_sided_nto_candidate_indices(
            donor_idx,
            coeffs.shape[1],
            side="donor",
            window=int(args.nto_json_index_window),
        )
        acceptor_candidate_indices = _one_sided_nto_candidate_indices(
            acc_idx,
            coeffs.shape[1],
            side="acceptor",
            window=int(args.nto_json_index_window),
        )
        vec.pairs.append(
            NTOOrbitalPair(
                spin=spin,
                donor_index=c.donor,
                acceptor_index=c.acceptor,
                weight=c.weight,
                donor_coeff=coeffs[:, donor_idx].copy(),
                acceptor_coeff=coeffs[:, acc_idx].copy(),
                source_file=source,
                donor_vector_index=donor_idx,
                acceptor_vector_index=acc_idx,
                donor_candidate_coeffs=coeffs[:, donor_candidate_indices].copy() if donor_candidate_indices else None,
                acceptor_candidate_coeffs=coeffs[:, acceptor_candidate_indices].copy() if acceptor_candidate_indices else None,
                donor_candidate_indices=donor_candidate_indices,
                acceptor_candidate_indices=acceptor_candidate_indices,
            )
        )
        donor_candidates = _format_vector_indices(donor_candidate_indices)
        acceptor_candidates = _format_vector_indices(acceptor_candidate_indices)
        window_note = ""
        if int(args.nto_json_index_window) > 0:
            window_note = f"; candidate windows donor={donor_candidates}, acceptor={acceptor_candidates}"
        vec.diagnostics.append(
            f"{c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin}: "
            f"printed indices -> JSON vectors {donor_idx + 1}/{acc_idx + 1}, n={c.weight:.8g}{window_note}"
        )
    if not vec.pairs:
        vec.diagnostics.append("No NTO pairs survived weight/cumulative selection and spin filtering.")
    return vec


def _select_nto_contributions_for_tracking(contribs, args):
    selected = select_nto_contributions(contribs, weight_min=args.nto_weight_min, cumulative=args.nto_cumulative)
    mode = getattr(args, "nto_pair_selection", "all")
    if mode == "all":
        return selected
    same_spin = [c for c in selected if c.donor_spin.lower()[:1] == c.acceptor_spin.lower()[:1]]
    if mode == "dominant-total":
        return [max(same_spin, key=lambda c: c.weight)] if same_spin else []
    if mode == "dominant-per-spin":
        out = []
        for spin in ("a", "b"):
            spin_contribs = [c for c in same_spin if c.donor_spin.lower().startswith(spin)]
            if spin_contribs:
                out.append(max(spin_contribs, key=lambda c: c.weight))
        return sorted(out, key=lambda c: c.weight, reverse=True)
    return selected


def _one_sided_nto_candidate_indices(center_idx: int, n_vectors: int, side: str, window: int) -> List[int]:
    if window <= 0:
        return [int(center_idx)]
    center = int(center_idx)
    if side == "donor":
        start = max(0, center - int(window))
        stop = center
    else:
        start = center
        stop = min(n_vectors - 1, center + int(window))
    return list(range(start, stop + 1))


def _format_vector_indices(indices: Sequence[int]) -> str:
    return ",".join(str(i + 1) for i in indices)


def _validate_vector(vec: NTOStateVector, step_order: int, same_overlaps, orth_results):
    if step_order not in same_overlaps:
        result = validate_orbital_orthonormality(
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            vec.step_label,
            vec.source_file,
            vec.source_type,
        )
        orth_results.append(result)
        return result
    if vec.pairs:
        result = _validate_nto_vector_by_spin(vec, same_overlaps[step_order])
    else:
        coeffs = coefficients_for_validation(vec)
        result = validate_orbital_orthonormality(coeffs, same_overlaps[step_order], vec.step_label, vec.source_file, vec.source_type)
    orth_results.append(result)
    return result


def _validate_nto_vector_by_spin(vec: NTOStateVector, overlap: np.ndarray) -> OrthonormalityResult:
    spin_results = []
    for spin in ("alpha", "beta"):
        cols_by_index = {}
        anonymous_cols = []
        for pair in vec.pairs:
            if pair.spin == spin:
                for idx, coeff in _pair_candidate_columns(pair, "donor"):
                    if idx is None:
                        anonymous_cols.append(coeff)
                    else:
                        cols_by_index.setdefault(("donor", idx), coeff)
                for idx, coeff in _pair_candidate_columns(pair, "acceptor"):
                    if idx is None:
                        anonymous_cols.append(coeff)
                    else:
                        cols_by_index.setdefault(("acceptor", idx), coeff)
        cols = list(cols_by_index.values()) + anonymous_cols
        if not cols:
            continue
        coeffs = np.column_stack(cols)
        spin_results.append(
            validate_orbital_orthonormality(
                coeffs,
                overlap,
                vec.step_label,
                vec.source_file,
                f"{vec.source_type}-{spin}",
            )
        )
    if not spin_results:
        return validate_orbital_orthonormality(np.zeros((0, 0)), overlap, vec.step_label, vec.source_file, vec.source_type)
    passed = all(r.passed for r in spin_results)

    def max_optional(values):
        present = [v for v in values if v is not None]
        return max(present) if present else None

    rms_vals = [r.rms_offdiag for r in spin_results if r.rms_offdiag is not None]
    message = "passed" if passed else "Spin-separated C.T S C validation failed for one or more NTO spin blocks."
    return OrthonormalityResult(
        step_label=vec.step_label,
        source_file=vec.source_file,
        source_type=vec.source_type + "-spin-separated",
        max_diag_error=max_optional(r.max_diag_error for r in spin_results),
        max_offdiag=max_optional(r.max_offdiag for r in spin_results),
        rms_offdiag=max(rms_vals) if rms_vals else None,
        passed=passed,
        message=message,
    )


def _pair_candidate_columns(pair: NTOOrbitalPair, side: str):
    if side == "donor":
        indices = pair.donor_candidate_indices
        coeffs = pair.donor_candidate_coeffs
        fallback_idx = pair.donor_vector_index
        fallback = pair.donor_coeff
    else:
        indices = pair.acceptor_candidate_indices
        coeffs = pair.acceptor_candidate_coeffs
        fallback_idx = pair.acceptor_vector_index
        fallback = pair.acceptor_coeff
    if coeffs is None or not indices:
        return [(fallback_idx, fallback)]
    return [(idx, coeffs[:, pos]) for pos, idx in enumerate(indices)]


def _append_mapping_rows(rows, vec: NTOStateVector, validation) -> None:
    status = "passed" if validation.passed else "failed"
    for pair in vec.pairs:
        rows.append(
            NTOIndexMapping(
                step_label=vec.step_label,
                root=vec.root,
                source_file=vec.source_file,
                printed_donor=pair.donor_index,
                donor_spin="a" if pair.spin == "alpha" else "b",
                parsed_donor_vector=None if pair.donor_vector_index is None else pair.donor_vector_index + 1,
                printed_acceptor=pair.acceptor_index,
                acceptor_spin="a" if pair.spin == "alpha" else "b",
                parsed_acceptor_vector=None if pair.acceptor_vector_index is None else pair.acceptor_vector_index + 1,
                weight=pair.weight,
                validation_status=status,
                donor_candidate_vectors=_format_vector_indices(pair.donor_candidate_indices),
                acceptor_candidate_vectors=_format_vector_indices(pair.acceptor_candidate_indices),
                mapping_note=(
                    "one-sided JSON local window"
                    if len(pair.donor_candidate_indices) > 1 or len(pair.acceptor_candidate_indices) > 1
                    else "exact printed index"
                ),
            )
        )


def _enforce_orthonormality(orth_results, args) -> None:
    failed = [r for r in orth_results if not r.passed]
    if failed and not args.allow_failed_orthonormality:
        first = failed[0]
        raise OrthonormalityError(
            f"Same-geometry orbital orthonormality validation failed for {first.source_file}: {first.message} "
            "This usually indicates inconsistent AO ordering, spherical/cartesian convention, normalization, "
            "or NTO index mapping. Pass --allow-failed-orthonormality only for dangerous debugging."
        )


def _make_matrix_provider(steps, roots, same_overlaps, args, outdir: Path, statuses: List[ExtractionStatus]):
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    molden_cache = MoldenCache()

    def provider(i: int, j: int) -> np.ndarray:
        key = (i, j)
        if key in cache:
            return cache[key]
        step_a = steps[i]
        step_b = steps[j]
        if args.engine == "tden-json":
            if not step_a.tden_vectors or not step_b.tden_vectors:
                raise MissingDataError(f"Missing TDEN/CIS amplitudes for {step_a.job.label}->{step_b.job.label}.")
            if not step_a.mo_coefficients or not step_b.mo_coefficients:
                raise MissingDataError(f"Missing MO coefficients for {step_a.job.label}->{step_b.job.label}.")
            s_cross = _tden_cross_overlap(step_a, step_b, same_overlaps, args, outdir, statuses)
            mat = cis_transition_density_overlap_matrix(
                step_a.tden_vectors,
                step_b.tden_vectors,
                roots,
                step_a.mo_coefficients,
                step_b.mo_coefficients,
                s_cross,
            )
            cache[key] = mat
            return mat
        if args.engine not in {"nto-json", "nto-molden"}:
            raise MissingDataError("Unknown overlap engine.")
        if args.engine == "nto-json":
            s_cross = _tden_cross_overlap(step_a, step_b, same_overlaps, args, outdir, statuses)
            mat = nto_similarity_matrix(step_a.nto_vectors, step_b.nto_vectors, roots, s_cross)
            cache[key] = mat
            return mat
        source_a = _representative_molden_source(step_a, roots)
        source_b = _representative_molden_source(step_b, roots)
        molden_a = molden_cache.get(source_a)
        molden_b = molden_cache.get(source_b)
        mode = "pyscf-cross" if args.ao_overlap_mode == "auto" else args.ao_overlap_mode
        s_cross = get_cross_overlap(
            mode,
            molden_a,
            molden_b,
            same_overlap_a=same_overlaps.get(step_a.job.order),
            same_overlap_b=same_overlaps.get(step_b.job.order),
            require_order_check=args.require_orthonormality_check,
        )
        mat = nto_similarity_matrix(step_a.nto_vectors, step_b.nto_vectors, roots, s_cross)
        cache[key] = mat
        return mat

    return provider


def _tden_cross_overlap(step_a: StepData, step_b: StepData, same_overlaps, args, outdir: Path, statuses) -> np.ndarray:
    if step_a.job.order not in same_overlaps or step_b.job.order not in same_overlaps:
        raise MissingDataError(f"Same-geometry S-Matrix data are missing for {step_a.job.label}->{step_b.job.label}.")
    mode = "orca-auxiliary" if args.ao_overlap_mode == "auto" else args.ao_overlap_mode
    if mode == "orca-auxiliary":
        orca_exe = _resolve_orca_executable(args)
        json_exe = _resolve_executable(args.orca_2json)
        if orca_exe is None:
            raise MissingDataError("ORCA executable was not found. Pass --orca /path/to/orca for --ao-overlap-mode orca-auxiliary.")
        if json_exe is None:
            raise MissingDataError("orca_2json executable was not found. Pass --orca-2json /path/to/orca_2json.")
        return orca_auxiliary_cross_overlap(
            step_a.job,
            step_b.job,
            same_overlaps[step_a.job.order],
            same_overlaps[step_b.job.order],
            outdir,
            orca_exe,
            json_exe,
            statuses,
            force=args.force_json,
        )
    if mode == "json-cross":
        raise MissingDataError(
            "ao-overlap-mode=json-cross was requested, but no precomputed cross-geometry "
            "AO overlap JSON source is available. Use --ao-overlap-mode orca-auxiliary with ORCA 6.1.1."
        )
    if mode == "pyscf-cross":
        raise MissingDataError(
            "pyscf-cross is not enabled because PySCF basis/order validation against ORCA JSON "
            "has not been implemented. Use --ao-overlap-mode orca-auxiliary."
        )
    raise MissingDataError(f"Unknown AO overlap mode: {args.ao_overlap_mode}")


def _representative_molden_source(step: StepData, roots: Sequence[int]) -> Path:
    for root in roots:
        if root in step.nto_vectors:
            return step.nto_vectors[root].source_file
    raise MissingDataError(f"No NTO Molden source is available for step {step.job.label}.")


def _resolve_executable(value: str) -> Optional[str]:
    p = Path(value)
    if p.is_absolute() or "/" in value:
        return str(p) if p.exists() else None
    return shutil.which(value)


def _resolve_orca_executable(args) -> Optional[str]:
    if args.orca:
        return _resolve_executable(args.orca)
    json_path = _resolve_executable(args.orca_2json)
    if json_path:
        sibling = Path(json_path).with_name("orca")
        if sibling.exists():
            return str(sibling)
    return shutil.which("orca")


def _capture_help(exe: str, log_path: Path) -> None:
    chunks = []
    for flag in ("--help", "-h"):
        try:
            proc = subprocess.run([exe, flag], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            chunks.append(f"$ {exe} {flag}\nreturncode={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n")
            if proc.stdout or proc.stderr:
                break
        except Exception as exc:
            chunks.append(f"$ {exe} {flag}\nERROR: {exc}\n")
    log_path.write_text("\n".join(chunks))


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .models import (
    CISAmplitudeCheck,
    ExtractionStatus,
    NTOIndexMapping,
    NTOVisualizationRecord,
    OrthonormalityResult,
    SelfNTOReconstructionRecord,
    SelfNTOWeightRecord,
    StepData,
    TrackSegment,
    TrackingResult,
)


def ensure_output_dirs(outdir: Path) -> None:
    outdir = Path(outdir)
    for sub in ["json_cache", "molden_cache", "utility_logs", "diagnostics", "generated_inputs", "self_nto_cubes", "self_nto_images"]:
        (outdir / sub).mkdir(parents=True, exist_ok=True)


def write_outputs(
    outdir: Path,
    result: Optional[TrackingResult],
    steps: Sequence[StepData],
    extraction_statuses: Sequence[ExtractionStatus],
    orthonormality: Sequence[OrthonormalityResult],
    nto_index_mappings: Sequence[NTOIndexMapping] = (),
    cis_amplitude_checks: Sequence[CISAmplitudeCheck] = (),
    self_nto_weights: Sequence[SelfNTOWeightRecord] = (),
    self_nto_reconstruction: Sequence[SelfNTOReconstructionRecord] = (),
    self_nto_visualizations: Sequence[NTOVisualizationRecord] = (),
    write_html: bool = False,
    failure_message: Optional[str] = None,
    selected_roots: Optional[Sequence[int]] = None,
    engine: Optional[str] = None,
    nto_purity_threshold: float = 0.80,
    nto_purity_guard: bool = False,
    reverse_result: Optional[TrackingResult] = None,
    track_segments: Sequence[TrackSegment] = (),
) -> None:
    outdir = Path(outdir)
    ensure_output_dirs(outdir)
    nto_purity_rows = _nto_purity_rows(self_nto_weights, nto_purity_threshold)
    if result is not None:
        _write_diabatic_assignments(outdir / "diabatic_assignments.csv", result)
        _write_tracked_energies(outdir / "tracked_state_energies.csv", result)
        _write_similarity_matrices(outdir, result)
        _write_assignment_confidence(outdir / "assignment_confidence.csv", result)
        _write_ambiguous_regions(outdir / "ambiguous_regions.csv", result)
        _write_manifold_episodes(outdir / "manifold_episodes.csv", result)
        _write_bidirectional(outdir / "bidirectional_disagreements.csv", result)
        _write_track_segments(outdir / "track_segments.csv", track_segments)
        if reverse_result is not None:
            _write_diabatic_assignments(outdir / "reverse_diabatic_assignments.csv", reverse_result)
            _write_assignment_confidence(outdir / "reverse_assignment_confidence.csv", reverse_result)
            _write_ambiguous_regions(outdir / "reverse_ambiguous_regions.csv", reverse_result)
            _write_manifold_episodes(outdir / "reverse_manifold_episodes.csv", reverse_result)
    else:
        _write_empty_primary_outputs(outdir)
    _write_extraction_status(outdir / "diagnostics" / "extraction_status.csv", extraction_statuses)
    _write_orthonormality(outdir / "diagnostics" / "orthonormality_checks.csv", orthonormality)
    _write_nto_index_mapping(outdir / "diagnostics" / "nto_index_mapping.csv", nto_index_mappings)
    _write_cis_amplitude_checks(outdir / "diagnostics" / "cis_amplitude_checks.csv", cis_amplitude_checks)
    _write_self_nto_weights(outdir / "diagnostics" / "self_nto_weights.csv", self_nto_weights)
    _write_self_nto_reconstruction(outdir / "diagnostics" / "self_nto_reconstruction.csv", self_nto_reconstruction)
    _write_nto_purity(outdir / "diagnostics" / "nto_purity.csv", nto_purity_rows)
    _write_self_nto_visualizations(outdir / "diagnostics" / "self_nto_visualizations.csv", self_nto_visualizations)
    if write_html:
        _write_html(
            outdir / "state_tracking_report.html",
            result,
            steps,
            extraction_statuses,
            orthonormality,
            nto_index_mappings,
            cis_amplitude_checks,
            self_nto_weights,
            self_nto_reconstruction,
            nto_purity_rows,
            self_nto_visualizations,
            failure_message,
            selected_roots=selected_roots,
            requested_engine=engine,
            nto_purity_threshold=nto_purity_threshold,
            nto_purity_guard=nto_purity_guard,
            reverse_result=reverse_result,
            track_segments=track_segments,
        )


def _write_diabatic_assignments(path: Path, result: TrackingResult) -> None:
    fields = [
        "track_id",
        "step_order",
        "step_label",
        "scan_step",
        "adiabatic_root",
        "energy_eh",
        "excitation_ev",
        "s2",
        "multiplicity",
        "similarity_to_previous",
        "similarity_to_anchor",
        "assignment_margin",
        "confidence",
        "manifold_id",
        "engine",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in result.track_points:
            w.writerow({k: getattr(p, k) for k in fields})


def _write_tracked_energies(path: Path, result: TrackingResult) -> None:
    fields = [
        "track_id",
        "step_order",
        "step_label",
        "scan_step",
        "adiabatic_root",
        "excitation_ev",
        "abs_energy_eh",
        "energy_relative_to_local_reference_ev",
        "confidence",
        "manifold_id",
        "engine",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in result.track_points:
            w.writerow(
                {
                    "track_id": p.track_id,
                    "step_order": p.step_order,
                    "step_label": p.step_label,
                    "scan_step": p.scan_step,
                    "adiabatic_root": p.adiabatic_root,
                    "excitation_ev": p.excitation_ev,
                    "abs_energy_eh": p.energy_eh,
                    "energy_relative_to_local_reference_ev": p.excitation_ev,
                    "confidence": p.confidence,
                    "manifold_id": p.manifold_id,
                    "engine": p.engine,
                }
            )


def _write_similarity_matrices(outdir: Path, result: TrackingResult) -> None:
    csv_path = outdir / "adjacent_similarity_matrices.csv"
    roots = list(result.roots)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step_a", "step_b", "root_a", "root_b", "similarity", "engine"])
        w.writeheader()
        for key, mat in result.adjacent_matrices.items():
            step_a, step_b = _split_pair_key(key)
            for i, root_a in enumerate(roots):
                for j, root_b in enumerate(roots):
                    w.writerow(
                        {
                            "step_a": step_a,
                            "step_b": step_b,
                            "root_a": root_a,
                            "root_b": root_b,
                            "similarity": float(mat[i, j]),
                            "engine": result.engine,
                        }
                    )
    np.savez(outdir / "adjacent_similarity_matrices.npz", **{_safe_npz_key(k): v for k, v in result.adjacent_matrices.items()})


def _write_assignment_confidence(path: Path, result: TrackingResult) -> None:
    fields = [
        "step_a",
        "step_b",
        "track_id",
        "root_a",
        "root_b",
        "previous_similarity",
        "anchor_similarity",
        "best_similarity",
        "second_similarity",
        "margin",
        "ratio",
        "confidence",
        "reason",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in result.assignment_edges:
            row = {k: getattr(e, k) for k in fields if hasattr(e, k)}
            row["previous_similarity"] = e.similarity_to_previous
            row["anchor_similarity"] = e.similarity_to_anchor
            w.writerow(row)


def _write_ambiguous_regions(path: Path, result: TrackingResult) -> None:
    fields = ["region_id", "start_step", "end_step", "manifold_roots", "tracks_involved", "min_subspace_score", "reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in result.ambiguous_regions:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_manifold_episodes(path: Path, result: TrackingResult) -> None:
    fields = [
        "episode_id",
        "start_step",
        "end_step",
        "manifold_roots",
        "tracks_involved",
        "min_subspace_score",
        "region_count",
        "reason",
    ]
    rows = _coalesced_manifold_episodes(result)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_bidirectional(path: Path, result: TrackingResult) -> None:
    fields = ["step_pair", "root_a", "root_b", "reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in result.bidirectional_disagreements:
            w.writerow(row)


def _write_track_segments(path: Path, rows: Sequence[TrackSegment]) -> None:
    fields = [
        "segment_id",
        "direction",
        "support",
        "seed_track_id",
        "seed_step",
        "seed_root",
        "start_step",
        "end_step",
        "start_root",
        "end_root",
        "step_count",
        "step_sequence",
        "root_sequence",
        "min_adjacent_similarity",
        "min_purity",
        "mean_purity",
        "max_effective_rank",
        "status",
        "reason",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: getattr(row, k) for k in fields})


def _write_extraction_status(path: Path, rows: Sequence[ExtractionStatus]) -> None:
    fields = ["step_label", "root", "extractor", "source_file", "json_file", "molden_file", "status", "message"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "step_label": r.step_label,
                    "root": r.root,
                    "extractor": r.extractor,
                    "source_file": _path_or_none(r.source_file),
                    "json_file": _path_or_none(r.json_file),
                    "molden_file": _path_or_none(r.molden_file),
                    "status": r.status,
                    "message": r.message,
                }
            )


def _write_orthonormality(path: Path, rows: Sequence[OrthonormalityResult]) -> None:
    fields = [
        "step_label",
        "source_file",
        "source_type",
        "max_diag_error",
        "max_offdiag",
        "rms_offdiag",
        "passed",
        "message",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "step_label": r.step_label,
                    "source_file": str(r.source_file),
                    "source_type": r.source_type,
                    "max_diag_error": r.max_diag_error,
                    "max_offdiag": r.max_offdiag,
                    "rms_offdiag": r.rms_offdiag,
                    "passed": r.passed,
                    "message": r.message,
                }
            )


def _write_nto_index_mapping(path: Path, rows: Sequence[NTOIndexMapping]) -> None:
    fields = [
        "step_label",
        "root",
        "source_file",
        "printed_donor",
        "donor_spin",
        "parsed_donor_vector",
        "printed_acceptor",
        "acceptor_spin",
        "parsed_acceptor_vector",
        "weight",
        "validation_status",
        "donor_candidate_vectors",
        "acceptor_candidate_vectors",
        "mapping_note",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_cis_amplitude_checks(path: Path, rows: Sequence[CISAmplitudeCheck]) -> None:
    fields = [
        "step_label",
        "root",
        "source_file",
        "donor",
        "donor_spin",
        "acceptor",
        "acceptor_spin",
        "printed_coefficient",
        "binary_coefficient",
        "abs_error",
        "passed",
        "message",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_self_nto_weights(path: Path, rows: Sequence[SelfNTOWeightRecord]) -> None:
    fields = [
        "step_label",
        "root",
        "spin",
        "source_file",
        "pair_index",
        "singular_value",
        "weight",
        "spin_weight_fraction",
        "state_weight_fraction",
        "cumulative_spin_fraction",
        "selected",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: getattr(r, k) for k in fields}
            row["source_file"] = str(r.source_file)
            w.writerow(row)


def _write_self_nto_reconstruction(path: Path, rows: Sequence[SelfNTOReconstructionRecord]) -> None:
    fields = [
        "step_label",
        "root",
        "spin",
        "source_file",
        "n_occ",
        "n_virt",
        "rank",
        "matrix_norm",
        "reconstruction_error",
        "relative_error",
        "selected_pairs",
        "selected_spin_weight_fraction",
        "passed",
        "message",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: getattr(r, k) for k in fields}
            row["source_file"] = str(r.source_file)
            w.writerow(row)


def _nto_purity_rows(weights: Sequence[SelfNTOWeightRecord], threshold: float) -> List[dict]:
    grouped = {}
    for row in weights:
        key = (row.step_label, int(row.root))
        grouped.setdefault(key, []).append(row)
    out: List[dict] = []
    for (step_label, root), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        selected = [r for r in rows if r.selected]
        basis = selected if selected else list(rows)
        total = float(sum(max(0.0, r.weight) for r in basis))
        if total > 0.0 and basis:
            dominant = max(basis, key=lambda r: r.weight)
            dominant_fraction = float(max(0.0, dominant.weight) / total)
            effective_rank = float((total * total) / sum(max(0.0, r.weight) ** 2 for r in basis))
            selected_pairs = len(selected)
            significant_pairs = sum(1 for r in basis if total > 0 and r.weight / total >= 0.01)
            status = "pure" if dominant_fraction >= threshold else "mixed"
            source_file = dominant.source_file
            dominant_spin = dominant.spin
            dominant_pair = dominant.pair_index
            dominant_weight = dominant.weight
        else:
            dominant_fraction = None
            effective_rank = None
            selected_pairs = len(selected)
            significant_pairs = 0
            status = "missing"
            source_file = Path("")
            dominant_spin = ""
            dominant_pair = None
            dominant_weight = None
        out.append(
            {
                "step_label": step_label,
                "root": root,
                "source_file": str(source_file),
                "dominant_fraction": dominant_fraction,
                "dominant_spin": dominant_spin,
                "dominant_pair_index": dominant_pair,
                "dominant_weight": dominant_weight,
                "selected_weight_sum": total,
                "selected_pairs": selected_pairs,
                "significant_pairs_ge_1pct": significant_pairs,
                "effective_nto_rank": effective_rank,
                "purity_threshold": float(threshold),
                "purity_status": status,
            }
        )
    return out


def _write_nto_purity(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "step_label",
        "root",
        "source_file",
        "dominant_fraction",
        "dominant_spin",
        "dominant_pair_index",
        "dominant_weight",
        "selected_weight_sum",
        "selected_pairs",
        "significant_pairs_ge_1pct",
        "effective_nto_rank",
        "purity_threshold",
        "purity_status",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _write_self_nto_visualizations(path: Path, rows: Sequence[NTOVisualizationRecord]) -> None:
    fields = [
        "step_label",
        "root",
        "spin",
        "pair_index",
        "side",
        "weight",
        "cube_file",
        "png_file",
        "status",
        "message",
        "overlap_max_abs_error",
        "overlap_rms_error",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "step_label": r.step_label,
                    "root": r.root,
                    "spin": r.spin,
                    "pair_index": r.pair_index,
                    "side": r.side,
                    "weight": r.weight,
                    "cube_file": _path_or_none(r.cube_file),
                    "png_file": _path_or_none(r.png_file),
                    "status": r.status,
                    "message": r.message,
                    "overlap_max_abs_error": r.overlap_max_abs_error,
                    "overlap_rms_error": r.overlap_rms_error,
                }
            )


def _write_empty_primary_outputs(outdir: Path) -> None:
    empty_specs = {
        "diabatic_assignments.csv": [
            "track_id",
            "step_order",
            "step_label",
            "scan_step",
            "adiabatic_root",
            "energy_eh",
            "excitation_ev",
            "s2",
            "multiplicity",
            "similarity_to_previous",
            "similarity_to_anchor",
            "assignment_margin",
            "confidence",
            "manifold_id",
            "engine",
        ],
        "tracked_state_energies.csv": [
            "track_id",
            "step_order",
            "step_label",
            "scan_step",
            "adiabatic_root",
            "excitation_ev",
            "abs_energy_eh",
            "energy_relative_to_local_reference_ev",
            "confidence",
            "manifold_id",
            "engine",
        ],
        "adjacent_similarity_matrices.csv": ["step_a", "step_b", "root_a", "root_b", "similarity", "engine"],
        "assignment_confidence.csv": [
            "step_a",
            "step_b",
            "track_id",
            "root_a",
            "root_b",
            "previous_similarity",
            "anchor_similarity",
            "best_similarity",
            "second_similarity",
            "margin",
            "ratio",
            "confidence",
            "reason",
        ],
        "ambiguous_regions.csv": [
            "region_id",
            "start_step",
            "end_step",
            "manifold_roots",
            "tracks_involved",
            "min_subspace_score",
            "reason",
        ],
        "manifold_episodes.csv": [
            "episode_id",
            "start_step",
            "end_step",
            "manifold_roots",
            "tracks_involved",
            "min_subspace_score",
            "region_count",
            "reason",
        ],
        "bidirectional_disagreements.csv": ["step_pair", "root_a", "root_b", "reason"],
        "track_segments.csv": [
            "segment_id",
            "direction",
            "support",
            "seed_track_id",
            "seed_step",
            "seed_root",
            "start_step",
            "end_step",
            "start_root",
            "end_root",
            "step_count",
            "step_sequence",
            "root_sequence",
            "min_adjacent_similarity",
            "min_purity",
            "mean_purity",
            "max_effective_rank",
            "status",
            "reason",
        ],
        "reverse_diabatic_assignments.csv": [
            "track_id",
            "step_order",
            "step_label",
            "scan_step",
            "adiabatic_root",
            "energy_eh",
            "excitation_ev",
            "s2",
            "multiplicity",
            "similarity_to_previous",
            "similarity_to_anchor",
            "assignment_margin",
            "confidence",
            "manifold_id",
            "engine",
        ],
        "reverse_assignment_confidence.csv": [
            "step_a",
            "step_b",
            "track_id",
            "root_a",
            "root_b",
            "previous_similarity",
            "anchor_similarity",
            "best_similarity",
            "second_similarity",
            "margin",
            "ratio",
            "confidence",
            "reason",
        ],
        "reverse_ambiguous_regions.csv": [
            "region_id",
            "start_step",
            "end_step",
            "manifold_roots",
            "tracks_involved",
            "min_subspace_score",
            "reason",
        ],
        "reverse_manifold_episodes.csv": [
            "episode_id",
            "start_step",
            "end_step",
            "manifold_roots",
            "tracks_involved",
            "min_subspace_score",
            "region_count",
            "reason",
        ],
    }
    for name, fields in empty_specs.items():
        with (outdir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
    np.savez(outdir / "adjacent_similarity_matrices.npz")


def _write_html(
    path: Path,
    result: Optional[TrackingResult],
    steps: Sequence[StepData],
    extraction_statuses: Sequence[ExtractionStatus],
    orthonormality: Sequence[OrthonormalityResult],
    nto_index_mappings: Sequence[NTOIndexMapping],
    cis_amplitude_checks: Sequence[CISAmplitudeCheck],
    self_nto_weights: Sequence[SelfNTOWeightRecord],
    self_nto_reconstruction: Sequence[SelfNTOReconstructionRecord],
    nto_purity_rows: Sequence[dict],
    self_nto_visualizations: Sequence[NTOVisualizationRecord],
    failure_message: Optional[str],
    selected_roots: Optional[Sequence[int]] = None,
    requested_engine: Optional[str] = None,
    nto_purity_threshold: float = 0.80,
    nto_purity_guard: bool = False,
    reverse_result: Optional[TrackingResult] = None,
    track_segments: Sequence[TrackSegment] = (),
) -> None:
    reliable = low = ambiguous = failed = disagreements = manifolds = manifold_episodes = 0
    if result is not None:
        reliable = sum(1 for e in result.assignment_edges if e.confidence == "reliable")
        low = sum(1 for e in result.assignment_edges if e.confidence == "low_confidence")
        ambiguous = sum(1 for e in result.assignment_edges if e.confidence == "ambiguous")
        failed = sum(1 for e in result.assignment_edges if e.confidence == "failed")
        disagreements = len(result.bidirectional_disagreements)
        manifolds = len(result.ambiguous_regions)
        manifold_episodes = len(_coalesced_manifold_episodes(result))
    engine = result.engine if result is not None else (requested_engine or "not completed")
    roots = list(result.roots) if result is not None else list(selected_roots or [])
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>ORCA Excited-State Tracking</title>",
        _css(),
        "</head><body>",
        "<h1>ORCA Excited-State Tracking</h1>",
    ]
    if failure_message:
        parts.append(f"<section class='error'><h2>Run Stopped</h2><p>{html.escape(failure_message)}</p></section>")
    parts.append(
        "<section><h2>Executive Summary</h2>"
        f"<table><tbody>"
        f"<tr><th>Geometries</th><td>{len(steps)}</td></tr>"
        f"<tr><th>Roots</th><td>{html.escape(', '.join(map(str, roots)))}</td></tr>"
        f"<tr><th>Engine</th><td>{html.escape(engine)}</td></tr>"
        f"<tr><th>Tracks</th><td>{len(roots)}</td></tr>"
        f"<tr><th>Reliable assignments</th><td>{reliable}</td></tr>"
        f"<tr><th>Low confidence</th><td>{low}</td></tr>"
        f"<tr><th>Ambiguous assignments</th><td>{ambiguous}</td></tr>"
        f"<tr><th>Failed assignments</th><td>{failed}</td></tr>"
        f"<tr><th>Ambiguous manifold pair events</th><td>{manifolds}</td></tr>"
        f"<tr><th>Ambiguous manifold episodes</th><td>{manifold_episodes}</td></tr>"
        f"<tr><th>NTO purity guard</th><td>{'enabled' if nto_purity_guard else 'disabled'} "
        f"(threshold {nto_purity_threshold:.3f})</td></tr>"
        f"<tr><th>Reverse full tracking</th><td>{'enabled' if reverse_result is not None else 'not run'}</td></tr>"
        f"<tr><th>Stable local segments</th><td>{len(track_segments)}</td></tr>"
        f"<tr><th>Forward/backward disagreements</th><td>{disagreements}</td></tr>"
        "</tbody></table></section>"
    )
    parts.append("<section><h2>Energy Plots</h2>")
    parts.append("<h3>Absolute Adiabatic State Energies</h3>")
    parts.append(_energy_svg_steps(steps, roots, value="absolute"))
    if result is not None:
        parts.append("<h3>Absolute Diabatic Tracked Energies</h3>")
        parts.append(_energy_svg_tracks(result, value="absolute"))
        parts.append("<p><em>Dashed grey track segments mark low-confidence, ambiguous, failed, or manifold-tagged assignments.</em></p>")
    parts.append("<h3>Adiabatic Excitation Energies</h3>")
    parts.append(_energy_svg_steps(steps, roots, value="excitation"))
    if result is not None:
        parts.append("<h3>Diabatic Tracked Excitation Energies</h3>")
        parts.append(_energy_svg_tracks(result, value="excitation"))
        parts.append("<p><em>Dashed grey track segments mark low-confidence, ambiguous, failed, or manifold-tagged assignments.</em></p>")
    parts.append("</section>")
    if result is not None:
        parts.append("<section><h2>Similarity Heatmaps</h2>")
        for key, mat in result.adjacent_matrices.items():
            parts.append(f"<h3>{html.escape(key)}</h3>")
            parts.append(_heatmap_table(mat, roots, result.assignment_edges, key))
        parts.append("</section>")
        parts.append("<section><h2>Track Tables</h2>")
        parts.append(_track_tables(result, nto_purity_rows))
        parts.append("</section>")
        if track_segments:
            parts.append("<section><h2>Track Segments</h2>")
            parts.append(_track_segment_table(track_segments))
            parts.append("</section>")
        if self_nto_visualizations:
            parts.append("<section><h2>Self-Derived NTO Visualizations</h2>")
            parts.append(_self_nto_visualization_gallery(self_nto_visualizations, path.parent))
            parts.append("</section>")
        parts.append("<section><h2>Ambiguity</h2>")
        parts.append(_ambiguity_rescue_summary(result, nto_purity_rows, nto_purity_threshold))
        if result.ambiguous_regions:
            parts.append(_ambiguous_table(result))
            parts.append(_manifold_episode_table(result))
        else:
            parts.append("<p>No subspace manifolds were detected with the selected thresholds.</p>")
        low_edges = [e for e in result.assignment_edges if e.confidence in {"low_confidence", "ambiguous", "failed"}]
        if low_edges:
            parts.append(
                "<p>Low-similarity or forward/backward-disagreement regions should be treated as provisional; "
                "insert intermediate scan or optimization points where best similarity falls below threshold.</p>"
            )
        parts.append("</section>")
    parts.append("<section><h2>Diagnostics</h2>")
    parts.append("<h3>Extraction Status</h3>")
    parts.append(_extraction_table(extraction_statuses))
    parts.append("<h3>Orthonormality Checks</h3>")
    parts.append(_orth_table(orthonormality))
    parts.append("<h3>NTO Index Mapping</h3>")
    parts.append(_mapping_table(nto_index_mappings))
    parts.append("<h3>Self-Derived NTOs From CIS/TDA Amplitudes</h3>")
    parts.append(_self_nto_table(self_nto_weights, self_nto_reconstruction, engine))
    parts.append("<h3>NTO Purity Diagnostics</h3>")
    parts.append(_nto_purity_table(nto_purity_rows))
    parts.append("<h3>Self-Derived NTO Cube/PNG Status</h3>")
    parts.append(_self_nto_visualization_status_table(self_nto_visualizations))
    parts.append("<h3>CIS/TDA Amplitude Checks</h3>")
    parts.append(_cis_check_table(cis_amplitude_checks))
    parts.append("</section></body></html>")
    path.write_text("\n".join(parts))


def _energy_svg_steps(steps: Sequence[StepData], roots: Sequence[int], value: str = "absolute") -> str:
    series = []
    x_labels = {}
    for i, step in enumerate(steps):
        x = step.job.scan_step if step.job.scan_step is not None else i
        x_labels[x] = step.job.label
        
    for root in roots:
        pts = []
        for i, step in enumerate(steps):
            x = step.job.scan_step if step.job.scan_step is not None else i
            st = step.states.get(root)
            if st is None:
                y = None
            elif value == "excitation":
                y = st.exc_ev
            else:
                y = st.abs_energy_eh
            pts.append((x, y))
        series.append((f"root {root}", pts))
    ylabel = "Excitation energy (eV)" if value == "excitation" else "Absolute energy (Eh)"
    return _line_svg(series, ylabel=ylabel, x_labels=x_labels)


def _energy_svg_tracks(result: TrackingResult, value: str = "absolute") -> str:
    series = []
    x_labels = {}
    for point in result.track_points:
        x = point.scan_step if point.scan_step is not None else point.step_order
        x_labels[x] = point.step_label
        
    for track in sorted({p.track_id for p in result.track_points}):
        if value == "excitation":
            pts = [(p.scan_step if p.scan_step is not None else p.step_order, p.excitation_ev, p.confidence, p.manifold_id) for p in result.track_points if p.track_id == track]
        else:
            pts = [(p.scan_step if p.scan_step is not None else p.step_order, p.energy_eh, p.confidence, p.manifold_id) for p in result.track_points if p.track_id == track]
        series.append((f"track {track}", pts))
    ylabel = "Excitation energy (eV)" if value == "excitation" else "Absolute energy (Eh)"
    return _line_svg(series, ylabel=ylabel, x_labels=x_labels)


def _line_svg(series, ylabel: str = "Energy", x_labels: Optional[dict] = None) -> str:
    vals = [_point_y(p) for _, pts in series for p in pts if _point_y(p) is not None]
    xs = [_point_x(p) for _, pts in series for p in pts if _point_y(p) is not None]
    if not vals or not xs:
        return "<p><em>No energy values available.</em></p>"
    w, h = 860, 540
    left, right, top, bottom = 56, 92, 34, 58
    plot_w = w - left - right
    plot_h = h - top - bottom
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(vals), max(vals)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    ypad = 0.04 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y):
        return top + (ymax - y) / (ymax - ymin) * plot_h

    colors = ["#1b6ca8", "#b23a48", "#26734d", "#8a5a00", "#6f42c1", "#008c8c", "#a33f00"]
    out = [f"<svg class='plot' viewBox='0 0 {w} {h}' role='img'>"]
    
    if x_labels:
        tick_xs = sorted(x_labels.keys())
    else:
        tick_xs = [xmin + (xmax - xmin) * f for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
        
    for x in tick_xs:
        px = sx(x)
        out.append(f"<line class='xgrid' x1='{px:.1f}' y1='{top}' x2='{px:.1f}' y2='{h-bottom}'/>")
        label = str(x_labels.get(x, f"{x:.4g}")) if x_labels else f"{x:.4g}"
        out.append(
            f"<text class='xtick' x='{px:.1f}' y='{h - bottom + 17}' "
            f"transform='rotate(-45 {px:.1f} {h - bottom + 17})'>{html.escape(label)}</text>"
        )
    for frac in (0.25, 0.5, 0.75):
        py = top + frac * plot_h
        out.append(f"<line class='ygrid' x1='{left}' y1='{py:.1f}' x2='{w-right}' y2='{py:.1f}'/>")
    out.append(f"<line class='axis' x1='{left}' y1='{h-bottom}' x2='{w-right}' y2='{h-bottom}'/>")
    out.append(f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{h-bottom}'/>")
    out.append(f"<text class='ylabel' x='{left}' y='{top - 12}'>{html.escape(ylabel)}</text>")
    out.append(f"<text class='xlabel' x='{w - right - 48}' y='{h - 10}'>scan step</text>")
    out.append(f"<text class='ytick' x='{left + 4}' y='{top + 12}'>{ymax:.6g}</text>")
    out.append(f"<text class='ytick' x='{left + 4}' y='{h - bottom - 5}'>{ymin:.6g}</text>")
    for idx, (name, pts) in enumerate(series):
        usable = [
            (sx(_point_x(p)), sy(_point_y(p)), _point_confidence(p), _point_manifold(p))
            for p in pts
            if _point_y(p) is not None
        ]
        if not usable:
            continue
        color = colors[idx % len(colors)]
        for seg_idx in range(1, len(usable)):
            x0, y0, _, _ = usable[seg_idx - 1]
            x1, y1, conf, manifold = usable[seg_idx]
            uncertain = conf in {"low_confidence", "ambiguous", "failed"} or bool(manifold)
            stroke = "#8a8f98" if uncertain else color
            dash = " stroke-dasharray='4 4'" if uncertain else ""
            opacity = "0.65" if uncertain else "1"
            out.append(
                f"<line x1='{x0:.1f}' y1='{y0:.1f}' x2='{x1:.1f}' y2='{y1:.1f}' "
                f"stroke='{stroke}' stroke-width='1.8' opacity='{opacity}'{dash}/>"
            )
        for x, y, conf, manifold in usable:
            uncertain = conf in {"low_confidence", "ambiguous", "failed"} or bool(manifold)
            fill = "#ffffff" if uncertain else color
            stroke = "#8a8f98" if uncertain else color
            out.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.4' fill='{fill}' stroke='{stroke}' stroke-width='1.1'/>")
        lx, ly, _, _ = usable[-1]
        out.append(f"<text x='{lx + 4:.1f}' y='{ly:.1f}' fill='{color}'>{html.escape(name)}</text>")
    out.append("</svg>")
    return "\n".join(out)


def _point_x(point) -> float:
    return point[0]


def _point_y(point):
    return point[1]


def _point_confidence(point) -> str:
    return point[2] if len(point) > 2 else "reliable"


def _point_manifold(point):
    return point[3] if len(point) > 3 else None


def _heatmap_table(mat: np.ndarray, roots: Sequence[int], edges, key: str) -> str:
    assigned = {(e.root_a, e.root_b): e.confidence for e in edges if f"{e.step_a}->{e.step_b}" == key}
    rows = ["<table class='heatmap'><thead><tr><th></th>" + "".join(f"<th>{r}</th>" for r in roots) + "</tr></thead><tbody>"]
    maxv = max(1e-12, float(np.nanmax(mat))) if mat.size else 1.0
    for i, ra in enumerate(roots):
        cells = [f"<th>{ra}</th>"]
        for j, rb in enumerate(roots):
            val = float(mat[i, j])
            shade = int(255 - 160 * max(0.0, min(1.0, val / maxv)))
            cls = " assigned" if (ra, rb) in assigned else ""
            label = "*" if (ra, rb) in assigned else ""
            cells.append(f"<td class='{cls}' style='background: rgb({shade},{shade},255)'>{val:.3f}{label}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _track_tables(result: TrackingResult, purity_rows: Sequence[dict] = ()) -> str:
    out = []
    purity = _purity_lookup(purity_rows)
    for track in sorted({p.track_id for p in result.track_points}):
        pts = sorted([p for p in result.track_points if p.track_id == track], key=lambda p: p.step_order)
        seq = " -> ".join(f"{p.step_label} root {p.adiabatic_root}" for p in pts)
        out.append(f"<h3>Track {track}</h3><p>{html.escape(seq)}</p>")
        out.append(
            "<table><thead><tr><th>Step</th><th>Root</th><th>Exc eV</th><th>Prev sim</th>"
            "<th>Reference score</th><th>NTO purity</th><th>NTO rank</th><th>Confidence</th><th>Manifold</th></tr></thead><tbody>"
        )
        for p in pts:
            prow = purity.get((p.step_label, p.adiabatic_root), {})
            out.append(
                f"<tr><td>{html.escape(p.step_label)}</td><td>{p.adiabatic_root}</td>"
                f"<td>{_fmt(p.excitation_ev)}</td><td>{_fmt(p.similarity_to_previous)}</td>"
                f"<td>{_fmt(p.similarity_to_anchor)}</td>"
                f"<td>{_fmt(prow.get('dominant_fraction'))}</td><td>{_fmt(prow.get('effective_nto_rank'))}</td>"
                f"<td>{html.escape(p.confidence)}</td><td>{html.escape(p.manifold_id or '')}</td></tr>"
            )
        out.append("</tbody></table>")
    return "\n".join(out)


def _track_segment_table(rows: Sequence[TrackSegment]) -> str:
    if not rows:
        return "<p><em>No stable local track segments were detected with the selected thresholds.</em></p>"
    two = sum(1 for r in rows if r.support == "two_sided")
    fwd = sum(1 for r in rows if r.support == "forward_only")
    rev = sum(1 for r in rows if r.support == "reverse_only")
    shown = list(rows[:300])
    out = [
        "<p>Segments are locally continuous pure branches detected from forward and/or reverse tracking. "
        "They are not automatically stitched across unresolved mixed regions. A reverse-only segment is "
        "useful evidence for a clean right-anchored branch whose left identity was not established.</p>",
        "<table><tbody>"
        f"<tr><th>Two-sided segments</th><td>{two}</td></tr>"
        f"<tr><th>Forward-only segments</th><td>{fwd}</td></tr>"
        f"<tr><th>Reverse-only segments</th><td>{rev}</td></tr>"
        "</tbody></table>",
        "<table><thead><tr><th>Segment</th><th>Direction</th><th>Support</th><th>Seed</th>"
        "<th>Steps</th><th>Roots</th><th>N</th><th>min adj S</th><th>min purity</th>"
        "<th>mean purity</th><th>max NTO rank</th><th>Reason</th></tr></thead><tbody>",
    ]
    for r in shown:
        out.append(
            f"<tr><td>{html.escape(r.segment_id)}</td><td>{html.escape(r.direction)}</td>"
            f"<td>{html.escape(r.support)}</td><td>{html.escape(r.seed_step)} r{r.seed_root} "
            f"(track {r.seed_track_id})</td><td>{html.escape(r.start_step)} to {html.escape(r.end_step)}</td>"
            f"<td>{html.escape(r.root_sequence)}</td><td>{r.step_count}</td>"
            f"<td>{_fmt(r.min_adjacent_similarity)}</td><td>{_fmt(r.min_purity)}</td>"
            f"<td>{_fmt(r.mean_purity)}</td><td>{_fmt(r.max_effective_rank)}</td>"
            f"<td>{html.escape(r.reason)}</td></tr>"
        )
    out.append("</tbody></table>")
    if len(rows) > len(shown):
        out.append(f"<p><em>Showing first {len(shown)} of {len(rows)} segments; see track_segments.csv.</em></p>")
    return "\n".join(out)


def _ambiguous_table(result: TrackingResult) -> str:
    out = ["<table><thead><tr><th>Region</th><th>Steps</th><th>Roots</th><th>Tracks</th><th>Subspace score</th><th>Reason</th></tr></thead><tbody>"]
    for r in result.ambiguous_regions:
        out.append(
            f"<tr><td>{html.escape(r.region_id)}</td><td>{html.escape(r.start_step)} to {html.escape(r.end_step)}</td>"
            f"<td>{html.escape(r.manifold_roots)}</td><td>{html.escape(r.tracks_involved)}</td>"
            f"<td>{r.min_subspace_score:.3f}</td><td>{html.escape(r.reason)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _manifold_episode_table(result: TrackingResult) -> str:
    rows = _coalesced_manifold_episodes(result)
    if not rows:
        return ""
    out = [
        "<h3>Manifold Episodes</h3>",
        "<p>Adjacent manifold detections are consolidated here when they involve the same roots and tracks over contiguous scan steps. "
        "Inside these episodes, individual root labels should be read as provisional; the manifold is the tracked object.</p>",
        "<table><thead><tr><th>Episode</th><th>Steps</th><th>Roots</th><th>Tracks</th>"
        "<th>Min subspace score</th><th>Pair events</th><th>Reason</th></tr></thead><tbody>",
    ]
    for row in rows:
        out.append(
            f"<tr><td>{html.escape(str(row['episode_id']))}</td>"
            f"<td>{html.escape(str(row['start_step']))} to {html.escape(str(row['end_step']))}</td>"
            f"<td>{html.escape(str(row['manifold_roots']))}</td>"
            f"<td>{html.escape(str(row['tracks_involved']))}</td>"
            f"<td>{_fmt(row['min_subspace_score'])}</td><td>{row['region_count']}</td>"
            f"<td>{html.escape(str(row['reason']))}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _coalesced_manifold_episodes(result: TrackingResult) -> List[dict]:
    if not result.ambiguous_regions:
        return []
    order_by_step = {p.step_label: p.step_order for p in result.track_points}

    def sort_key(region):
        return (order_by_step.get(region.start_step, 10**9), order_by_step.get(region.end_step, 10**9), region.manifold_roots)

    episodes: List[dict] = []
    current = None
    for region in sorted(result.ambiguous_regions, key=sort_key):
        start_order = order_by_step.get(region.start_step)
        end_order = order_by_step.get(region.end_step)
        if start_order is None or end_order is None:
            start_order = end_order = 10**9
        same_object = (
            current is not None
            and current["manifold_roots"] == region.manifold_roots
            and current["tracks_involved"] == region.tracks_involved
            and start_order <= current["_end_order"] + 1
        )
        if not same_object:
            if current is not None:
                episodes.append(current)
            current = {
                "episode_id": f"E{len(episodes) + 1:03d}",
                "start_step": region.start_step,
                "end_step": region.end_step,
                "manifold_roots": region.manifold_roots,
                "tracks_involved": region.tracks_involved,
                "min_subspace_score": float(region.min_subspace_score),
                "region_count": 1,
                "reason": region.reason,
                "_end_order": end_order,
            }
            continue
        current["end_step"] = region.end_step
        current["_end_order"] = max(current["_end_order"], end_order)
        current["min_subspace_score"] = min(float(current["min_subspace_score"]), float(region.min_subspace_score))
        current["region_count"] = int(current["region_count"]) + 1
        if region.reason not in str(current["reason"]):
            current["reason"] = f"{current['reason']} | {region.reason}"
    if current is not None:
        episodes.append(current)
    for row in episodes:
        row.pop("_end_order", None)
    return episodes


def _ambiguity_rescue_summary(result: TrackingResult, purity_rows: Sequence[dict], threshold: float) -> str:
    purity = _purity_lookup(purity_rows)
    mixed_edges = []
    rescue_edges = []
    manifold_edges = []
    for edge in result.assignment_edges:
        reason = edge.reason or ""
        if "NTO-mixed" in reason:
            mixed_edges.append(edge)
        if "reconnected to the last stable" in reason:
            rescue_edges.append(edge)
        if edge.manifold_id:
            manifold_edges.append(edge)
    out = [
        "<h3>Purity And Rescue Summary</h3>",
        "<p>Mixed adiabatic roots are treated as provisional label carriers. When ambiguity rescue is active, "
        "the next step is compared to the last stable unmixed anchor rather than to the mixed root.</p>",
        "<table><tbody>"
        f"<tr><th>NTO purity threshold</th><td>{threshold:.3f}</td></tr>"
        f"<tr><th>Mixed/purity-guarded assignments</th><td>{len(mixed_edges)}</td></tr>"
        f"<tr><th>Anchor reconnections</th><td>{len(rescue_edges)}</td></tr>"
        f"<tr><th>Assignments inside manifolds</th><td>{len(manifold_edges)}</td></tr>"
        "</tbody></table>",
    ]
    if mixed_edges:
        out.append("<h3>Mixed Roots</h3>")
        out.append(
            "<table><thead><tr><th>Track</th><th>Step</th><th>Root</th><th>Dominant NTO fraction</th>"
            "<th>Effective rank</th><th>Reason</th></tr></thead><tbody>"
        )
        for edge in mixed_edges[:200]:
            prow = purity.get((edge.step_b, edge.root_b), {})
            out.append(
                f"<tr><td>{edge.track_id}</td><td>{html.escape(edge.step_b)}</td><td>{edge.root_b}</td>"
                f"<td>{_fmt(prow.get('dominant_fraction'))}</td><td>{_fmt(prow.get('effective_nto_rank'))}</td>"
                f"<td>{html.escape(edge.reason)}</td></tr>"
            )
        out.append("</tbody></table>")
    if rescue_edges:
        out.append("<h3>Anchor Reconnections</h3>")
        out.append(
            "<table><thead><tr><th>Track</th><th>Step pair</th><th>Previous root</th><th>Reconnected root</th>"
            "<th>Anchor similarity</th><th>Reason</th></tr></thead><tbody>"
        )
        for edge in rescue_edges[:200]:
            out.append(
                f"<tr><td>{edge.track_id}</td><td>{html.escape(edge.step_a)} to {html.escape(edge.step_b)}</td>"
                f"<td>{edge.root_a}</td><td>{edge.root_b}</td><td>{_fmt(edge.similarity_to_anchor)}</td>"
                f"<td>{html.escape(edge.reason)}</td></tr>"
            )
        out.append("</tbody></table>")
    return "\n".join(out)


def _extraction_table(rows: Sequence[ExtractionStatus]) -> str:
    out = ["<table><thead><tr><th>Step</th><th>Root</th><th>Extractor</th><th>Status</th><th>Message</th></tr></thead><tbody>"]
    for r in rows:
        out.append(
            f"<tr><td>{html.escape(r.step_label)}</td><td>{'' if r.root is None else r.root}</td>"
            f"<td>{html.escape(r.extractor)}</td><td>{html.escape(r.status)}</td><td>{html.escape(r.message)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _orth_table(rows: Sequence[OrthonormalityResult]) -> str:
    out = ["<table><thead><tr><th>Step</th><th>Source</th><th>diag err</th><th>max offdiag</th><th>RMS offdiag</th><th>Passed</th><th>Message</th></tr></thead><tbody>"]
    for r in rows:
        out.append(
            f"<tr><td>{html.escape(r.step_label)}</td><td>{html.escape(Path(r.source_file).name)}</td>"
            f"<td>{_fmt(r.max_diag_error)}</td><td>{_fmt(r.max_offdiag)}</td><td>{_fmt(r.rms_offdiag)}</td>"
            f"<td>{r.passed}</td><td>{html.escape(r.message)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _mapping_table(rows: Sequence[NTOIndexMapping]) -> str:
    if not rows:
        return "<p><em>No NTO index mapping records were produced.</em></p>"
    out = [
        "<table><thead><tr><th>Step</th><th>Root</th><th>Donor index</th><th>Donor vector</th>"
        "<th>Donor candidates</th><th>Acceptor index</th><th>Acceptor vector</th><th>Acceptor candidates</th>"
        "<th>n</th><th>Validation</th><th>Note</th></tr></thead><tbody>"
    ]
    for r in rows:
        out.append(
            f"<tr><td>{html.escape(r.step_label)}</td><td>{r.root}</td>"
            f"<td>{r.printed_donor}{html.escape(r.donor_spin)}</td><td>{'' if r.parsed_donor_vector is None else r.parsed_donor_vector}</td>"
            f"<td>{html.escape(r.donor_candidate_vectors)}</td>"
            f"<td>{r.printed_acceptor}{html.escape(r.acceptor_spin)}</td><td>{'' if r.parsed_acceptor_vector is None else r.parsed_acceptor_vector}</td>"
            f"<td>{html.escape(r.acceptor_candidate_vectors)}</td>"
            f"<td>{r.weight:.6g}</td><td>{html.escape(r.validation_status)}</td><td>{html.escape(r.mapping_note)}</td></tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def _self_nto_table(
    weights: Sequence[SelfNTOWeightRecord],
    reconstruction: Sequence[SelfNTOReconstructionRecord],
    engine: str,
) -> str:
    if not weights and not reconstruction:
        return "<p><em>No self-derived NTO diagnostics were produced.</em></p>"
    out = [
        "<p>For <code>tden-json</code>, these NTOs are obtained by SVD of the validated ORCA "
        "<code>job.cis</code> CIS/TDA amplitude matrices. They are diagnostic/interpretable NTOs; "
        "the assignment metric remains the full transition-density overlap and does not use ORCA "
        "<code>.nto</code> JSON orbital indices.</p>"
    ]
    if reconstruction:
        failed = [r for r in reconstruction if not r.passed]
        max_rel = max((r.relative_error for r in reconstruction), default=0.0)
        out.append(
            "<table><tbody>"
            f"<tr><th>Engine</th><td>{html.escape(engine)}</td></tr>"
            f"<tr><th>Spin blocks decomposed</th><td>{len(reconstruction)}</td></tr>"
            f"<tr><th>Failed reconstructions</th><td>{len(failed)}</td></tr>"
            f"<tr><th>Max relative reconstruction error</th><td>{max_rel:.3e}</td></tr>"
            "</tbody></table>"
        )
    selected = [r for r in weights if r.selected]
    if selected:
        shown = selected[:200]
        out.append(
            "<table><thead><tr><th>Step</th><th>Root</th><th>Spin</th><th>Pair</th>"
            "<th>sigma</th><th>n</th><th>spin frac</th><th>state frac</th><th>cum spin frac</th></tr></thead><tbody>"
        )
        for r in shown:
            out.append(
                f"<tr><td>{html.escape(r.step_label)}</td><td>{r.root}</td><td>{html.escape(r.spin)}</td>"
                f"<td>{r.pair_index}</td><td>{_fmt(r.singular_value)}</td><td>{_fmt(r.weight)}</td>"
                f"<td>{_fmt(r.spin_weight_fraction)}</td><td>{_fmt(r.state_weight_fraction)}</td>"
                f"<td>{_fmt(r.cumulative_spin_fraction)}</td></tr>"
            )
        out.append("</tbody></table>")
        if len(selected) > len(shown):
            out.append(f"<p><em>Showing first {len(shown)} of {len(selected)} selected self-derived NTO pairs; see diagnostics/self_nto_weights.csv.</em></p>")
    else:
        out.append("<p><em>No self-derived NTO pairs passed the selected weight/cumulative thresholds.</em></p>")
    return "\n".join(out)


def _nto_purity_table(rows: Sequence[dict]) -> str:
    if not rows:
        return "<p><em>No NTO purity diagnostics were produced.</em></p>"
    shown = list(rows[:300])
    mixed = [r for r in rows if r.get("purity_status") == "mixed"]
    out = [
        "<p>NTO purity is computed from the selected self-derived NTO pairs for each root. "
        "A low dominant fraction means the adiabatic root is a poor single-state label carrier, "
        "even if the mixed state is physically meaningful.</p>",
        "<table><tbody>"
        f"<tr><th>Root/step records</th><td>{len(rows)}</td></tr>"
        f"<tr><th>Mixed records</th><td>{len(mixed)}</td></tr>"
        "</tbody></table>",
        "<table><thead><tr><th>Step</th><th>Root</th><th>Status</th><th>Dominant frac</th>"
        "<th>Dominant pair</th><th>Dominant spin</th><th>Selected pairs</th>"
        "<th>Effective rank</th></tr></thead><tbody>",
    ]
    for r in shown:
        out.append(
            f"<tr><td>{html.escape(str(r.get('step_label', '')))}</td><td>{r.get('root', '')}</td>"
            f"<td>{html.escape(str(r.get('purity_status', '')))}</td>"
            f"<td>{_fmt(r.get('dominant_fraction'))}</td>"
            f"<td>{'' if r.get('dominant_pair_index') is None else r.get('dominant_pair_index')}</td>"
            f"<td>{html.escape(str(r.get('dominant_spin', '')))}</td>"
            f"<td>{r.get('selected_pairs', '')}</td><td>{_fmt(r.get('effective_nto_rank'))}</td></tr>"
        )
    out.append("</tbody></table>")
    if len(rows) > len(shown):
        out.append(f"<p><em>Showing first {len(shown)} of {len(rows)} purity records; see diagnostics/nto_purity.csv.</em></p>")
    return "\n".join(out)


def _purity_lookup(rows: Sequence[dict]) -> dict:
    out = {}
    for row in rows:
        step = row.get("step_label")
        root = row.get("root")
        if step in (None, "") or root in (None, ""):
            continue
        try:
            key = (str(step), int(root))
        except Exception:
            continue
        out[key] = row
    return out


def _self_nto_visualization_gallery(rows: Sequence[NTOVisualizationRecord], report_dir: Path) -> str:
    rendered = [r for r in rows if r.png_file and Path(r.png_file).exists() and Path(r.png_file).stat().st_size > 0]
    if not rendered:
        return (
            "<p><em>No self-derived NTO PNGs were rendered. See the diagnostics table below for the cube-generation "
            "or Jmol failure reason.</em></p>"
        )
    out = [
        "<p>These images are rendered from Gaussian cube files generated directly from the self-derived NTO vectors "
        "obtained by SVD of the validated <code>job.cis</code> CIS/TDA amplitude matrices. ORCA <code>.nto</code> "
        "files are not used in this visualization path. The red/blue phase is arbitrary for each NTO.</p>"
    ]
    groups = {}
    for r in rendered:
        key = (r.step_label, r.root, r.spin, r.pair_index)
        groups.setdefault(key, {})[r.side] = r
    for key in sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else int(k[1]), k[2], -1 if k[3] is None else int(k[3]))):
        step, root, spin, pair = key
        sides = groups[key]
        weight = next((r.weight for r in sides.values() if r.weight is not None), None)
        out.append(
            "<div class='nto-pair'>"
            f"<h3>{html.escape(step)} root {'' if root is None else root} "
            f"{html.escape(spin)} pair {'' if pair is None else pair}; n={_fmt(weight)}</h3>"
            "<div class='figgrid'>"
        )
        for side in ("donor", "acceptor"):
            rec = sides.get(side)
            if rec is None or rec.png_file is None:
                out.append(f"<figure><div class='missing-img'>missing {html.escape(side)}</div><figcaption>{html.escape(side)}</figcaption></figure>")
                continue
            rel = _rel_path(Path(rec.png_file), report_dir)
            cube_rel = _rel_path(Path(rec.cube_file), report_dir) if rec.cube_file else ""
            caption = f"{side}; cube {cube_rel}" if cube_rel else side
            out.append(
                f"<figure><img src='{html.escape(rel)}' alt='{html.escape(step)} root {root} {side} NTO'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
        out.append("</div></div>")
    return "\n".join(out)


def _self_nto_visualization_status_table(rows: Sequence[NTOVisualizationRecord]) -> str:
    if not rows:
        return "<p><em>No self-derived NTO cube/PNG visualization was requested.</em></p>"
    shown = list(rows[:300])
    ok = sum(1 for r in rows if r.status == "ok")
    failed = sum(1 for r in rows if "failed" in r.status)
    out = [
        "<table><tbody>"
        f"<tr><th>Visualization records</th><td>{len(rows)}</td></tr>"
        f"<tr><th>Rendered OK</th><td>{ok}</td></tr>"
        f"<tr><th>Failed or incomplete</th><td>{failed}</td></tr>"
        "</tbody></table>",
        "<table><thead><tr><th>Step</th><th>Root</th><th>Spin</th><th>Pair</th><th>Side</th>"
        "<th>n</th><th>Status</th><th>max |dS|</th><th>RMS dS</th><th>Message</th></tr></thead><tbody>",
    ]
    for r in shown:
        out.append(
            f"<tr><td>{html.escape(r.step_label)}</td><td>{'' if r.root is None else r.root}</td>"
            f"<td>{html.escape(r.spin)}</td><td>{'' if r.pair_index is None else r.pair_index}</td>"
            f"<td>{html.escape(r.side)}</td><td>{_fmt(r.weight)}</td><td>{html.escape(r.status)}</td>"
            f"<td>{_fmt(r.overlap_max_abs_error)}</td><td>{_fmt(r.overlap_rms_error)}</td>"
            f"<td>{html.escape(r.message)}</td></tr>"
        )
    out.append("</tbody></table>")
    if len(rows) > len(shown):
        out.append(f"<p><em>Showing first {len(shown)} of {len(rows)} visualization records; see diagnostics/self_nto_visualizations.csv.</em></p>")
    return "\n".join(out)


def _cis_check_table(rows: Sequence[CISAmplitudeCheck]) -> str:
    if not rows:
        return "<p><em>No CIS/TDA amplitude validation records were produced.</em></p>"
    shown = list(rows[:200])
    out = [
        "<table><thead><tr><th>Step</th><th>Root</th><th>Transition</th><th>Printed c</th>"
        "<th>Binary c</th><th>|diff|</th><th>Passed</th><th>Message</th></tr></thead><tbody>"
    ]
    for r in shown:
        trans = f"{r.donor}{r.donor_spin} -> {r.acceptor}{r.acceptor_spin}"
        out.append(
            f"<tr><td>{html.escape(r.step_label)}</td><td>{r.root}</td>"
            f"<td>{html.escape(trans)}</td><td>{_fmt(r.printed_coefficient)}</td>"
            f"<td>{_fmt(r.binary_coefficient)}</td><td>{_fmt(r.abs_error)}</td>"
            f"<td>{r.passed}</td><td>{html.escape(r.message)}</td></tr>"
        )
    out.append("</tbody></table>")
    if len(rows) > len(shown):
        out.append(f"<p><em>Showing first {len(shown)} of {len(rows)} checks; see diagnostics/cis_amplitude_checks.csv.</em></p>")
    return "\n".join(out)


def _css() -> str:
    return """<style>
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;color:#202124}
section{margin:24px 0} table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid #d4d7dd;padding:6px 8px;text-align:left}
th{background:#f2f4f8}.error{border-left:5px solid #b3261e;background:#fff4f4;padding:12px 16px}.plot{width:100%;max-width:900px;height:auto}
.plot .axis{stroke:#5f6368;stroke-width:1.2}.plot .xgrid,.plot .ygrid{stroke:#d9dde5;stroke-width:.8}.plot text{font-size:10px}.plot .xlabel,.plot .ylabel{font-size:11px;font-weight:600}.plot .xtick,.plot .ytick{fill:#4b5563}.heatmap td{text-align:right;font-variant-numeric:tabular-nums}.heatmap .assigned{outline:2px solid #111;font-weight:700}
.nto-pair{margin:18px 0}.figgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;align-items:start}.figgrid figure{margin:0;border:1px solid #d4d7dd;padding:8px;background:#fff}.figgrid img{display:block;width:100%;height:auto}.figgrid figcaption{font-size:12px;color:#4b5563;margin-top:6px;overflow-wrap:anywhere}.missing-img{height:180px;display:grid;place-items:center;background:#f2f4f8;color:#6b7280}
</style>"""


def _split_pair_key(key: str):
    if "->" in key:
        return key.split("->", 1)
    return key, ""


def _safe_npz_key(key: str) -> str:
    return key.replace("->", "__").replace("/", "_").replace(" ", "_")


def _path_or_none(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path)


def _rel_path(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)


def _fmt(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)

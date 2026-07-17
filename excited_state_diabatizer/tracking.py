from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import (
    AmbiguousRegion,
    AssignmentEdge,
    ConfidenceResult,
    StepData,
    TrackPoint,
    TrackingResult,
)


@dataclass
class TrackingConfig:
    assignment: str = "hungarian"
    reference_mode: str = "previous"
    similarity_threshold_good: float = 0.70
    similarity_threshold_low: float = 0.45
    assignment_margin_threshold: float = 0.10
    subspace_detection: bool = False
    subspace_gap_ev: float = 0.10
    subspace_ratio_threshold: float = 0.85
    max_subspace_size: int = 3
    anchor_weight: float = 0.25
    previous_weight: float = 0.65
    predictor_weight: float = 0.10
    energy_penalty_weight: float = 0.0
    s2_penalty_weight: float = 0.0
    nto_purity_guard: bool = False
    nto_purity_threshold: float = 0.80
    ambiguity_rescue: bool = False
    degeneracy_guard: bool = False
    degeneracy_gap_ev: float = 0.05
    dynamic_anchor_restart: bool = False
    dynamic_restart_threshold: float = 0.90


def hungarian_assignment(matrix: np.ndarray) -> List[Tuple[int, int]]:
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("Assignment matrix must be two-dimensional.")
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-mat)
        return list(zip([int(r) for r in rows], [int(c) for c in cols]))
    except Exception as exc:
        if mat.shape[0] > 8 or mat.shape[1] > 8:
            raise RuntimeError("scipy is required for Hungarian assignment on matrices larger than 8x8.") from exc
        return _bruteforce_assignment(mat)


def _bruteforce_assignment(mat: np.ndarray) -> List[Tuple[int, int]]:
    nrows, ncols = mat.shape
    best_score = -np.inf
    best_pairs: List[Tuple[int, int]] = []
    if nrows <= ncols:
        for cols in itertools.permutations(range(ncols), nrows):
            pairs = list(zip(range(nrows), cols))
            score = sum(float(mat[r, c]) for r, c in pairs)
            if score > best_score:
                best_score = score
                best_pairs = pairs
    else:
        for rows in itertools.combinations(range(nrows), ncols):
            for cols in itertools.permutations(range(ncols), ncols):
                pairs = list(zip(rows, cols))
                score = sum(float(mat[r, c]) for r, c in pairs)
                if score > best_score:
                    best_score = score
                    best_pairs = pairs
    return best_pairs


def classify_confidence(
    best: float,
    second: float,
    good_threshold: float = 0.70,
    low_threshold: float = 0.45,
    margin_threshold: float = 0.10,
) -> ConfidenceResult:
    margin = float(best - second)
    ratio = float(second / best) if best > 0 else float("inf")
    if best < low_threshold:
        return ConfidenceResult("failed", f"best similarity {best:.3f} < {low_threshold:.3f}", best, second, margin, ratio)
    if best >= good_threshold and margin >= margin_threshold:
        return ConfidenceResult("reliable", "best similarity and margin pass thresholds", best, second, margin, ratio)
    return ConfidenceResult(
        "low_confidence",
        f"best={best:.3f}, margin={margin:.3f}; at least one confidence threshold is marginal",
        best,
        second,
        margin,
        ratio,
    )


def run_tracking(
    steps: Sequence[StepData],
    roots: Sequence[int],
    matrix_provider: Callable[[int, int], np.ndarray],
    config: TrackingConfig,
    engine: str,
) -> TrackingResult:
    if not steps:
        return TrackingResult([], [], [], {}, roots, engine)
    roots = list(roots)
    root_index = {root: i for i, root in enumerate(roots)}
    track_ids = list(roots)
    current_root = {track: track for track in track_ids}
    anchor_step = {track: 0 for track in track_ids}
    anchor_root = {track: track for track in track_ids}
    track_broken = {track: False for track in track_ids}
    adjacent_matrices: Dict[str, np.ndarray] = {}
    track_points: List[TrackPoint] = []
    assignment_edges: List[AssignmentEdge] = []
    ambiguous_regions: List[AmbiguousRegion] = []

    for track in track_ids:
        st = steps[0].states.get(track)
        track_points.append(_make_track_point(track, steps[0], track, None, None, None, "seed", None, engine))

    for step_idx in range(1, len(steps)):
        prev_idx = step_idx - 1
        step_a = steps[prev_idx]
        step_b = steps[step_idx]
        prev_matrix = matrix_provider(prev_idx, step_idx)
        adjacent_matrices[f"{step_a.job.label}->{step_b.job.label}"] = prev_matrix
        score, anchor_scores = _score_matrix_for_step(
            step_idx,
            steps,
            roots,
            track_ids,
            current_root,
            anchor_step,
            anchor_root,
            track_broken,
            prev_matrix,
            matrix_provider,
            config,
        )
        _apply_penalties(score, steps, step_idx, roots, track_ids, current_root, config)
        pairs = hungarian_assignment(score)
        by_row = {r: c for r, c in pairs}
        provisional_edges: List[AssignmentEdge] = []
        for row, track in enumerate(track_ids):
            if row not in by_row:
                continue
            col = by_row[row]
            root_b = roots[col]
            root_a = current_root[track]
            row_vals = score[row, :]
            best = float(row_vals[col])
            second = _second_best(row_vals, col)
            conf = classify_confidence(
                best,
                second,
                good_threshold=config.similarity_threshold_good,
                low_threshold=config.similarity_threshold_low,
                margin_threshold=config.assignment_margin_threshold,
            )
            reason = _annotate_assignment_reason(conf.reason, row_vals, col, roots, root_a, root_b, conf.confidence)
            prev_sim = float(prev_matrix[root_index[root_a], col]) if root_a in root_index else best
            anchor_sim = None
            if anchor_scores is not None:
                anchor_sim = float(anchor_scores[row, col])
            edge = AssignmentEdge(
                step_a=step_a.job.label,
                step_b=step_b.job.label,
                track_id=track,
                root_a=root_a,
                root_b=root_b,
                best_similarity=best,
                second_similarity=second,
                margin=conf.margin,
                ratio=conf.ratio,
                confidence=conf.confidence,
                reason=reason,
                similarity_to_previous=prev_sim,
                similarity_to_anchor=anchor_sim,
            )
            if config.dynamic_anchor_restart and prev_sim >= config.dynamic_restart_threshold:
                if edge.confidence in {"failed", "low_confidence"}:
                    edge.confidence = "reliable"
                    edge.reason = edge.reason + f"; local similarity {prev_sim:.3f} >= {config.dynamic_restart_threshold:.3f} overrides low global confidence"
            _apply_mixed_degeneracy_guard(edge, step_b, roots, config)
            provisional_edges.append(edge)

        if config.subspace_detection:
            regions = detect_subspaces_for_pair(
                prev_matrix,
                roots,
                provisional_edges,
                step_a,
                step_b,
                config,
                start_index=len(ambiguous_regions) + 1,
            )
            ambiguous_regions.extend(regions)
            _tag_edges_with_regions(provisional_edges, regions)

        for edge in provisional_edges:
            was_provisional = track_broken.get(edge.track_id, False)
            reconnected = was_provisional and _reconnects_to_stable_anchor(edge, config)
            if was_provisional and edge.confidence == "reliable" and not reconnected:
                edge.confidence = "ambiguous"
                edge.reason = (
                    edge.reason
                    + "; an earlier provisional assignment left this track unresolved, so this later adjacent match is provisional"
                )
            elif reconnected:
                edge.reason = edge.reason + "; reconnected to the last stable pre-ambiguity anchor"
            assignment_edges.append(edge)
            current_root[edge.track_id] = edge.root_b
            if config.reference_mode in {"adaptive", "hybrid"} or config.nto_purity_guard or config.ambiguity_rescue:
                if _can_update_anchor(edge, config):
                    anchor_step[edge.track_id] = step_idx
                    anchor_root[edge.track_id] = edge.root_b
                    track_broken[edge.track_id] = False
            if edge.confidence in {"failed", "low_confidence", "ambiguous"}:
                track_broken[edge.track_id] = True
            track_points.append(
                _make_track_point(
                    edge.track_id,
                    step_b,
                    edge.root_b,
                    edge.similarity_to_previous,
                    edge.similarity_to_anchor,
                    edge.margin,
                    edge.confidence,
                    edge.manifold_id,
                    engine,
                )
            )

    disagreements = bidirectional_disagreements(adjacent_matrices, roots)
    return TrackingResult(
        track_points=track_points,
        assignment_edges=assignment_edges,
        ambiguous_regions=ambiguous_regions,
        adjacent_matrices=adjacent_matrices,
        roots=roots,
        engine=engine,
        bidirectional_disagreements=disagreements,
    )


def _score_matrix_for_step(
    step_idx: int,
    steps: Sequence[StepData],
    roots: Sequence[int],
    track_ids: Sequence[int],
    current_root: Dict[int, int],
    anchor_step: Dict[int, int],
    anchor_root: Dict[int, int],
    track_broken: Dict[int, bool],
    prev_matrix: np.ndarray,
    matrix_provider: Callable[[int, int], np.ndarray],
    config: TrackingConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    root_index = {root: i for i, root in enumerate(roots)}
    score = np.zeros((len(track_ids), len(roots)), dtype=float)
    anchor_scores = None
    mode = config.reference_mode
    if mode == "first":
        first = matrix_provider(0, step_idx)
        for row, track in enumerate(track_ids):
            score[row, :] = first[root_index[track], :]
        return score, first[[root_index[t] for t in track_ids], :]
    if mode == "previous":
        if config.nto_purity_guard or config.ambiguity_rescue:
            anchor_scores = np.zeros_like(score)
        for row, track in enumerate(track_ids):
            if config.ambiguity_rescue and track_broken.get(track, False):
                amat = matrix_provider(anchor_step[track], step_idx)
                anchor_scores[row, :] = amat[root_index[anchor_root[track]], :]
                score[row, :] = anchor_scores[row, :]
            else:
                score[row, :] = prev_matrix[root_index[current_root[track]], :]
                if anchor_scores is not None:
                    anchor_scores[row, :] = score[row, :]
        return score, anchor_scores
    if mode in {"adaptive", "hybrid"}:
        anchor_scores = np.zeros_like(score)
        for row, track in enumerate(track_ids):
            amat = matrix_provider(anchor_step[track], step_idx)
            anchor_scores[row, :] = amat[root_index[anchor_root[track]], :]
            if mode == "adaptive":
                score[row, :] = anchor_scores[row, :]
            else:
                score[row, :] = (
                    config.previous_weight * prev_matrix[root_index[current_root[track]], :]
                    + config.anchor_weight * anchor_scores[row, :]
                )
        return score, anchor_scores
    raise ValueError(f"Unknown reference mode: {mode}")


def _can_update_anchor(edge: AssignmentEdge, config: TrackingConfig) -> bool:
    if edge.confidence != "reliable" or edge.manifold_id:
        return False
    if config.dynamic_anchor_restart and edge.similarity_to_previous is not None:
        if edge.similarity_to_previous >= config.dynamic_restart_threshold:
            return True
    if edge.best_similarity < 0.65 or edge.margin < config.assignment_margin_threshold:
        return False
    if edge.similarity_to_anchor is not None and edge.similarity_to_anchor < 0.65:
        return False
    return True


def _apply_mixed_degeneracy_guard(edge: AssignmentEdge, current_step: StepData, roots: Sequence[int], config: TrackingConfig) -> None:
    if not config.nto_purity_guard:
        return
        
    purity = _dominant_nto_fraction(current_step, edge.root_b)
    if purity is None:
        edge.reason = edge.reason + "; no self-derived NTO purity diagnostic was available"
        return
        
    if purity >= config.nto_purity_threshold:
        return
        
    if config.degeneracy_guard:
        current_state = current_step.states.get(edge.root_b)
        if not current_state or current_state.exc_ev is None:
            return
            
        is_degenerate = False
        degenerate_root = None
        min_gap = float('inf')
        
        for other_root in roots:
            if other_root == edge.root_b:
                continue
            other_state = current_step.states.get(other_root)
            if not other_state or other_state.exc_ev is None:
                continue
                
            gap = abs(current_state.exc_ev - other_state.exc_ev)
            if gap < config.degeneracy_gap_ev:
                is_degenerate = True
                if gap < min_gap:
                    min_gap = gap
                    degenerate_root = other_root
                    
        if not is_degenerate:
            return
            
        if edge.confidence != "failed":
            edge.confidence = "ambiguous"
        edge.reason = (
            edge.reason
            + f"; current root {edge.root_b} is NTO-mixed (purity {purity:.3f} < {config.nto_purity_threshold:.3f}) "
            f"AND near-degenerate with root {degenerate_root} (gap {min_gap:.3f} eV < {config.degeneracy_gap_ev:.3f} eV); "
            f"this point will not update the stable reference"
        )
    else:
        if edge.confidence != "failed":
            edge.confidence = "ambiguous"
        edge.reason = (
            edge.reason
            + f"; current root is NTO-mixed: dominant pair fraction {purity:.3f} "
            f"< {config.nto_purity_threshold:.3f}; this point will not update the stable reference"
        )


def _dominant_nto_fraction(step: StepData, root: int) -> Optional[float]:
    vec = step.self_nto_vectors.get(root) or step.nto_vectors.get(root)
    if vec is None or not vec.pairs:
        return None
    total = float(sum(max(0.0, p.weight) for p in vec.pairs))
    if total <= 0:
        return None
    return float(max(max(0.0, p.weight) for p in vec.pairs) / total)


def _reconnects_to_stable_anchor(edge: AssignmentEdge, config: TrackingConfig) -> bool:
    if edge.similarity_to_anchor is None:
        return False
    return _can_update_anchor(edge, config)


def _apply_penalties(
    score: np.ndarray,
    steps: Sequence[StepData],
    step_idx: int,
    roots: Sequence[int],
    track_ids: Sequence[int],
    current_root: Dict[int, int],
    config: TrackingConfig,
) -> None:
    if config.energy_penalty_weight <= 0 and config.s2_penalty_weight <= 0:
        return
    prev_step = steps[step_idx - 1]
    curr_step = steps[step_idx]
    for row, track in enumerate(track_ids):
        prev_state = prev_step.states.get(current_root[track])
        for col, root in enumerate(roots):
            curr_state = curr_step.states.get(root)
            if config.energy_penalty_weight > 0 and prev_state and curr_state:
                if prev_state.exc_ev is not None and curr_state.exc_ev is not None:
                    score[row, col] -= config.energy_penalty_weight * abs(curr_state.exc_ev - prev_state.exc_ev)
            if config.s2_penalty_weight > 0 and prev_state and curr_state:
                if prev_state.s2 is not None and curr_state.s2 is not None:
                    score[row, col] -= config.s2_penalty_weight * abs(curr_state.s2 - prev_state.s2)


def detect_subspaces_for_pair(
    matrix: np.ndarray,
    roots: Sequence[int],
    edges: Sequence[AssignmentEdge],
    step_a: StepData,
    step_b: StepData,
    config: TrackingConfig,
    start_index: int = 1,
) -> List[AmbiguousRegion]:
    root_index = {root: i for i, root in enumerate(roots)}
    regions: List[AmbiguousRegion] = []
    next_id = start_index
    seen_blocks = set()
    for edge in edges:
        row = root_index.get(edge.root_a)
        col = root_index.get(edge.root_b)
        if row is None or col is None:
            continue
        vals = np.asarray(matrix[row, :], dtype=float)
        if len(vals) < 2:
            continue
        second_col = _second_best_col(vals, col)
        if second_col is None:
            continue
        second_root = roots[second_col]
        best = float(vals[col])
        second = float(vals[second_col])
        ratio = second / best if best > 0 else float("inf")
        close_overlap = ratio > config.subspace_ratio_threshold or (best - second) < config.assignment_margin_threshold
        if not close_overlap:
            continue
        gap_ok = _energy_gap_ok(step_b, edge.root_b, second_root, config.subspace_gap_ev)
        if not gap_ok:
            continue
        curr_roots = sorted({edge.root_b, second_root})
        if len(curr_roots) < 2:
            continue
        prev_roots = [edge.root_a]
        for other in edges:
            if other is not edge and other.root_b in curr_roots and other.root_a not in prev_roots:
                prev_roots.append(other.root_a)
        prev_roots = prev_roots[: config.max_subspace_size]
        curr_roots = curr_roots[: config.max_subspace_size]
        block_key = (tuple(sorted(prev_roots)), tuple(sorted(curr_roots)))
        if block_key in seen_blocks:
            continue
        seen_blocks.add(block_key)
        sub = matrix[[root_index[r] for r in prev_roots], :][:, [root_index[r] for r in curr_roots]]
        size = max(1, max(sub.shape))
        subspace_score = min(1.0, float(np.linalg.norm(sub, ord="fro") / np.sqrt(size)))
        region_id = f"M{next_id:03d}"
        next_id += 1
        tracks = sorted({e.track_id for e in edges if e.root_a in prev_roots or e.root_b in curr_roots})
        regions.append(
            AmbiguousRegion(
                region_id=region_id,
                start_step=step_a.job.label,
                end_step=step_b.job.label,
                manifold_roots="/".join(str(r) for r in curr_roots),
                tracks_involved="/".join(str(t) for t in tracks),
                min_subspace_score=subspace_score,
                reason=(
                    f"competitor/assigned={ratio:.3f}, assigned-competitor margin={best - second:.3f}; "
                    f"current roots {curr_roots} are close in overlap and energy"
                ),
            )
        )
    return regions


def bidirectional_disagreements(adjacent_matrices: Dict[str, np.ndarray], roots: Sequence[int]) -> List[dict]:
    out: List[dict] = []
    roots = list(roots)
    for key, mat in adjacent_matrices.items():
        forward = {(roots[r], roots[c]) for r, c in hungarian_assignment(mat)}
        backward_pairs = hungarian_assignment(np.asarray(mat).T)
        backward = {(roots[c], roots[r]) for r, c in backward_pairs}
        for pair in sorted(forward.symmetric_difference(backward)):
            out.append({"step_pair": key, "root_a": pair[0], "root_b": pair[1], "reason": "forward/backward mismatch"})
    return out


def _tag_edges_with_regions(edges: Sequence[AssignmentEdge], regions: Sequence[AmbiguousRegion]) -> None:
    for region in regions:
        roots = {int(x) for x in region.manifold_roots.split("/") if x}
        tracks = {int(x) for x in region.tracks_involved.split("/") if x}
        for edge in edges:
            if edge.track_id in tracks or edge.root_b in roots:
                edge.manifold_id = region.region_id
                if edge.confidence == "reliable":
                    edge.confidence = "ambiguous"
                    edge.reason = f"{edge.reason}; provisional inside {region.region_id}"


def _energy_gap_ok(step: StepData, root_a: int, root_b: int, gap_ev: float) -> bool:
    sa = step.states.get(root_a)
    sb = step.states.get(root_b)
    if not sa or not sb or sa.exc_ev is None or sb.exc_ev is None:
        return True
    return abs(sa.exc_ev - sb.exc_ev) <= gap_ev


def _make_track_point(
    track: int,
    step: StepData,
    root: int,
    sim_prev: Optional[float],
    sim_anchor: Optional[float],
    margin: Optional[float],
    confidence: str,
    manifold_id: Optional[str],
    engine: str,
) -> TrackPoint:
    state = step.states.get(root)
    return TrackPoint(
        track_id=track,
        step_order=step.job.order,
        step_label=step.job.label,
        scan_step=step.job.scan_step,
        adiabatic_root=root,
        energy_eh=None if state is None else state.abs_energy_eh,
        excitation_ev=None if state is None else state.exc_ev,
        s2=None if state is None else state.s2,
        multiplicity=None if state is None else state.multiplicity,
        similarity_to_previous=sim_prev,
        similarity_to_anchor=sim_anchor,
        assignment_margin=margin,
        confidence=confidence,
        manifold_id=manifold_id,
        engine=engine,
    )


def _second_best(vals: np.ndarray, selected_col: int) -> float:
    if vals.size <= 1:
        return 0.0
    mask = np.ones(vals.size, dtype=bool)
    mask[selected_col] = False
    return float(np.max(vals[mask]))


def _second_best_col(vals: np.ndarray, selected_col: int) -> Optional[int]:
    if vals.size <= 1:
        return None
    mask = np.ones(vals.size, dtype=bool)
    mask[selected_col] = False
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return None
    return int(candidates[int(np.argmax(vals[candidates]))])


def _annotate_assignment_reason(
    reason: str,
    row_vals: np.ndarray,
    selected_col: int,
    roots: Sequence[int],
    root_a: int,
    root_b: int,
    confidence: str,
) -> str:
    if row_vals.size == 0:
        return reason
    row_best_col = int(np.argmax(row_vals))
    pieces = [reason]
    if row_best_col != selected_col:
        row_best_root = roots[row_best_col]
        pieces.append(
            f"global assignment selected root {root_b}, but row-best candidate is root {row_best_root} "
            f"(S={float(row_vals[row_best_col]):.3f})"
        )
    boundary_roots = {min(roots), max(roots)} if roots else set()
    if confidence == "failed" and ({root_a, root_b, roots[row_best_col]} & boundary_roots):
        pieces.append("root-window boundary is involved; include more roots if available")
    return "; ".join(pieces)

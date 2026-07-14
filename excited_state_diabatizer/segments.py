from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import StepData, TrackPoint, TrackSegment, TrackingResult


def make_reversed_matrix_provider(matrix_provider: Callable[[int, int], np.ndarray], n_steps: int):
    """Adapt a forward chronological matrix provider to reversed StepData order."""

    def provider(i: int, j: int) -> np.ndarray:
        oi = n_steps - 1 - i
        oj = n_steps - 1 - j
        if oi == oj:
            raise ValueError("Cannot build a state-overlap matrix for identical steps.")
        if oi < oj:
            return matrix_provider(oi, oj)
        return np.asarray(matrix_provider(oj, oi), dtype=float).T

    return provider


def build_track_segments(
    forward_result: Optional[TrackingResult],
    reverse_result: Optional[TrackingResult],
    steps: Sequence[StepData],
    min_purity: float = 0.90,
    min_adjacent_similarity: float = 0.85,
    min_length: int = 2,
    seed_only_pure_roots: bool = True,
) -> List[TrackSegment]:
    forward = _segments_for_result(
        forward_result,
        steps,
        "forward",
        min_purity,
        min_adjacent_similarity,
        min_length,
        seed_only_pure_roots,
    )
    reverse = _segments_for_result(
        reverse_result,
        steps,
        "reverse",
        min_purity,
        min_adjacent_similarity,
        min_length,
        seed_only_pure_roots,
    )
    _annotate_support(forward, reverse)
    _renumber(forward + reverse)
    return forward + reverse


def _segments_for_result(
    result: Optional[TrackingResult],
    steps: Sequence[StepData],
    direction: str,
    min_purity: float,
    min_adjacent_similarity: float,
    min_length: int,
    seed_only_pure_roots: bool,
) -> List[TrackSegment]:
    if result is None:
        return []
    step_by_label = {step.job.label: step for step in steps}
    grouped: Dict[int, List[TrackPoint]] = {}
    for point in result.track_points:
        grouped.setdefault(int(point.track_id), []).append(point)
    out: List[TrackSegment] = []
    for track_id, raw_points in sorted(grouped.items()):
        reverse = direction == "reverse"
        points = sorted(raw_points, key=lambda p: p.step_order, reverse=reverse)
        run: List[TrackPoint] = []
        for point in points:
            purity = _dominant_fraction(step_by_label.get(point.step_label), point.adiabatic_root)
            node_ok = purity is not None and purity >= min_purity and not point.manifold_id
            edge_ok = not run or (
                point.similarity_to_previous is not None and float(point.similarity_to_previous) >= min_adjacent_similarity
            )
            if node_ok and edge_ok:
                if not run and seed_only_pure_roots and purity < min_purity:
                    continue
                run.append(point)
                continue
            _append_segment(out, run, step_by_label, direction, track_id, min_length)
            run = [point] if node_ok else []
        _append_segment(out, run, step_by_label, direction, track_id, min_length)
    return out


def _append_segment(
    out: List[TrackSegment],
    run: Sequence[TrackPoint],
    step_by_label: Dict[str, StepData],
    direction: str,
    track_id: int,
    min_length: int,
) -> None:
    if len(run) < min_length:
        return
    chronological = sorted(run, key=lambda p: p.step_order)
    traversal = list(run)
    purities = [_dominant_fraction(step_by_label.get(p.step_label), p.adiabatic_root) for p in chronological]
    ranks = [_effective_rank(step_by_label.get(p.step_label), p.adiabatic_root) for p in chronological]
    edge_sims = [p.similarity_to_previous for p in traversal[1:] if p.similarity_to_previous is not None]
    seed = traversal[0]
    start = chronological[0]
    end = chronological[-1]
    out.append(
        TrackSegment(
            segment_id=f"{direction[0].upper()}{len(out) + 1:03d}",
            direction=direction,
            support=f"{direction}_only",
            seed_track_id=track_id,
            seed_step=seed.step_label,
            seed_root=seed.adiabatic_root,
            start_step=start.step_label,
            end_step=end.step_label,
            start_root=start.adiabatic_root,
            end_root=end.adiabatic_root,
            step_count=len(chronological),
            step_sequence=" -> ".join(p.step_label for p in chronological),
            root_sequence=" -> ".join(f"{p.step_label}:r{p.adiabatic_root}" for p in chronological),
            min_adjacent_similarity=None if not edge_sims else float(min(edge_sims)),
            min_purity=_min_optional(purities),
            mean_purity=_mean_optional(purities),
            max_effective_rank=_max_optional(ranks),
            status="local_stable_branch",
            reason=(
                "Locally continuous pure branch. This is segment evidence, not proof that the branch "
                "connects to a pre-ambiguity anchor."
            ),
        )
    )


def _annotate_support(forward: Sequence[TrackSegment], reverse: Sequence[TrackSegment]) -> None:
    for seg in forward:
        if _has_matching_segment(seg, reverse):
            seg.support = "two_sided"
            seg.reason += " Confirmed by an overlapping reverse-tracked segment."
    for seg in reverse:
        if _has_matching_segment(seg, forward):
            seg.support = "two_sided"
            seg.reason += " Confirmed by an overlapping forward-tracked segment."


def _has_matching_segment(segment: TrackSegment, others: Sequence[TrackSegment]) -> bool:
    sig = _segment_signature(segment)
    if len(sig) < 2:
        return False
    for other in others:
        osig = _segment_signature(other)
        common = len(sig & osig)
        if common >= 2 and common / max(1, min(len(sig), len(osig))) >= 0.80:
            return True
    return False


def _segment_signature(segment: TrackSegment) -> set[Tuple[str, int]]:
    out = set()
    for item in segment.root_sequence.split(" -> "):
        if ":r" not in item:
            continue
        step, root = item.split(":r", 1)
        try:
            out.add((step, int(root)))
        except ValueError:
            continue
    return out


def _renumber(segments: Sequence[TrackSegment]) -> None:
    for idx, seg in enumerate(sorted(segments, key=lambda s: (s.start_step, s.end_step, s.direction, s.seed_track_id)), start=1):
        seg.segment_id = f"S{idx:03d}"


def _dominant_fraction(step: Optional[StepData], root: int) -> Optional[float]:
    if step is None:
        return None
    vec = step.self_nto_vectors.get(root) or step.nto_vectors.get(root)
    if vec is None or not vec.pairs:
        return None
    weights = [max(0.0, float(pair.weight)) for pair in vec.pairs]
    total = sum(weights)
    if total <= 0.0:
        return None
    return max(weights) / total


def _effective_rank(step: Optional[StepData], root: int) -> Optional[float]:
    if step is None:
        return None
    vec = step.self_nto_vectors.get(root) or step.nto_vectors.get(root)
    if vec is None or not vec.pairs:
        return None
    weights = [max(0.0, float(pair.weight)) for pair in vec.pairs]
    total = sum(weights)
    denom = sum(w * w for w in weights)
    if total <= 0.0 or denom <= 0.0:
        return None
    return (total * total) / denom


def _min_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return min(vals) if vals else None


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _max_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import (
    ExtractionStatus,
    MissingDataError,
    NTOOrbitalPair,
    NTOStateVector,
    SelfNTOReconstructionRecord,
    SelfNTOWeightRecord,
    StepData,
)

CACHE_VERSION = "self-nto-from-cis-cache-v2"


def derive_self_ntos_from_tden_steps(
    steps: Sequence[StepData],
    roots: Sequence[int],
    weight_min: float = 0.01,
    cumulative: float = 0.995,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[List[SelfNTOWeightRecord], List[SelfNTOReconstructionRecord], List[ExtractionStatus]]:
    """Derive diagnostic NTOs by SVD of validated CIS/TDA amplitude matrices.

    The assignment engine still uses the full transition-density overlap. These
    NTOs are an interpretation layer derived from the same numerical amplitudes,
    avoiding ORCA .nto JSON indexing assumptions.
    """

    weight_rows: List[SelfNTOWeightRecord] = []
    reconstruction_rows: List[SelfNTOReconstructionRecord] = []
    statuses: List[ExtractionStatus] = []
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for step in steps:
        step.self_nto_vectors.clear()
        for root in roots:
            state = step.tden_vectors.get(root)
            if state is None:
                continue
            source = _source_file(state)
            cache_path = _cache_path(cache_dir, step.job.label, root, weight_min, cumulative)
            try:
                cached = None if force else _try_read_cache(cache_path, step.job.label, root, source, weight_min, cumulative)
                if cached is not None:
                    weights, recon = cached
                    selected_count = sum(1 for row in weights if row.selected)
                    step.self_nto_vectors[root] = NTOStateVector(
                        step_label=step.job.label,
                        step_order=step.job.order,
                        root=root,
                        source_file=source,
                        source_type="self-nto-from-cis-cache",
                        diagnostics=["Loaded cached self-derived NTO diagnostics; coefficient vectors were not cached."],
                    )
                    status = "cached"
                    message = f"Loaded {selected_count} selected self-derived NTO pair(s) from cache."
                else:
                    vec, weights, recon = derive_self_nto_state(
                        step.job.label,
                        step.job.order,
                        root,
                        state,
                        step.mo_coefficients,
                        weight_min=weight_min,
                        cumulative=cumulative,
                    )
                    step.self_nto_vectors[root] = vec
                    selected_count = len(vec.pairs)
                    _write_cache(cache_path, step.job.label, root, source, weight_min, cumulative, weights, recon)
                    status = "ok"
                    message = f"Derived {selected_count} selected NTO pair(s) from CIS/TDA amplitude SVD."

                weight_rows.extend(weights)
                reconstruction_rows.extend(recon)
                statuses.append(
                    ExtractionStatus(
                        step.job.label,
                        root,
                        "self-nto-from-cis",
                        source_file=source,
                        json_file=cache_path,
                        status=status,
                        message=message,
                    )
                )
            except Exception as exc:
                statuses.append(
                    ExtractionStatus(
                        step.job.label,
                        root,
                        "self-nto-from-cis",
                        source_file=source,
                        json_file=cache_path,
                        status="failed",
                        message=str(exc),
                    )
                )

    return weight_rows, reconstruction_rows, statuses


def derive_self_nto_state(
    step_label: str,
    step_order: int,
    root: int,
    state: dict,
    mo_coefficients: Dict[str, np.ndarray],
    weight_min: float = 0.01,
    cumulative: float = 0.995,
) -> Tuple[NTOStateVector, List[SelfNTOWeightRecord], List[SelfNTOReconstructionRecord]]:
    source = _source_file(state)
    vec = NTOStateVector(
        step_label=step_label,
        step_order=step_order,
        root=root,
        source_file=source,
        source_type="self-nto-from-cis",
    )
    spin_results: Dict[str, _SpinNTOResult] = {}
    state_weight = 0.0
    for spin in ("alpha", "beta"):
        matrix = _spin_matrix(state, spin)
        if matrix is None or matrix.size == 0:
            continue
        result = _derive_spin_ntos(state, spin, matrix, mo_coefficients)
        spin_results[spin] = result
        state_weight += result.total_weight

    weight_rows: List[SelfNTOWeightRecord] = []
    reconstruction_rows: List[SelfNTOReconstructionRecord] = []
    selected_by_spin = _global_selected_masks(spin_results, state_weight, weight_min, cumulative)
    for spin, result in spin_results.items():
        selected = selected_by_spin.get(spin, np.zeros(result.weights.shape, dtype=bool))
        cumulative_spin = 0.0
        selected_weight = 0.0
        for idx, (singular_value, weight) in enumerate(zip(result.singular_values, result.weights), start=1):
            cumulative_spin += float(weight)
            if selected[idx - 1]:
                selected_weight += float(weight)
                vec.pairs.append(
                    NTOOrbitalPair(
                        spin=spin,
                        donor_index=idx,
                        acceptor_index=idx,
                        weight=float(weight),
                        donor_coeff=result.hole_coefficients[:, idx - 1].copy(),
                        acceptor_coeff=result.particle_coefficients[:, idx - 1].copy(),
                        source_file=source,
                        donor_vector_index=idx - 1,
                        acceptor_vector_index=idx - 1,
                    )
                )
            weight_rows.append(
                SelfNTOWeightRecord(
                    step_label=step_label,
                    root=root,
                    spin=spin,
                    source_file=source,
                    pair_index=idx,
                    singular_value=float(singular_value),
                    weight=float(weight),
                    spin_weight_fraction=_safe_fraction(weight, result.total_weight),
                    state_weight_fraction=_safe_fraction(weight, state_weight),
                    cumulative_spin_fraction=_safe_fraction(cumulative_spin, result.total_weight),
                    selected=bool(selected[idx - 1]),
                )
            )
        reconstruction_rows.append(
            SelfNTOReconstructionRecord(
                step_label=step_label,
                root=root,
                spin=spin,
                source_file=source,
                n_occ=result.matrix.shape[0],
                n_virt=result.matrix.shape[1],
                rank=len(result.singular_values),
                matrix_norm=result.matrix_norm,
                reconstruction_error=result.reconstruction_error,
                relative_error=result.relative_error,
                selected_pairs=int(np.count_nonzero(selected)),
                selected_spin_weight_fraction=_safe_fraction(selected_weight, result.total_weight),
                passed=result.relative_error <= 1.0e-10,
                message="full SVD reconstructs CIS/TDA spin block" if result.relative_error <= 1.0e-10 else "large SVD reconstruction residual",
            )
        )
    if not vec.pairs:
        vec.diagnostics.append("No self-derived NTO pairs survived the selected weight/cumulative thresholds.")
    else:
        vec.diagnostics.append(
            "Self-derived from validated ORCA job.cis amplitudes; pair indices are SVD ranks, not ORCA .nto file orbital indices."
        )
    return vec, weight_rows, reconstruction_rows


class _SpinNTOResult:
    def __init__(
        self,
        matrix: np.ndarray,
        singular_values: np.ndarray,
        weights: np.ndarray,
        hole_coefficients: np.ndarray,
        particle_coefficients: np.ndarray,
        reconstruction_error: float,
    ) -> None:
        self.matrix = matrix
        self.singular_values = singular_values
        self.weights = weights
        self.hole_coefficients = hole_coefficients
        self.particle_coefficients = particle_coefficients
        self.reconstruction_error = float(reconstruction_error)
        self.matrix_norm = float(np.linalg.norm(matrix))
        self.relative_error = _safe_fraction(reconstruction_error, self.matrix_norm)
        self.total_weight = float(np.sum(weights))


def _derive_spin_ntos(
    state: dict,
    spin: str,
    matrix: np.ndarray,
    mo_coefficients: Dict[str, np.ndarray],
) -> _SpinNTOResult:
    coeffs = mo_coefficients.get(spin)
    if coeffs is None:
        raise MissingDataError(f"Missing {spin} MO coefficients needed to derive NTOs from CIS/TDA amplitudes.")
    tden = np.asarray(matrix, dtype=float)
    if tden.ndim != 2:
        raise MissingDataError(f"{spin} transition-density amplitudes are not a two-dimensional matrix.")
    u, singular_values, vt = np.linalg.svd(tden, full_matrices=False)
    reconstructed = (u * singular_values) @ vt
    reconstruction_error = float(np.linalg.norm(tden - reconstructed))
    c_occ = _active_mo_block(coeffs, _active_range(state, spin, "occ"), tden.shape[0], spin, "occupied")
    c_virt = _active_mo_block(coeffs, _active_range(state, spin, "virt"), tden.shape[1], spin, "virtual")
    hole_coefficients = c_occ @ u
    particle_coefficients = c_virt @ vt.T
    return _SpinNTOResult(
        matrix=tden,
        singular_values=np.asarray(singular_values, dtype=float),
        weights=np.asarray(singular_values * singular_values, dtype=float),
        hole_coefficients=hole_coefficients,
        particle_coefficients=particle_coefficients,
        reconstruction_error=reconstruction_error,
    )


def _spin_matrix(state: dict, spin: str) -> Optional[np.ndarray]:
    for key in (spin, spin[0], f"{spin}_amplitudes", f"{spin}_transition_density"):
        if key in state:
            return np.asarray(state[key], dtype=float)
    if spin == "alpha":
        for key in ("amplitudes", "transition_density", "matrix"):
            if key in state:
                return np.asarray(state[key], dtype=float)
    return None


def _active_range(state: dict, spin: str, kind: str) -> Tuple[int, int]:
    key = f"{spin}_{kind}_range"
    if key not in state:
        key = f"{spin[0]}_{kind}_range"
    if key not in state:
        raise MissingDataError(f"Missing {spin} {kind} active MO range in CIS/TDA amplitude record.")
    start, end = state[key]
    return int(start), int(end)


def _active_mo_block(coefficients: np.ndarray, active_range: Tuple[int, int], expected_width: int, spin: str, kind: str) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.ndim != 2:
        raise MissingDataError(f"{spin} MO coefficient block is not two-dimensional.")
    start, end = active_range
    start0 = start - 1
    end0 = end
    if start0 < 0 or end0 > coeffs.shape[1] or start > end:
        raise MissingDataError(
            f"{spin} {kind} active MO range {start}..{end} is incompatible with {coeffs.shape[1]} available MO columns."
        )
    block = coeffs[:, start0:end0]
    if block.shape[1] != expected_width:
        raise MissingDataError(
            f"{spin} {kind} active MO range {start}..{end} gives {block.shape[1]} columns, "
            f"but the CIS/TDA amplitude matrix expects {expected_width}."
        )
    return block


def _global_selected_masks(
    spin_results: Dict[str, _SpinNTOResult],
    state_total: float,
    weight_min: float,
    cumulative: float,
) -> Dict[str, np.ndarray]:
    masks = {spin: np.zeros(result.weights.shape, dtype=bool) for spin, result in spin_results.items()}
    if state_total <= 0:
        return masks
    threshold = float(weight_min)
    target = max(0.0, min(1.0, float(cumulative)))
    ranked = []
    for spin, result in spin_results.items():
        for idx, weight in enumerate(result.weights):
            ranked.append((float(weight), spin, idx))
    ranked.sort(reverse=True, key=lambda row: row[0])
    running = 0.0
    for weight, spin, idx in ranked:
        state_fraction = weight / state_total
        if state_fraction >= threshold or running / state_total < target:
            masks[spin][idx] = True
        running += weight
    return masks


def _cache_path(cache_dir: Optional[Path], step_label: str, root: int, weight_min: float, cumulative: float) -> Optional[Path]:
    if cache_dir is None:
        return None
    w = _float_token(weight_min)
    c = _float_token(cumulative)
    return Path(cache_dir) / f"{step_label}_root{int(root):03d}_w{w}_c{c}.json"


def _float_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _try_read_cache(
    path: Optional[Path],
    step_label: str,
    root: int,
    source: Path,
    weight_min: float,
    cumulative: float,
) -> Optional[Tuple[List[SelfNTOWeightRecord], List[SelfNTOReconstructionRecord]]]:
    if path is None or not Path(path).exists():
        return None
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return None
    if payload.get("version") != CACHE_VERSION:
        return None
    if payload.get("step_label") != step_label or int(payload.get("root", -1)) != int(root):
        return None
    if abs(float(payload.get("weight_min", -1.0)) - float(weight_min)) > 1.0e-15:
        return None
    if abs(float(payload.get("cumulative", -1.0)) - float(cumulative)) > 1.0e-15:
        return None
    if payload.get("source_signature") != _source_signature(source):
        return None
    weights = [_weight_from_dict(row) for row in payload.get("weights", [])]
    reconstruction = [_reconstruction_from_dict(row) for row in payload.get("reconstruction", [])]
    return weights, reconstruction


def _write_cache(
    path: Optional[Path],
    step_label: str,
    root: int,
    source: Path,
    weight_min: float,
    cumulative: float,
    weights: Sequence[SelfNTOWeightRecord],
    reconstruction: Sequence[SelfNTOReconstructionRecord],
) -> None:
    if path is None:
        return
    payload = {
        "version": CACHE_VERSION,
        "step_label": step_label,
        "root": int(root),
        "source_file": str(source),
        "source_signature": _source_signature(source),
        "weight_min": float(weight_min),
        "cumulative": float(cumulative),
        "weights": [_weight_to_dict(row) for row in weights],
        "reconstruction": [_reconstruction_to_dict(row) for row in reconstruction],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_signature(source: Path) -> dict:
    path = Path(source)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _weight_to_dict(row: SelfNTOWeightRecord) -> dict:
    return {
        "step_label": row.step_label,
        "root": int(row.root),
        "spin": row.spin,
        "source_file": str(row.source_file),
        "pair_index": int(row.pair_index),
        "singular_value": float(row.singular_value),
        "weight": float(row.weight),
        "spin_weight_fraction": float(row.spin_weight_fraction),
        "state_weight_fraction": float(row.state_weight_fraction),
        "cumulative_spin_fraction": float(row.cumulative_spin_fraction),
        "selected": bool(row.selected),
    }


def _weight_from_dict(row: dict) -> SelfNTOWeightRecord:
    return SelfNTOWeightRecord(
        step_label=str(row["step_label"]),
        root=int(row["root"]),
        spin=str(row["spin"]),
        source_file=Path(row.get("source_file", "")),
        pair_index=int(row["pair_index"]),
        singular_value=float(row["singular_value"]),
        weight=float(row["weight"]),
        spin_weight_fraction=float(row["spin_weight_fraction"]),
        state_weight_fraction=float(row["state_weight_fraction"]),
        cumulative_spin_fraction=float(row["cumulative_spin_fraction"]),
        selected=bool(row["selected"]),
    )


def _reconstruction_to_dict(row: SelfNTOReconstructionRecord) -> dict:
    return {
        "step_label": row.step_label,
        "root": int(row.root),
        "spin": row.spin,
        "source_file": str(row.source_file),
        "n_occ": int(row.n_occ),
        "n_virt": int(row.n_virt),
        "rank": int(row.rank),
        "matrix_norm": float(row.matrix_norm),
        "reconstruction_error": float(row.reconstruction_error),
        "relative_error": float(row.relative_error),
        "selected_pairs": int(row.selected_pairs),
        "selected_spin_weight_fraction": float(row.selected_spin_weight_fraction),
        "passed": bool(row.passed),
        "message": row.message,
    }


def _reconstruction_from_dict(row: dict) -> SelfNTOReconstructionRecord:
    return SelfNTOReconstructionRecord(
        step_label=str(row["step_label"]),
        root=int(row["root"]),
        spin=str(row["spin"]),
        source_file=Path(row.get("source_file", "")),
        n_occ=int(row["n_occ"]),
        n_virt=int(row["n_virt"]),
        rank=int(row["rank"]),
        matrix_norm=float(row["matrix_norm"]),
        reconstruction_error=float(row["reconstruction_error"]),
        relative_error=float(row["relative_error"]),
        selected_pairs=int(row["selected_pairs"]),
        selected_spin_weight_fraction=float(row["selected_spin_weight_fraction"]),
        passed=bool(row["passed"]),
        message=str(row.get("message", "")),
    )


def _source_file(state: dict) -> Path:
    source = state.get("source_file") if isinstance(state, dict) else None
    return Path(source) if source is not None else Path("")


def _safe_fraction(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator

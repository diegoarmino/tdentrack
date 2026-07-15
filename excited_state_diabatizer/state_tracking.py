"""Transactional, optimizer-facing excited-state selection primitives.

The scan-analysis code in :mod:`excited_state_diabatizer.tracking` assigns many
tracks at once.  Geometry optimization has a different requirement: several
trial geometries may be inspected, but none of those inspections may advance
the electronic reference until one trial is explicitly accepted.  This module
provides that small state machine while deliberately leaving all electronic
structure and overlap calculations to callers.

The public data objects are immutable.  In particular, ``ElectronicSnapshot``
copies its coordinates and marks the copy read-only, so an optimizer cannot
silently change the geometry represented by a stored electronic state.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .models import HARTREE_TO_EV

if TYPE_CHECKING:
    from .models import StepData


MetadataValue = Union[str, int, float, bool, None]
NormValues = Union[float, Sequence[float], np.ndarray]


class SelectionStatus(str, Enum):
    """Outcome of inspecting a trial geometry.

    ``ACCEPT`` is the only status that :class:`TrackingSession` will commit.
    ``MANIFOLD`` denotes a meaningful but non-unique match, ``RETRY`` asks the
    optimizer to propose another geometry/root window, and ``HALT`` reports an
    invalid survey that should not be retried without correcting its data.
    """

    ACCEPT = "ACCEPT"
    MANIFOLD = "MANIFOLD"
    RETRY = "RETRY"
    HALT = "HALT"


def _readonly_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if array.ndim not in (1, 2):
        raise ValueError("Snapshot coordinates must be a flat vector or a two-dimensional array.")
    if array.ndim == 2 and array.shape[1] != 3:
        raise ValueError("Two-dimensional snapshot coordinates must have shape (n_atoms, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError("Snapshot coordinates contain non-finite values.")
    # ``array.setflags(write=False)`` alone is reversible when the array owns
    # its allocation.  Back the published view with immutable ``bytes`` so a
    # caller cannot re-enable writes and mutate a committed geometry in place.
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _readonly_numeric_array(
    values: np.ndarray,
    *,
    ndim: int,
    field_name: str,
) -> np.ndarray:
    """Copy a numerical array into immutable byte-backed storage."""

    array = np.array(values, dtype=float, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{field_name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains non-finite values.")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _norm_vector(values: NormValues, size: int, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{field_name} must be a scalar or have shape ({size},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains non-finite values.")
    if np.any(array <= 0.0):
        raise ValueError(f"{field_name} must contain only positive values.")
    return array


def _root_tuple(roots: Iterable[int], field_name: str) -> Tuple[int, ...]:
    result = tuple(int(root) for root in roots)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} contains duplicate roots.")
    return result


def _root_float_mapping(values: Mapping[int, float], field_name: str) -> Mapping[int, float]:
    result: Dict[int, float] = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{key}] is not finite.")
        result[int(key)] = number
    return MappingProxyType(result)


def _root_int_mapping(values: Mapping[int, int]) -> Mapping[int, int]:
    return MappingProxyType({int(key): int(value) for key, value in values.items()})


def _metadata_mapping(values: Mapping[str, MetadataValue]) -> Mapping[str, MetadataValue]:
    result: Dict[str, MetadataValue] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("Snapshot metadata keys must be strings.")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Unsupported metadata value for {key!r}: {type(value).__name__}.")
        result[key] = value
    return MappingProxyType(result)


def normalize_signed_overlap_block(
    overlaps: np.ndarray,
    reference_norms: NormValues = 1.0,
    candidate_norms: NormValues = 1.0,
    *,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Normalize a signed root-to-root overlap block without discarding phase.

    ``overlaps[i, j]`` is divided by the geometric mean of the corresponding
    reference and candidate self-overlap norms.  A small numerical excursion
    outside ``[-1, 1]`` is clipped; a larger one is rejected as an inconsistent
    overlap/norm combination.  The returned matrix is an immutable copy.

    This operation only normalizes individual roots.  Principal angles also
    require the roots within each set to be mutually orthonormal.  Call
    :func:`analyze_subspace_continuity` with self-overlap Gram matrices when
    that assumption is not justified by the overlap metric.
    """

    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative.")
    block = np.asarray(overlaps, dtype=float)
    if block.ndim != 2 or 0 in block.shape:
        raise ValueError("overlaps must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(block)):
        raise ValueError("overlaps contains non-finite values.")
    ref_norms = _norm_vector(reference_norms, block.shape[0], "reference_norms")
    cand_norms = _norm_vector(candidate_norms, block.shape[1], "candidate_norms")
    normalized = block / np.sqrt(np.outer(ref_norms, cand_norms))
    largest = float(np.max(np.abs(normalized)))
    if largest > 1.0 + tolerance:
        raise ValueError(f"A normalized root overlap exceeds unity ({largest:.8f}).")
    normalized = np.clip(normalized, -1.0, 1.0)
    return _readonly_numeric_array(
        normalized,
        ndim=2,
        field_name="normalized_signed_overlaps",
    )


def _normalized_gram_inverse_sqrt(
    gram: Optional[np.ndarray],
    norms: np.ndarray,
    *,
    field_name: str,
    tolerance: float,
) -> Tuple[np.ndarray, bool]:
    """Return the inverse square root of a root self-overlap Gram matrix."""

    size = len(norms)
    if gram is None:
        return np.eye(size), False
    matrix = np.asarray(gram, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{field_name} must have shape ({size}, {size}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} contains non-finite values.")
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0.0):
        raise ValueError(f"{field_name} must be symmetric.")
    if not np.allclose(np.diag(matrix), norms, atol=tolerance, rtol=tolerance):
        raise ValueError(f"The diagonal of {field_name} does not match the supplied norms.")

    normalized = matrix / np.sqrt(np.outer(norms, norms))
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    if float(np.min(eigenvalues)) <= tolerance:
        raise ValueError(f"{field_name} is singular or linearly dependent.")
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return inverse_sqrt, True


@dataclass(frozen=True, eq=False)
class SubspaceContinuity:
    """Immutable SVD diagnostics for two root manifolds.

    Singular values are cosines of the principal angles after optional Gram
    orthonormalization.  Therefore a rotating, phase-flipping, or permuted
    representation of the same manifold has singular values close to one even
    when no individual root has a unique pairwise match.
    """

    reference_roots: Tuple[int, ...]
    candidate_roots: Tuple[int, ...]
    normalized_signed_overlaps: np.ndarray = field(repr=False)
    orthonormalized_signed_overlaps: np.ndarray = field(repr=False)
    singular_values: np.ndarray
    principal_angles_rad: np.ndarray
    used_reference_gram: bool = False
    used_candidate_gram: bool = False

    def __post_init__(self) -> None:
        reference_roots = _root_tuple(self.reference_roots, "reference_roots")
        candidate_roots = _root_tuple(self.candidate_roots, "candidate_roots")
        if not reference_roots or not candidate_roots:
            raise ValueError("A subspace continuity result requires roots on both sides.")
        shape = (len(reference_roots), len(candidate_roots))
        normalized = _readonly_numeric_array(
            self.normalized_signed_overlaps,
            ndim=2,
            field_name="normalized_signed_overlaps",
        )
        orthonormalized = _readonly_numeric_array(
            self.orthonormalized_signed_overlaps,
            ndim=2,
            field_name="orthonormalized_signed_overlaps",
        )
        if normalized.shape != shape or orthonormalized.shape != shape:
            raise ValueError(f"Subspace overlap matrices must have shape {shape}.")
        singular_values = _readonly_numeric_array(
            self.singular_values,
            ndim=1,
            field_name="singular_values",
        )
        angles = _readonly_numeric_array(
            self.principal_angles_rad,
            ndim=1,
            field_name="principal_angles_rad",
        )
        expected = min(shape)
        if singular_values.shape != (expected,) or angles.shape != (expected,):
            raise ValueError(f"Subspace score vectors must have shape ({expected},).")
        if np.any(singular_values < 0.0) or np.any(singular_values > 1.0):
            raise ValueError("Subspace singular values must lie between zero and one.")
        if np.any(angles < 0.0) or np.any(angles > math.pi / 2.0):
            raise ValueError("Principal angles must lie between zero and pi/2.")

        object.__setattr__(self, "reference_roots", reference_roots)
        object.__setattr__(self, "candidate_roots", candidate_roots)
        object.__setattr__(self, "normalized_signed_overlaps", normalized)
        object.__setattr__(self, "orthonormalized_signed_overlaps", orthonormalized)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "principal_angles_rad", angles)
        object.__setattr__(self, "used_reference_gram", bool(self.used_reference_gram))
        object.__setattr__(self, "used_candidate_gram", bool(self.used_candidate_gram))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubspaceContinuity):
            return NotImplemented
        return (
            self.reference_roots == other.reference_roots
            and self.candidate_roots == other.candidate_roots
            and self.used_reference_gram == other.used_reference_gram
            and self.used_candidate_gram == other.used_candidate_gram
            and np.array_equal(self.normalized_signed_overlaps, other.normalized_signed_overlaps)
            and np.array_equal(
                self.orthonormalized_signed_overlaps,
                other.orthonormalized_signed_overlaps,
            )
            and np.array_equal(self.singular_values, other.singular_values)
            and np.array_equal(self.principal_angles_rad, other.principal_angles_rad)
        )

    __hash__ = None

    @property
    def dimension_match(self) -> bool:
        return len(self.reference_roots) == len(self.candidate_roots)

    @property
    def minimum_singular_value(self) -> float:
        """Worst-preserved principal direction (the conservative score)."""

        return float(np.min(self.singular_values))

    @property
    def rms_singular_value(self) -> float:
        """Root-mean-square preservation over all principal directions."""

        return float(np.sqrt(np.mean(self.singular_values**2)))

    @property
    def maximum_principal_angle_rad(self) -> float:
        return float(np.max(self.principal_angles_rad))

    @property
    def maximum_principal_angle_deg(self) -> float:
        return math.degrees(self.maximum_principal_angle_rad)


def analyze_subspace_continuity(
    overlaps: np.ndarray,
    reference_roots: Sequence[int],
    candidate_roots: Sequence[int],
    *,
    reference_norms: Optional[NormValues] = None,
    candidate_norms: Optional[NormValues] = None,
    reference_gram: Optional[np.ndarray] = None,
    candidate_gram: Optional[np.ndarray] = None,
    normalization_tolerance: float = 1.0e-6,
    linear_dependence_tolerance: float = 1.0e-10,
) -> SubspaceContinuity:
    """Calculate phase- and root-rotation-invariant manifold continuity.

    The signed cross-overlap block is first normalized by root self-overlap
    norms.  When within-geometry Gram matrices are supplied, both root sets are
    symmetrically orthonormalized before the SVD.  The resulting singular
    values are the cosines of the principal angles between the two subspaces.

    If Gram matrices are omitted, roots within each set are assumed mutually
    orthogonal in the chosen overlap metric.  This is appropriate only when
    guaranteed by that metric; transition-density Frobenius overlaps generally
    need explicit Gram matrices for rigorous principal angles.
    """

    if normalization_tolerance < 0.0:
        raise ValueError("normalization_tolerance cannot be negative.")
    if linear_dependence_tolerance < 0.0:
        raise ValueError("linear_dependence_tolerance cannot be negative.")
    ref_roots = _root_tuple(reference_roots, "reference_roots")
    cand_roots = _root_tuple(candidate_roots, "candidate_roots")
    block = np.asarray(overlaps, dtype=float)
    expected_shape = (len(ref_roots), len(cand_roots))
    if not ref_roots or not cand_roots or block.shape != expected_shape:
        raise ValueError(f"overlaps must have shape {expected_shape} for the supplied roots.")

    if reference_norms is None:
        reference_norms = np.diag(reference_gram) if reference_gram is not None else 1.0
    if candidate_norms is None:
        candidate_norms = np.diag(candidate_gram) if candidate_gram is not None else 1.0
    ref_norms = _norm_vector(reference_norms, len(ref_roots), "reference_norms")
    cand_norms = _norm_vector(candidate_norms, len(cand_roots), "candidate_norms")
    normalized = normalize_signed_overlap_block(
        block,
        ref_norms,
        cand_norms,
        tolerance=normalization_tolerance,
    )
    ref_inverse_sqrt, used_ref_gram = _normalized_gram_inverse_sqrt(
        reference_gram,
        ref_norms,
        field_name="reference_gram",
        tolerance=linear_dependence_tolerance,
    )
    cand_inverse_sqrt, used_cand_gram = _normalized_gram_inverse_sqrt(
        candidate_gram,
        cand_norms,
        field_name="candidate_gram",
        tolerance=linear_dependence_tolerance,
    )
    orthonormalized = ref_inverse_sqrt @ normalized @ cand_inverse_sqrt
    singular_values = np.linalg.svd(orthonormalized, compute_uv=False)
    largest = float(np.max(singular_values))
    if largest > 1.0 + normalization_tolerance:
        raise ValueError(f"A subspace singular value exceeds unity ({largest:.8f}).")
    singular_values = np.clip(singular_values, 0.0, 1.0)
    principal_angles = np.arccos(singular_values)
    return SubspaceContinuity(
        reference_roots=ref_roots,
        candidate_roots=cand_roots,
        normalized_signed_overlaps=normalized,
        orthonormalized_signed_overlaps=orthonormalized,
        singular_values=singular_values,
        principal_angles_rad=principal_angles,
        used_reference_gram=used_ref_gram,
        used_candidate_gram=used_cand_gram,
    )


@dataclass(frozen=True, eq=False)
class RootOverlapBlock:
    """Complete signed overlaps and self-overlap Grams for two root windows.

    Unlike :class:`SubspaceContinuity`, this object does not choose a manifold.
    It stores the full production result so the selector can first detect a
    near-degenerate subset using energies and scalar root scores, then analyze
    exactly the corresponding rows and columns.  All matrices are copied into
    immutable byte-backed storage.
    """

    reference_roots: Tuple[int, ...]
    candidate_roots: Tuple[int, ...]
    overlaps: np.ndarray = field(repr=False)
    reference_gram: np.ndarray = field(repr=False)
    candidate_gram: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        reference_roots = _root_tuple(self.reference_roots, "reference_roots")
        candidate_roots = _root_tuple(self.candidate_roots, "candidate_roots")
        if not reference_roots or not candidate_roots:
            raise ValueError("A root overlap block requires roots on both sides.")
        overlap_shape = (len(reference_roots), len(candidate_roots))
        reference_shape = (len(reference_roots), len(reference_roots))
        candidate_shape = (len(candidate_roots), len(candidate_roots))
        overlaps = _readonly_numeric_array(self.overlaps, ndim=2, field_name="overlaps")
        reference_gram = _readonly_numeric_array(
            self.reference_gram,
            ndim=2,
            field_name="reference_gram",
        )
        candidate_gram = _readonly_numeric_array(
            self.candidate_gram,
            ndim=2,
            field_name="candidate_gram",
        )
        if overlaps.shape != overlap_shape:
            raise ValueError(f"overlaps must have shape {overlap_shape}.")
        if reference_gram.shape != reference_shape:
            raise ValueError(f"reference_gram must have shape {reference_shape}.")
        if candidate_gram.shape != candidate_shape:
            raise ValueError(f"candidate_gram must have shape {candidate_shape}.")

        # Validate symmetry, positive definiteness, root norms, and the full
        # block Cauchy/principal-angle bounds before publishing the object.
        analyze_subspace_continuity(
            overlaps,
            reference_roots,
            candidate_roots,
            reference_gram=reference_gram,
            candidate_gram=candidate_gram,
        )
        object.__setattr__(self, "reference_roots", reference_roots)
        object.__setattr__(self, "candidate_roots", candidate_roots)
        object.__setattr__(self, "overlaps", overlaps)
        object.__setattr__(self, "reference_gram", reference_gram)
        object.__setattr__(self, "candidate_gram", candidate_gram)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RootOverlapBlock):
            return NotImplemented
        return (
            self.reference_roots == other.reference_roots
            and self.candidate_roots == other.candidate_roots
            and np.array_equal(self.overlaps, other.overlaps)
            and np.array_equal(self.reference_gram, other.reference_gram)
            and np.array_equal(self.candidate_gram, other.candidate_gram)
        )

    __hash__ = None

    def subset(
        self,
        reference_roots: Sequence[int],
        candidate_roots: Sequence[int],
    ) -> "RootOverlapBlock":
        """Return an immutable block restricted to the requested root order."""

        ref_roots = _root_tuple(reference_roots, "reference_roots")
        cand_roots = _root_tuple(candidate_roots, "candidate_roots")
        if not ref_roots or not cand_roots:
            raise ValueError("A root overlap subset requires roots on both sides.")
        ref_index = {root: index for index, root in enumerate(self.reference_roots)}
        cand_index = {root: index for index, root in enumerate(self.candidate_roots)}
        missing_reference = tuple(root for root in ref_roots if root not in ref_index)
        missing_candidate = tuple(root for root in cand_roots if root not in cand_index)
        if missing_reference:
            raise ValueError(f"Reference roots are absent from the full block: {missing_reference}.")
        if missing_candidate:
            raise ValueError(f"Candidate roots are absent from the full block: {missing_candidate}.")
        ref_indices = [ref_index[root] for root in ref_roots]
        cand_indices = [cand_index[root] for root in cand_roots]
        return RootOverlapBlock(
            reference_roots=ref_roots,
            candidate_roots=cand_roots,
            overlaps=self.overlaps[np.ix_(ref_indices, cand_indices)],
            reference_gram=self.reference_gram[np.ix_(ref_indices, ref_indices)],
            candidate_gram=self.candidate_gram[np.ix_(cand_indices, cand_indices)],
        )

    def analyze(
        self,
        reference_roots: Sequence[int],
        candidate_roots: Sequence[int],
        *,
        normalization_tolerance: float = 1.0e-6,
        linear_dependence_tolerance: float = 1.0e-10,
    ) -> SubspaceContinuity:
        """Calculate principal-angle continuity for one selected sub-block."""

        selected = self.subset(reference_roots, candidate_roots)
        return analyze_subspace_continuity(
            selected.overlaps,
            selected.reference_roots,
            selected.candidate_roots,
            reference_gram=selected.reference_gram,
            candidate_gram=selected.candidate_gram,
            normalization_tolerance=normalization_tolerance,
            linear_dependence_tolerance=linear_dependence_tolerance,
        )


@dataclass(frozen=True, eq=False)
class ElectronicSnapshot:
    """Immutable electronic data associated with exactly one geometry.

    ``roots`` are the roots actually parsed.  ``requested_roots`` describes the
    root window that the electronic-structure job was expected to return; an
    empty tuple means that completeness is unknown.  Energies and spin data are
    optional because overlap-only surveys are useful and should remain valid.
    """

    label: str
    coordinates: np.ndarray = field(repr=False)
    roots: Tuple[int, ...]
    selected_root: Optional[int] = None
    requested_roots: Tuple[int, ...] = ()
    energies_eh: Mapping[int, float] = field(default_factory=dict)
    excitation_energies_ev: Mapping[int, float] = field(default_factory=dict)
    multiplicities: Mapping[int, int] = field(default_factory=dict)
    spin_squared: Mapping[int, float] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("An electronic snapshot requires a non-empty label.")
        coordinates = _readonly_array(self.coordinates)
        roots = _root_tuple(self.roots, "roots")
        requested = _root_tuple(self.requested_roots, "requested_roots")
        if self.selected_root is not None and int(self.selected_root) not in roots:
            raise ValueError(f"Selected root {self.selected_root} is not present in snapshot {self.label!r}.")

        root_set = set(roots)
        mappings = {
            "energies_eh": self.energies_eh,
            "excitation_energies_ev": self.excitation_energies_ev,
            "multiplicities": self.multiplicities,
            "spin_squared": self.spin_squared,
        }
        for name, values in mappings.items():
            extra = set(int(root) for root in values) - root_set
            if extra:
                raise ValueError(f"{name} contains roots not present in the snapshot: {sorted(extra)}.")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "requested_roots", requested)
        object.__setattr__(self, "selected_root", None if self.selected_root is None else int(self.selected_root))
        object.__setattr__(self, "energies_eh", _root_float_mapping(self.energies_eh, "energies_eh"))
        object.__setattr__(
            self,
            "excitation_energies_ev",
            _root_float_mapping(self.excitation_energies_ev, "excitation_energies_ev"),
        )
        object.__setattr__(self, "multiplicities", _root_int_mapping(self.multiplicities))
        object.__setattr__(self, "spin_squared", _root_float_mapping(self.spin_squared, "spin_squared"))
        object.__setattr__(self, "artifacts", MappingProxyType({str(k): Path(v) for k, v in self.artifacts.items()}))
        object.__setattr__(self, "metadata", _metadata_mapping(self.metadata))

    @property
    def missing_roots(self) -> Tuple[int, ...]:
        """Requested roots that were not returned by the calculation."""

        available = set(self.roots)
        return tuple(root for root in self.requested_roots if root not in available)

    @property
    def root_window_complete(self) -> Optional[bool]:
        """Whether all requested roots are present, or ``None`` if unknown."""

        if not self.requested_roots:
            return None
        return not self.missing_roots

    def with_selected_root(self, root: int) -> "ElectronicSnapshot":
        """Return a new snapshot selecting ``root``; the original is unchanged."""

        return replace(self, selected_root=int(root))

    @classmethod
    def from_step_data(
        cls,
        step: "StepData",
        coordinates: np.ndarray,
        *,
        selected_root: Optional[int] = None,
        requested_roots: Sequence[int] = (),
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ) -> "ElectronicSnapshot":
        """Adapt the existing offline :class:`~.models.StepData` model.

        This keeps the new optimizer API additive; the scan CLI and its mutable
        ``StepData`` records do not need to change.
        """

        roots = tuple(step.states)
        energies = {
            root: state.abs_energy_eh
            for root, state in step.states.items()
            if state.abs_energy_eh is not None
        }
        excitations = {
            root: state.exc_ev for root, state in step.states.items() if state.exc_ev is not None
        }
        multiplicities = {
            root: state.multiplicity
            for root, state in step.states.items()
            if state.multiplicity is not None
        }
        spin_squared = {root: state.s2 for root, state in step.states.items() if state.s2 is not None}
        artifacts = {
            "output": step.job.out_path,
            "geometry": step.job.geom_path,
        }
        if step.job.gbw_path is not None:
            artifacts["gbw"] = step.job.gbw_path
        if step.json_file is not None:
            artifacts["json"] = step.json_file
        return cls(
            label=step.job.label,
            coordinates=coordinates,
            roots=roots,
            selected_root=selected_root,
            requested_roots=tuple(requested_roots),
            energies_eh=energies,
            excitation_energies_ev=excitations,
            multiplicities=multiplicities,
            spin_squared=spin_squared,
            artifacts=artifacts,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class SelectionConfig:
    """Thresholds for one-root optimizer decisions.

    Energy descent is deliberately opt-in: quasi-Newton optimizers can accept a
    small uphill step under their own trust criterion.  Set
    ``max_energy_increase_eh=0.0`` to require non-increasing state energy, or a
    positive value to allow a specified tolerance.
    """

    min_score: float = 0.65
    min_margin: float = 0.10
    manifold_gap_ev: float = 0.10
    require_complete_root_window: bool = True
    require_energy: bool = False
    max_energy_increase_eh: Optional[float] = None
    require_same_multiplicity: bool = True
    manifold_when_energy_unavailable: bool = True
    normalization_tolerance: float = 1.0e-6
    min_subspace_singular_value: Optional[float] = None
    require_equal_subspace_dimensions: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must lie between zero and one.")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("min_margin must lie between zero and one.")
        if self.manifold_gap_ev < 0.0:
            raise ValueError("manifold_gap_ev cannot be negative.")
        if self.max_energy_increase_eh is not None and self.max_energy_increase_eh < 0.0:
            raise ValueError("max_energy_increase_eh cannot be negative.")
        if self.normalization_tolerance < 0.0:
            raise ValueError("normalization_tolerance cannot be negative.")
        if (
            self.min_subspace_singular_value is not None
            and not 0.0 <= self.min_subspace_singular_value <= 1.0
        ):
            raise ValueError("min_subspace_singular_value must lie between zero and one.")

    @classmethod
    def from_tracking_config(cls, config: object, **overrides: object) -> "SelectionConfig":
        """Create optimizer thresholds from the existing scan ``TrackingConfig``.

        ``config`` is intentionally duck-typed to avoid coupling the two
        tracking modules through an import cycle.
        """

        values = {
            "min_score": float(getattr(config, "similarity_threshold_good")),
            "min_margin": float(getattr(config, "assignment_margin_threshold")),
            "manifold_gap_ev": float(getattr(config, "subspace_gap_ev")),
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, eq=False)
class StateSurvey:
    """Read-only result of evaluating one uncommitted trial geometry."""

    survey_id: str
    generation: int
    reference_label: str
    candidate: ElectronicSnapshot
    overlaps: Mapping[int, float]
    reference_norm: float = 1.0
    candidate_norms: Mapping[int, float] = field(default_factory=dict)
    step_scale: float = 1.0
    subspace_continuity: Optional[SubspaceContinuity] = None
    root_overlap_block: Optional[RootOverlapBlock] = None

    def __post_init__(self) -> None:
        if not self.survey_id:
            raise ValueError("A survey requires a non-empty identifier.")
        if self.generation < 0:
            raise ValueError("Survey generation cannot be negative.")
        if not self.reference_label:
            raise ValueError("A survey requires a reference label.")
        if not math.isfinite(float(self.reference_norm)):
            raise ValueError("Survey reference norm is not finite.")
        if not math.isfinite(float(self.step_scale)) or self.step_scale <= 0.0:
            raise ValueError("Survey step_scale must be finite and positive.")
        overlaps = _root_float_mapping(self.overlaps, "overlaps")
        norms = _root_float_mapping(self.candidate_norms, "candidate_norms")
        unknown = set(overlaps) - set(self.candidate.roots)
        if unknown:
            raise ValueError(f"Survey overlaps contain roots absent from the candidate: {sorted(unknown)}.")
        unknown_norms = set(norms) - set(self.candidate.roots)
        if unknown_norms:
            raise ValueError(f"Survey norms contain roots absent from the candidate: {sorted(unknown_norms)}.")
        if norms:
            missing_norms = set(overlaps) - set(norms)
            if missing_norms:
                raise ValueError(f"Survey norms are missing scored roots: {sorted(missing_norms)}.")
        if self.subspace_continuity is not None:
            if not isinstance(self.subspace_continuity, SubspaceContinuity):
                raise TypeError("subspace_continuity must be a SubspaceContinuity result.")
            unknown_subspace_roots = set(self.subspace_continuity.candidate_roots) - set(
                self.candidate.roots
            )
            if unknown_subspace_roots:
                raise ValueError(
                    "Subspace continuity contains roots absent from the candidate: "
                    f"{sorted(unknown_subspace_roots)}."
                )
        if self.root_overlap_block is not None:
            if not isinstance(self.root_overlap_block, RootOverlapBlock):
                raise TypeError("root_overlap_block must be a RootOverlapBlock.")
            unknown_block_roots = set(self.root_overlap_block.candidate_roots) - set(
                self.candidate.roots
            )
            if unknown_block_roots:
                raise ValueError(
                    "Root overlap block contains roots absent from the candidate: "
                    f"{sorted(unknown_block_roots)}."
                )
        object.__setattr__(self, "overlaps", overlaps)
        object.__setattr__(self, "candidate_norms", norms)
        object.__setattr__(self, "reference_norm", float(self.reference_norm))
        object.__setattr__(self, "step_scale", float(self.step_scale))


@dataclass(frozen=True)
class SelectionDecision:
    """Explicit, auditable result from :func:`select_state`."""

    status: SelectionStatus
    survey_id: str
    generation: int
    reference_label: str
    candidate_label: str
    selected_root: Optional[int]
    best_score: float
    second_root: Optional[int]
    second_score: float
    margin: float
    normalized_scores: Mapping[int, float]
    missing_roots: Tuple[int, ...] = ()
    manifold_roots: Tuple[int, ...] = ()
    selected_energy_eh: Optional[float] = None
    energy_change_eh: Optional[float] = None
    competitor_gap_ev: Optional[float] = None
    reason: str = ""
    signed_normalized_scores: Mapping[int, float] = field(default_factory=dict)
    subspace_continuity: Optional[SubspaceContinuity] = None
    subspace_continuous: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SelectionStatus(self.status))
        object.__setattr__(self, "normalized_scores", _root_float_mapping(self.normalized_scores, "normalized_scores"))
        object.__setattr__(
            self,
            "signed_normalized_scores",
            _root_float_mapping(self.signed_normalized_scores, "signed_normalized_scores"),
        )
        object.__setattr__(self, "missing_roots", tuple(int(root) for root in self.missing_roots))
        object.__setattr__(self, "manifold_roots", tuple(int(root) for root in self.manifold_roots))
        if self.subspace_continuity is not None and not isinstance(
            self.subspace_continuity,
            SubspaceContinuity,
        ):
            raise TypeError("subspace_continuity must be a SubspaceContinuity result.")
        if self.subspace_continuous is not None:
            object.__setattr__(self, "subspace_continuous", bool(self.subspace_continuous))

    @property
    def accepted(self) -> bool:
        return self.status is SelectionStatus.ACCEPT


def _decision(
    survey: StateSurvey,
    status: SelectionStatus,
    reason: str,
    *,
    selected_root: Optional[int] = None,
    best_score: float = 0.0,
    second_root: Optional[int] = None,
    second_score: float = 0.0,
    scores: Optional[Mapping[int, float]] = None,
    signed_scores: Optional[Mapping[int, float]] = None,
    missing_roots: Sequence[int] = (),
    manifold_roots: Sequence[int] = (),
    selected_energy_eh: Optional[float] = None,
    energy_change_eh: Optional[float] = None,
    competitor_gap_ev: Optional[float] = None,
    subspace_continuous: Optional[bool] = None,
    subspace_continuity: Optional[SubspaceContinuity] = None,
) -> SelectionDecision:
    return SelectionDecision(
        status=status,
        survey_id=survey.survey_id,
        generation=survey.generation,
        reference_label=survey.reference_label,
        candidate_label=survey.candidate.label,
        selected_root=selected_root,
        best_score=float(best_score),
        second_root=second_root,
        second_score=float(second_score),
        margin=float(best_score - second_score),
        normalized_scores={} if scores is None else scores,
        missing_roots=tuple(missing_roots),
        manifold_roots=tuple(manifold_roots),
        selected_energy_eh=selected_energy_eh,
        energy_change_eh=energy_change_eh,
        competitor_gap_ev=competitor_gap_ev,
        reason=reason,
        signed_normalized_scores={} if signed_scores is None else signed_scores,
        subspace_continuity=(
            survey.subspace_continuity
            if subspace_continuity is None
            else subspace_continuity
        ),
        subspace_continuous=subspace_continuous,
    )


def _normalized_scores(
    survey: StateSurvey,
    config: SelectionConfig,
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]], str]:
    if survey.reference_norm <= 0.0:
        return None, None, "reference self-overlap norm must be positive"
    scores: Dict[int, float] = {}
    signed_scores: Dict[int, float] = {}
    for root, overlap in survey.overlaps.items():
        candidate_norm = survey.candidate_norms.get(root, 1.0)
        if candidate_norm <= 0.0:
            return None, None, f"candidate self-overlap norm for root {root} must be positive"
        signed_score = overlap / math.sqrt(survey.reference_norm * candidate_norm)
        if not math.isfinite(signed_score):
            return None, None, f"normalized score for root {root} is not finite"
        if abs(signed_score) > 1.0 + config.normalization_tolerance:
            return (
                None,
                None,
                f"normalized score for root {root} exceeds unity ({abs(signed_score):.8f})",
            )
        signed_score = max(-1.0, min(1.0, signed_score))
        signed_scores[root] = signed_score
        scores[root] = abs(signed_score)
    return scores, signed_scores, ""


def _energy_gap_ev(snapshot: ElectronicSnapshot, root_a: int, root_b: int) -> Optional[float]:
    if root_a in snapshot.excitation_energies_ev and root_b in snapshot.excitation_energies_ev:
        return abs(snapshot.excitation_energies_ev[root_a] - snapshot.excitation_energies_ev[root_b])
    if root_a in snapshot.energies_eh and root_b in snapshot.energies_eh:
        return abs(snapshot.energies_eh[root_a] - snapshot.energies_eh[root_b]) * HARTREE_TO_EV
    return None


def _reference_manifold_roots(
    reference: ElectronicSnapshot,
    config: SelectionConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Find the energy-local, multiplicity-compatible reference manifold.

    The second tuple reports otherwise compatible roots whose energy gap to the
    selected root is unavailable.  They make an equal-dimension manifold
    assignment underdetermined and are handled as ``RETRY`` by the selector.
    """

    if reference.selected_root is None:
        return (), ()
    selected = reference.selected_root
    target_mult = reference.multiplicities.get(selected)
    roots = []
    unknown_energy = []
    for root in reference.roots:
        if root == selected:
            roots.append(root)
            continue
        if config.require_same_multiplicity and target_mult is not None:
            # Fail closed: absent multiplicity metadata is not evidence of a
            # matching spin manifold.
            if reference.multiplicities.get(root) != target_mult:
                continue
        gap = _energy_gap_ev(reference, selected, root)
        if gap is None:
            unknown_energy.append(root)
        elif gap <= config.manifold_gap_ev:
            roots.append(root)
    return tuple(roots), tuple(unknown_energy)


def select_state(
    reference: ElectronicSnapshot,
    survey: StateSurvey,
    config: Optional[SelectionConfig] = None,
) -> SelectionDecision:
    """Select a candidate root without mutating either snapshot or the survey."""

    config = SelectionConfig() if config is None else config
    if reference.selected_root is None:
        return _decision(survey, SelectionStatus.HALT, "reference snapshot has no selected root")
    if survey.reference_label != reference.label:
        return _decision(
            survey,
            SelectionStatus.HALT,
            f"survey was evaluated against {survey.reference_label!r}, not {reference.label!r}",
        )

    continuity = survey.subspace_continuity
    root_overlap_block = survey.root_overlap_block
    if continuity is not None:
        unknown_reference_roots = set(continuity.reference_roots) - set(reference.roots)
        if unknown_reference_roots:
            return _decision(
                survey,
                SelectionStatus.HALT,
                "subspace continuity contains roots absent from the reference: "
                f"{sorted(unknown_reference_roots)}",
            )
    if root_overlap_block is not None:
        unknown_block_roots = set(root_overlap_block.reference_roots) - set(reference.roots)
        if unknown_block_roots:
            return _decision(
                survey,
                SelectionStatus.HALT,
                "root overlap block contains roots absent from the reference: "
                f"{sorted(unknown_block_roots)}",
            )

    scores, signed_scores, normalization_error = _normalized_scores(survey, config)
    if scores is None:
        return _decision(survey, SelectionStatus.HALT, normalization_error)

    score_missing = tuple(root for root in survey.candidate.roots if root not in scores)
    missing_roots = tuple(dict.fromkeys((*survey.candidate.missing_roots, *score_missing)))
    if config.require_complete_root_window and missing_roots:
        return _decision(
            survey,
            SelectionStatus.RETRY,
            "requested root window is incomplete; rerun with all requested roots",
            scores=scores,
            signed_scores=signed_scores,
            missing_roots=missing_roots,
        )
    if not scores:
        return _decision(survey, SelectionStatus.HALT, "survey contains no root-overlap scores")

    eligible = list(scores)
    if config.require_same_multiplicity:
        target_mult = reference.multiplicities.get(reference.selected_root)
        if target_mult is not None:
            eligible = [
                root
                for root in eligible
                if survey.candidate.multiplicities.get(root) == target_mult
            ]
            if not eligible:
                return _decision(
                    survey,
                    SelectionStatus.RETRY,
                    f"candidate root window contains no root with multiplicity {target_mult}",
                    scores=scores,
                    signed_scores=signed_scores,
                )

    ranked = sorted(eligible, key=lambda root: (-scores[root], root))
    best_root = ranked[0]
    second_root = ranked[1] if len(ranked) > 1 else None
    best_score = scores[best_root]
    second_score = scores[second_root] if second_root is not None else 0.0
    candidate_energy = survey.candidate.energies_eh.get(best_root)
    reference_energy = reference.energies_eh.get(reference.selected_root)
    energy_change = (
        candidate_energy - reference_energy
        if candidate_energy is not None and reference_energy is not None
        else None
    )
    competitor_gap = (
        _energy_gap_ev(survey.candidate, best_root, second_root)
        if second_root is not None
        else None
    )

    common = dict(
        selected_root=best_root,
        best_score=best_score,
        second_root=second_root,
        second_score=second_score,
        scores=scores,
        signed_scores=signed_scores,
        selected_energy_eh=candidate_energy,
        energy_change_eh=energy_change,
        competitor_gap_ev=competitor_gap,
    )
    if best_score < config.min_score:
        return _decision(
            survey,
            SelectionStatus.RETRY,
            f"best normalized similarity {best_score:.3f} is below {config.min_score:.3f}",
            **common,
        )

    margin = best_score - second_score
    if second_root is not None and margin < config.min_margin:
        close_in_energy = competitor_gap is None or competitor_gap <= config.manifold_gap_ev
        if competitor_gap is None and not config.manifold_when_energy_unavailable:
            close_in_energy = False
        if close_in_energy:
            manifold_roots = tuple(
                root
                for root in ranked
                if best_score - scores[root] < config.min_margin
                and (
                    _energy_gap_ev(survey.candidate, best_root, root) is None
                    or _energy_gap_ev(survey.candidate, best_root, root) <= config.manifold_gap_ev
                )
            )
            gap_text = "unknown" if competitor_gap is None else f"{competitor_gap:.4f} eV"
            manifold_common = dict(common)
            # An ambiguous manifold has no unique selected root.  The leading
            # pairwise score remains available in ``normalized_scores`` and
            # ``best_score``, but cannot be mistaken for a committable choice.
            manifold_common["selected_root"] = None
            subspace_continuous: Optional[bool] = None
            derived_from_full_block = root_overlap_block is not None
            if root_overlap_block is not None:
                reference_manifold, unknown_reference_energies = _reference_manifold_roots(
                    reference,
                    config,
                )
                if (
                    config.require_equal_subspace_dimensions
                    and unknown_reference_energies
                ):
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        (
                            "reference manifold energies are unavailable for roots "
                            f"{unknown_reference_energies}; cannot determine an equal subspace"
                        ),
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                if (
                    config.require_equal_subspace_dimensions
                    and len(reference_manifold) != len(manifold_roots)
                ):
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        (
                            "reference and candidate manifold dimensions do not match "
                            f"({len(reference_manifold)} != {len(manifold_roots)}); "
                            "possible split/merge region"
                        ),
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                missing_reference_block = tuple(
                    root
                    for root in reference_manifold
                    if root not in root_overlap_block.reference_roots
                )
                missing_candidate_block = tuple(
                    root
                    for root in manifold_roots
                    if root not in root_overlap_block.candidate_roots
                )
                if missing_reference_block or missing_candidate_block:
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        (
                            "full root overlap block does not cover the detected manifold; "
                            f"missing reference={missing_reference_block}, "
                            f"candidate={missing_candidate_block}"
                        ),
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                try:
                    continuity = root_overlap_block.analyze(
                        reference_manifold,
                        manifold_roots,
                        normalization_tolerance=config.normalization_tolerance,
                    )
                except ValueError as exc:
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        f"could not analyze the detected manifold sub-block: {exc}",
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                manifold_common["subspace_continuity"] = continuity

            if config.min_subspace_singular_value is not None:
                if continuity is None:
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        "near-degenerate roots require a root-to-root subspace overlap block",
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                if config.require_equal_subspace_dimensions and not continuity.dimension_match:
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        "reference and candidate manifold dimensions do not match",
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                if not derived_from_full_block:
                    if set(continuity.candidate_roots) != set(manifold_roots):
                        return _decision(
                            survey,
                            SelectionStatus.RETRY,
                            "subspace candidate roots do not match the detected near-degenerate manifold",
                            manifold_roots=manifold_roots,
                            subspace_continuous=False,
                            **manifold_common,
                        )
                    if reference.selected_root not in continuity.reference_roots:
                        return _decision(
                            survey,
                            SelectionStatus.RETRY,
                            "the tracked reference root is absent from the supplied reference manifold",
                            manifold_roots=manifold_roots,
                            subspace_continuous=False,
                            **manifold_common,
                        )
                if any(root not in eligible for root in continuity.candidate_roots):
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        "the candidate manifold contains a multiplicity-incompatible root",
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                if (
                    continuity.minimum_singular_value
                    < config.min_subspace_singular_value
                ):
                    return _decision(
                        survey,
                        SelectionStatus.RETRY,
                        (
                            "weakest subspace singular value "
                            f"{continuity.minimum_singular_value:.3f} is below "
                            f"{config.min_subspace_singular_value:.3f}"
                        ),
                        manifold_roots=manifold_roots,
                        subspace_continuous=False,
                        **manifold_common,
                    )
                subspace_continuous = True
            return _decision(
                survey,
                SelectionStatus.MANIFOLD,
                (
                    f"overlap margin {margin:.3f} is below {config.min_margin:.3f}; "
                    f"competitor gap is {gap_text}"
                ),
                manifold_roots=manifold_roots,
                subspace_continuous=subspace_continuous,
                **manifold_common,
            )
        return _decision(
            survey,
            SelectionStatus.RETRY,
            (
                f"overlap margin {margin:.3f} is below {config.min_margin:.3f}, "
                f"but the competitor gap {competitor_gap:.4f} eV is not a near-degenerate manifold"
            ),
            **common,
        )

    if config.require_energy and energy_change is None:
        return _decision(
            survey,
            SelectionStatus.RETRY,
            "state-specific reference or candidate energy is unavailable",
            **common,
        )
    if (
        config.max_energy_increase_eh is not None
        and energy_change is not None
        and energy_change > config.max_energy_increase_eh
    ):
        return _decision(
            survey,
            SelectionStatus.RETRY,
            (
                f"state energy increased by {energy_change:.8f} Eh, exceeding the allowed "
                f"{config.max_energy_increase_eh:.8f} Eh"
            ),
            **common,
        )

    return _decision(
        survey,
        SelectionStatus.ACCEPT,
        "normalized similarity, assignment margin, root completeness, and available energy checks pass",
        **common,
    )


@dataclass(frozen=True)
class StateSelector:
    """A configuration-only, stateless callable selector."""

    config: SelectionConfig = field(default_factory=SelectionConfig)

    def select(self, reference: ElectronicSnapshot, survey: StateSurvey) -> SelectionDecision:
        return select_state(reference, survey, self.config)

    def __call__(self, reference: ElectronicSnapshot, survey: StateSurvey) -> SelectionDecision:
        return self.select(reference, survey)


class TrackingSession:
    """Transactional owner of the committed electronic reference.

    Calling :meth:`survey` only adds a pending immutable record.  :meth:`select`
    is read-only.  The committed snapshot and stable anchor move only when an
    ``ACCEPT`` decision is passed to :meth:`commit`; all unselected probes are
    then discarded atomically.
    """

    def __init__(
        self,
        initial: ElectronicSnapshot,
        selector: Optional[StateSelector] = None,
    ) -> None:
        if initial.selected_root is None:
            raise ValueError("A tracking session requires an initial selected root.")
        self._selector = StateSelector() if selector is None else selector
        self._committed = initial
        self._anchor = initial
        self._history = [initial]
        self._pending: Dict[str, StateSurvey] = {}
        self._generation = 0
        self._counter = 0
        self._lock = threading.RLock()

    @property
    def committed(self) -> ElectronicSnapshot:
        return self._committed

    @property
    def anchor(self) -> ElectronicSnapshot:
        return self._anchor

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def history(self) -> Tuple[ElectronicSnapshot, ...]:
        return tuple(self._history)

    @property
    def pending(self) -> Tuple[StateSurvey, ...]:
        return tuple(self._pending.values())

    def survey(
        self,
        candidate: ElectronicSnapshot,
        overlaps: Mapping[int, float],
        *,
        reference_norm: float = 1.0,
        candidate_norms: Optional[Mapping[int, float]] = None,
        step_scale: float = 1.0,
        against: str = "committed",
        survey_id: Optional[str] = None,
        subspace_continuity: Optional[SubspaceContinuity] = None,
        root_overlap_block: Optional[RootOverlapBlock] = None,
    ) -> StateSurvey:
        """Register an uncommitted trial evaluated against a stable reference."""

        with self._lock:
            if against == "committed":
                reference = self._committed
            elif against == "anchor":
                reference = self._anchor
            else:
                raise ValueError("against must be either 'committed' or 'anchor'.")
            if survey_id is None:
                self._counter += 1
                survey_id = f"survey-{self._counter:06d}"
            if survey_id in self._pending:
                raise ValueError(f"A pending survey already has identifier {survey_id!r}.")
            record = StateSurvey(
                survey_id=survey_id,
                generation=self._generation,
                reference_label=reference.label,
                candidate=candidate,
                overlaps=overlaps,
                reference_norm=reference_norm,
                candidate_norms={} if candidate_norms is None else candidate_norms,
                step_scale=step_scale,
                subspace_continuity=subspace_continuity,
                root_overlap_block=root_overlap_block,
            )
            self._pending[survey_id] = record
            return record

    def select(self, survey_id: Optional[str] = None) -> SelectionDecision:
        """Inspect one survey, or choose the best decision among all probes.

        Accepted probes are compared by selected state energy when all required
        energies are available, followed by overlap score, margin, and proximity
        to the unscaled optimizer proposal.  No session state changes here.
        """

        with self._lock:
            if survey_id is not None:
                survey = self._get_pending(survey_id)
                return self._select_survey(survey)
            if not self._pending:
                raise RuntimeError("There are no pending surveys to select.")
            decisions = [self._select_survey(survey) for survey in self._pending.values()]
            accepted = [decision for decision in decisions if decision.status is SelectionStatus.ACCEPT]
            if accepted:
                energies_known = all(decision.selected_energy_eh is not None for decision in accepted)

                def accepted_key(decision: SelectionDecision) -> tuple:
                    survey = self._pending[decision.survey_id]
                    energy = decision.selected_energy_eh if energies_known else 0.0
                    return (
                        energy,
                        -decision.best_score,
                        -decision.margin,
                        abs(survey.step_scale - 1.0),
                        decision.survey_id,
                    )

                return min(accepted, key=accepted_key)
            priority = {
                SelectionStatus.MANIFOLD: 0,
                SelectionStatus.RETRY: 1,
                SelectionStatus.HALT: 2,
            }
            return min(
                decisions,
                key=lambda decision: (
                    priority[decision.status],
                    -decision.best_score,
                    -decision.margin,
                    decision.survey_id,
                ),
            )

    def commit(
        self,
        decision: SelectionDecision,
        *,
        update_anchor: bool = True,
    ) -> ElectronicSnapshot:
        """Atomically accept one decision and discard every other trial."""

        with self._lock:
            if decision.status is not SelectionStatus.ACCEPT or decision.selected_root is None:
                raise ValueError("Only an ACCEPT decision with a selected root can be committed.")
            if decision.generation != self._generation:
                raise RuntimeError("Cannot commit a decision from an earlier session generation.")
            survey = self._get_pending(decision.survey_id)
            if (
                decision.reference_label != survey.reference_label
                or decision.candidate_label != survey.candidate.label
            ):
                raise RuntimeError("Decision labels do not match the pending survey.")
            # Re-evaluate to prevent a caller from fabricating or modifying the
            # decision that authorizes a commit.
            current = self._select_survey(survey)
            if current != decision:
                raise RuntimeError("Decision no longer matches the pending survey.")
            committed = survey.candidate.with_selected_root(decision.selected_root)
            self._committed = committed
            if update_anchor:
                self._anchor = committed
            self._history.append(committed)
            self._pending.clear()
            self._generation += 1
            return committed

    def discard(self, survey_id: Optional[str] = None) -> Union[int, StateSurvey]:
        """Discard one probe, or all pending probes when no id is supplied."""

        with self._lock:
            if survey_id is None:
                count = len(self._pending)
                self._pending.clear()
                return count
            return self._pending.pop(survey_id)

    def _get_pending(self, survey_id: str) -> StateSurvey:
        try:
            return self._pending[survey_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pending survey {survey_id!r}.") from exc

    def _select_survey(self, survey: StateSurvey) -> SelectionDecision:
        reference = self._committed if survey.reference_label == self._committed.label else self._anchor
        return self._selector(reference, survey)


__all__ = [
    "ElectronicSnapshot",
    "RootOverlapBlock",
    "SelectionConfig",
    "SelectionDecision",
    "SelectionStatus",
    "StateSelector",
    "StateSurvey",
    "SubspaceContinuity",
    "TrackingSession",
    "analyze_subspace_continuity",
    "normalize_signed_overlap_block",
    "select_state",
]

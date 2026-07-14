from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HARTREE_TO_EV = 27.211386245988
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


class DiabatizationError(RuntimeError):
    """Base class for actionable failures in the state-following workflow."""


class MissingDataError(DiabatizationError):
    """Raised when an engine cannot find the numerical data it requires."""


class OrthonormalityError(DiabatizationError):
    """Raised when parsed orbital coefficients fail C.T S C validation."""


@dataclass
class NTOContribution:
    donor: int
    donor_spin: str
    acceptor: int
    acceptor_spin: str
    weight: float


@dataclass
class CIContribution:
    donor: int
    donor_spin: str
    acceptor: int
    acceptor_spin: str
    weight: float
    coefficient: float


@dataclass
class CISAmplitudeCheck:
    step_label: str
    root: int
    source_file: Path
    donor: int
    donor_spin: str
    acceptor: int
    acceptor_spin: str
    printed_coefficient: float
    binary_coefficient: Optional[float]
    abs_error: Optional[float]
    passed: bool
    message: str = ""


@dataclass
class SelfNTOWeightRecord:
    step_label: str
    root: int
    spin: str
    source_file: Path
    pair_index: int
    singular_value: float
    weight: float
    spin_weight_fraction: float
    state_weight_fraction: float
    cumulative_spin_fraction: float
    selected: bool


@dataclass
class SelfNTOReconstructionRecord:
    step_label: str
    root: int
    spin: str
    source_file: Path
    n_occ: int
    n_virt: int
    rank: int
    matrix_norm: float
    reconstruction_error: float
    relative_error: float
    selected_pairs: int
    selected_spin_weight_fraction: float
    passed: bool
    message: str = ""


@dataclass
class NTOVisualizationRecord:
    step_label: str
    root: Optional[int]
    spin: str
    pair_index: Optional[int]
    side: str
    weight: Optional[float]
    cube_file: Optional[Path] = None
    png_file: Optional[Path] = None
    status: str = "unknown"
    message: str = ""
    overlap_max_abs_error: Optional[float] = None
    overlap_rms_error: Optional[float] = None


@dataclass
class StateRecord:
    step_label: str
    step_order: int
    scan_step: Optional[float]
    root: int
    exc_au: Optional[float] = None
    exc_ev: Optional[float] = None
    cm1: Optional[float] = None
    s2: Optional[float] = None
    multiplicity: Optional[int] = None
    ref_energy_eh: Optional[float] = None
    abs_energy_eh: Optional[float] = None
    contribs: List[NTOContribution] = field(default_factory=list)
    ci_contribs: List[CIContribution] = field(default_factory=list)


@dataclass
class JobRecord:
    order: int
    label: str
    scan_step: Optional[float]
    step_dir: Path
    out_path: Path
    geom_path: Path
    gbw_path: Optional[Path]
    uno_path: Optional[Path]
    nto_pattern: str


@dataclass
class ExtractionStatus:
    step_label: str
    root: Optional[int]
    extractor: str
    source_file: Optional[Path] = None
    json_file: Optional[Path] = None
    molden_file: Optional[Path] = None
    status: str = "unknown"
    message: str = ""


@dataclass
class OrthonormalityResult:
    step_label: str
    source_file: Path
    source_type: str
    max_diag_error: Optional[float]
    max_offdiag: Optional[float]
    rms_offdiag: Optional[float]
    passed: bool
    message: str = ""


@dataclass
class NTOIndexMapping:
    step_label: str
    root: int
    source_file: Path
    printed_donor: int
    donor_spin: str
    parsed_donor_vector: Optional[int]
    printed_acceptor: int
    acceptor_spin: str
    parsed_acceptor_vector: Optional[int]
    weight: float
    validation_status: str
    donor_candidate_vectors: str = ""
    acceptor_candidate_vectors: str = ""
    mapping_note: str = ""


@dataclass
class NTOOrbitalPair:
    spin: str
    donor_index: int
    acceptor_index: int
    weight: float
    donor_coeff: np.ndarray
    acceptor_coeff: np.ndarray
    source_file: Path
    donor_vector_index: Optional[int] = None
    acceptor_vector_index: Optional[int] = None
    donor_candidate_coeffs: Optional[np.ndarray] = None
    acceptor_candidate_coeffs: Optional[np.ndarray] = None
    donor_candidate_indices: List[int] = field(default_factory=list)
    acceptor_candidate_indices: List[int] = field(default_factory=list)


@dataclass
class NTOStateVector:
    step_label: str
    step_order: int
    root: int
    source_file: Path
    source_type: str
    pairs: List[NTOOrbitalPair] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        return float(sum(max(0.0, p.weight) for p in self.pairs))


@dataclass
class StepData:
    job: JobRecord
    states: Dict[int, StateRecord]
    nto_vectors: Dict[int, NTOStateVector] = field(default_factory=dict)
    self_nto_vectors: Dict[int, NTOStateVector] = field(default_factory=dict)
    tden_vectors: Dict[int, dict] = field(default_factory=dict)
    mo_coefficients: Dict[str, np.ndarray] = field(default_factory=dict)
    json_file: Optional[Path] = None


@dataclass
class ConfidenceResult:
    confidence: str
    reason: str
    best_similarity: float
    second_similarity: float
    margin: float
    ratio: float


@dataclass
class AssignmentEdge:
    step_a: str
    step_b: str
    track_id: int
    root_a: int
    root_b: int
    best_similarity: float
    second_similarity: float
    margin: float
    ratio: float
    confidence: str
    reason: str
    similarity_to_previous: Optional[float] = None
    similarity_to_anchor: Optional[float] = None
    manifold_id: Optional[str] = None


@dataclass
class TrackPoint:
    track_id: int
    step_order: int
    step_label: str
    scan_step: Optional[float]
    adiabatic_root: int
    energy_eh: Optional[float]
    excitation_ev: Optional[float]
    s2: Optional[float]
    multiplicity: Optional[int]
    similarity_to_previous: Optional[float]
    similarity_to_anchor: Optional[float]
    assignment_margin: Optional[float]
    confidence: str
    manifold_id: Optional[str]
    engine: str


@dataclass
class AmbiguousRegion:
    region_id: str
    start_step: str
    end_step: str
    manifold_roots: str
    tracks_involved: str
    min_subspace_score: float
    reason: str


@dataclass
class TrackingResult:
    track_points: List[TrackPoint]
    assignment_edges: List[AssignmentEdge]
    ambiguous_regions: List[AmbiguousRegion]
    adjacent_matrices: Dict[str, np.ndarray]
    roots: Sequence[int]
    engine: str
    bidirectional_disagreements: List[dict] = field(default_factory=list)


@dataclass
class TrackSegment:
    segment_id: str
    direction: str
    support: str
    seed_track_id: int
    seed_step: str
    seed_root: int
    start_step: str
    end_step: str
    start_root: int
    end_root: int
    step_count: int
    step_sequence: str
    root_sequence: str
    min_adjacent_similarity: Optional[float]
    min_purity: Optional[float]
    mean_purity: Optional[float]
    max_effective_rank: Optional[float]
    status: str
    reason: str

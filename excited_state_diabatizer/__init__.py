"""TDenTrack: transition-density based excited-state tracking for ORCA TDDFT scans."""

from .models import HARTREE_TO_EV
from .state_tracking import (
    ElectronicSnapshot,
    RootOverlapBlock,
    SelectionConfig,
    SelectionDecision,
    SelectionStatus,
    StateSelector,
    StateSurvey,
    SubspaceContinuity,
    TrackingSession,
    analyze_subspace_continuity,
    normalize_signed_overlap_block,
    select_state,
)

__version__ = "0.1.0"
__project_name__ = "TDenTrack"

__all__ = [
    "ElectronicSnapshot",
    "HARTREE_TO_EV",
    "RootOverlapBlock",
    "SelectionConfig",
    "SelectionDecision",
    "SelectionStatus",
    "StateSelector",
    "StateSurvey",
    "SubspaceContinuity",
    "TrackingSession",
    "__project_name__",
    "__version__",
    "analyze_subspace_continuity",
    "normalize_signed_overlap_block",
    "select_state",
]

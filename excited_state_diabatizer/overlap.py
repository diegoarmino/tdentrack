from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

from .models import NTOOrbitalPair, NTOStateVector


def nto_transition_density_similarity(
    state_a: NTOStateVector,
    state_b: NTOStateVector,
    s_ao_ab: np.ndarray,
) -> float:
    """Low-rank NTO transition-density similarity using AO cross overlaps."""
    pairs_a = [p for p in state_a.pairs if p.weight > 0]
    pairs_b = [p for p in state_b.pairs if p.weight > 0]
    norm_a = sum(p.weight for p in pairs_a)
    norm_b = sum(p.weight for p in pairs_b)
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    total = 0.0
    for pa in pairs_a:
        for pb in pairs_b:
            if _spin_key(pa.spin) != _spin_key(pb.spin):
                continue
            h = pair_side_overlap(pa, pb, "donor", s_ao_ab)
            p = pair_side_overlap(pa, pb, "acceptor", s_ao_ab)
            total += math.sqrt(pa.weight * pb.weight) * abs(h) * abs(p)
    return float(total / math.sqrt(norm_a * norm_b))


def pair_side_overlap(pair_a: NTOOrbitalPair, pair_b: NTOOrbitalPair, side: str, s_ao_ab: np.ndarray) -> float:
    a = _side_coefficients(pair_a, side)
    b = _side_coefficients(pair_b, side)
    s = np.asarray(s_ao_ab, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("NTO side coefficient blocks must be two-dimensional.")
    if s.shape != (a.shape[0], b.shape[0]):
        raise ValueError(f"Cross-overlap shape {s.shape} is incompatible with side blocks {a.shape}, {b.shape}.")
    overlaps = a.T @ s @ b
    if overlaps.size == 0:
        return 0.0
    flat_idx = int(np.argmax(np.abs(overlaps)))
    return float(overlaps.reshape(-1)[flat_idx])


def _side_coefficients(pair: NTOOrbitalPair, side: str) -> np.ndarray:
    if side == "donor":
        coeffs = pair.donor_candidate_coeffs
        fallback = pair.donor_coeff
    elif side == "acceptor":
        coeffs = pair.acceptor_candidate_coeffs
        fallback = pair.acceptor_coeff
    else:
        raise ValueError(f"Unknown NTO side: {side}")
    if coeffs is not None and coeffs.size:
        return np.asarray(coeffs, dtype=float)
    return np.asarray(fallback, dtype=float).reshape(-1, 1)


def orbital_overlap(coeff_a: np.ndarray, coeff_b: np.ndarray, s_ao_ab: np.ndarray) -> float:
    ca = np.asarray(coeff_a, dtype=float)
    cb = np.asarray(coeff_b, dtype=float)
    s = np.asarray(s_ao_ab, dtype=float)
    if ca.ndim != 1 or cb.ndim != 1:
        raise ValueError("Orbital coefficient vectors must be one-dimensional.")
    if s.shape != (ca.shape[0], cb.shape[0]):
        raise ValueError(f"Cross-overlap shape {s.shape} is incompatible with vectors {ca.shape}, {cb.shape}.")
    return float(ca @ s @ cb)


def nto_similarity_matrix(
    states_a: Dict[int, NTOStateVector],
    states_b: Dict[int, NTOStateVector],
    roots: Iterable[int],
    s_ao_ab: np.ndarray,
) -> np.ndarray:
    roots = list(roots)
    mat = np.zeros((len(roots), len(roots)), dtype=float)
    for i, ra in enumerate(roots):
        for j, rb in enumerate(roots):
            if ra in states_a and rb in states_b:
                mat[i, j] = nto_transition_density_similarity(states_a[ra], states_b[rb], s_ao_ab)
    return mat


def transition_density_overlap_matrix(
    tden_a: Dict[int, dict],
    tden_b: Dict[int, dict],
    roots: Iterable[int],
    s_occ_ab: np.ndarray,
    s_virt_ab: np.ndarray,
) -> np.ndarray:
    """Generic normalized MO-basis transition-density overlap.

    This helper supports artificial fixtures and future ORCA JSON exports that
    expose explicit occupied-virtual transition-amplitude matrices by spin. The
    caller is responsible for building occupied and virtual MO cross-overlap
    matrices in ORCA-consistent ordering.
    """
    roots = list(roots)
    mat = np.zeros((len(roots), len(roots)), dtype=float)
    for i, ra in enumerate(roots):
        for j, rb in enumerate(roots):
            mat[i, j] = transition_density_similarity(tden_a.get(ra), tden_b.get(rb), s_occ_ab, s_virt_ab)
    return mat


def cis_transition_density_overlap_matrix(
    tden_a: Dict[int, dict],
    tden_b: Dict[int, dict],
    roots: Iterable[int],
    mo_a: Dict[str, np.ndarray],
    mo_b: Dict[str, np.ndarray],
    s_ao_ab: np.ndarray,
) -> np.ndarray:
    """Return normalized CIS/TDA overlaps using zero-based active MO ranges.

    The inclusive ranges in each transition-density record are copied directly
    from ORCA's ``.cis`` header.  They therefore index the columns of the MO
    coefficient matrices without a one-based-to-zero-based conversion.
    """
    roots = list(roots)
    s_mo = {}
    for spin in ("alpha", "beta"):
        if spin in mo_a and spin in mo_b:
            ca = np.asarray(mo_a[spin], dtype=float)
            cb = np.asarray(mo_b[spin], dtype=float)
            s_mo[spin] = ca.T @ np.asarray(s_ao_ab, dtype=float) @ cb
    mat = np.zeros((len(roots), len(roots)), dtype=float)
    for i, ra in enumerate(roots):
        for j, rb in enumerate(roots):
            mat[i, j] = cis_transition_density_similarity(tden_a.get(ra), tden_b.get(rb), s_mo)
    return mat


def cis_transition_density_similarity(state_a: dict | None, state_b: dict | None, s_mo_by_spin: Dict[str, np.ndarray]) -> float:
    if not state_a or not state_b:
        return 0.0
    total = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for spin in ("alpha", "beta"):
        ta = _spin_tden(state_a, spin)
        tb = _spin_tden(state_b, spin)
        s_mo = s_mo_by_spin.get(spin)
        if ta is None or tb is None or s_mo is None:
            continue
        ta = np.asarray(ta, dtype=float)
        tb = np.asarray(tb, dtype=float)
        occ_a = _range_slice(state_a, spin, "occ")
        occ_b = _range_slice(state_b, spin, "occ")
        virt_a = _range_slice(state_a, spin, "virt")
        virt_b = _range_slice(state_b, spin, "virt")
        s_occ = s_mo[occ_a, occ_b]
        s_virt = s_mo[virt_a, virt_b]
        if s_occ.shape != (ta.shape[0], tb.shape[0]) or s_virt.shape != (ta.shape[1], tb.shape[1]):
            raise ValueError(
                f"CIS amplitude/MO overlap shape mismatch for {spin}: "
                f"ta={ta.shape}, tb={tb.shape}, s_occ={s_occ.shape}, s_virt={s_virt.shape}."
            )
        norm_a += float(np.sum(ta * ta))
        norm_b += float(np.sum(tb * tb))
        total += float(np.einsum("ia,ij,ab,jb->", ta, s_occ, s_virt, tb, optimize=True))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(abs(total) / math.sqrt(norm_a * norm_b))


def transition_density_similarity(
    state_a: dict | None,
    state_b: dict | None,
    s_occ_ab: np.ndarray,
    s_virt_ab: np.ndarray,
) -> float:
    if not state_a or not state_b:
        return 0.0
    total = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for spin in ("alpha", "beta"):
        ta = _spin_tden(state_a, spin)
        tb = _spin_tden(state_b, spin)
        if ta is None or tb is None:
            continue
        ta = np.asarray(ta, dtype=float)
        tb = np.asarray(tb, dtype=float)
        norm_a += float(np.sum(ta * ta))
        norm_b += float(np.sum(tb * tb))
        total += float(np.einsum("ia,ij,ab,jb->", ta, s_occ_ab, s_virt_ab, tb, optimize=True))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(abs(total) / math.sqrt(norm_a * norm_b))


def _spin_tden(state: dict, spin: str):
    for key in (spin, spin[0], f"{spin}_amplitudes", f"{spin}_transition_density"):
        if key in state:
            return state[key]
    if spin == "alpha":
        for key in ("amplitudes", "transition_density", "matrix"):
            if key in state:
                return state[key]
    return None


def _range_slice(state: dict, spin: str, kind: str) -> slice:
    key = f"{spin}_{kind}_range"
    if key not in state:
        short = "a" if spin == "alpha" else "b"
        key = f"{short}_{kind}_range"
    if key not in state:
        raise ValueError(f"Missing {spin} {kind} orbital range in transition-density record.")
    index_base = int(state.get("orbital_index_base", 0))
    if index_base != 0:
        raise ValueError(
            f"Unsupported CIS/TDA orbital index base {index_base}; TDenTrack active MO ranges are zero-based and inclusive."
        )
    start, end = state[key]
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid zero-based {spin} {kind} orbital range {start}..{end}.")
    return slice(start, end + 1)


def _spin_key(spin: str) -> str:
    s = str(spin).lower()
    if s.startswith("a"):
        return "alpha"
    if s.startswith("b"):
        return "beta"
    return s

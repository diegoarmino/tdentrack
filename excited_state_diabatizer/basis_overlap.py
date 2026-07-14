from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .json_io import parse_orca_json
from .models import MissingDataError, OrthonormalityResult
from .molden_io import MoldenFile, parse_molden

ANGULAR_MAP = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def load_same_geometry_overlap(job_json: Path) -> np.ndarray:
    parsed = parse_orca_json(job_json)
    if parsed.overlap is None:
        raise MissingDataError(
            f"{job_json} does not contain a same-geometry AO overlap matrix. "
            "Same-geometry C.T S C validation is mandatory; check the orca_2json export."
        )
    return parsed.overlap


def validate_orbital_orthonormality(
    coeffs: np.ndarray,
    overlap: np.ndarray,
    step_label: str,
    source_file: Path,
    source_type: str,
    diag_tol: float = 5e-4,
    offdiag_tol: float = 5e-4,
) -> OrthonormalityResult:
    if coeffs.size == 0:
        return OrthonormalityResult(
            step_label=step_label,
            source_file=Path(source_file),
            source_type=source_type,
            max_diag_error=None,
            max_offdiag=None,
            rms_offdiag=None,
            passed=False,
            message="No orbital coefficient vectors were available for validation.",
        )
    c = np.asarray(coeffs, dtype=float)
    s = np.asarray(overlap, dtype=float)
    if c.ndim != 2:
        return OrthonormalityResult(
            step_label=step_label,
            source_file=Path(source_file),
            source_type=source_type,
            max_diag_error=None,
            max_offdiag=None,
            rms_offdiag=None,
            passed=False,
            message=f"Coefficient array is not two-dimensional: shape={c.shape}.",
        )
    if s.shape[0] != s.shape[1] or s.shape[0] != c.shape[0]:
        return OrthonormalityResult(
            step_label=step_label,
            source_file=Path(source_file),
            source_type=source_type,
            max_diag_error=None,
            max_offdiag=None,
            rms_offdiag=None,
            passed=False,
            message=f"AO overlap shape {s.shape} is incompatible with coefficient shape {c.shape}.",
        )
    m = c.T @ s @ c
    diag = np.diag(m)
    max_diag_error = float(np.max(np.abs(diag - 1.0))) if diag.size else None
    off = m.copy()
    np.fill_diagonal(off, 0.0)
    max_offdiag = float(np.max(np.abs(off))) if off.size else 0.0
    if off.shape[0] > 1:
        rms_offdiag = float(np.sqrt(np.sum(off * off) / (off.shape[0] * (off.shape[0] - 1))))
    else:
        rms_offdiag = 0.0
    passed = bool(
        max_diag_error is not None
        and max_diag_error <= diag_tol
        and max_offdiag is not None
        and max_offdiag <= offdiag_tol
    )
    message = (
        "passed"
        if passed
        else "C.T S C validation failed. AO ordering, spherical/cartesian convention, "
        "normalization, or orbital-index mapping is inconsistent."
    )
    return OrthonormalityResult(
        step_label=step_label,
        source_file=Path(source_file),
        source_type=source_type,
        max_diag_error=max_diag_error,
        max_offdiag=max_offdiag,
        rms_offdiag=rms_offdiag,
        passed=passed,
        message=message,
    )


def get_cross_overlap(
    mode: str,
    molden_a: MoldenFile,
    molden_b: MoldenFile,
    same_overlap_a: Optional[np.ndarray] = None,
    same_overlap_b: Optional[np.ndarray] = None,
    require_order_check: bool = False,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "json-cross":
        raise MissingDataError(
            "ao-overlap-mode=json-cross was requested, but ORCA JSON cross-geometry AO overlap data "
            "were not found. A same-geometry S-Matrix is not a cross-geometry overlap. Use "
            "--ao-overlap-mode pyscf-cross with PySCF installed and validated, or an ORCA auxiliary "
            "cross-overlap export when available."
        )
    if mode == "orca-auxiliary":
        raise MissingDataError(
            "ao-overlap-mode=orca-auxiliary is a reserved backend for ORCA-generated cross-overlap jobs. "
            "This implementation does not fabricate an auxiliary ORCA syntax because it is version-dependent; "
            "provide a JSON cross-overlap export or use --ao-overlap-mode pyscf-cross."
        )
    if mode == "pyscf-cross":
        return pyscf_cross_overlap(
            molden_a,
            molden_b,
            same_overlap_a=same_overlap_a,
            same_overlap_b=same_overlap_b,
            require_order_check=require_order_check,
        )
    raise ValueError(f"Unknown AO overlap mode: {mode}")


def pyscf_cross_overlap(
    molden_a: MoldenFile,
    molden_b: MoldenFile,
    same_overlap_a: Optional[np.ndarray] = None,
    same_overlap_b: Optional[np.ndarray] = None,
    require_order_check: bool = False,
) -> np.ndarray:
    try:
        from pyscf import gto
    except Exception as exc:
        raise MissingDataError(
            "PySCF is required for --ao-overlap-mode pyscf-cross, but it is not importable. "
            "Install pyscf or choose an ORCA-generated cross-overlap mode."
        ) from exc
    mol_a = molden_to_pyscf_mol(molden_a)
    mol_b = molden_to_pyscf_mol(molden_b)
    if require_order_check:
        if same_overlap_a is not None:
            _check_pyscf_same_overlap(mol_a, same_overlap_a, molden_a.source)
        if same_overlap_b is not None:
            _check_pyscf_same_overlap(mol_b, same_overlap_b, molden_b.source)
    s_cross = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    if s_cross.shape != (len(molden_a.aos), len(molden_b.aos)):
        raise MissingDataError(
            f"PySCF cross-overlap shape {s_cross.shape} does not match parsed Molden AO counts "
            f"{len(molden_a.aos)} and {len(molden_b.aos)}."
        )
    return np.asarray(s_cross, dtype=float)


def molden_to_pyscf_mol(molden: MoldenFile):
    from pyscf import gto

    atom_entries = []
    basis: Dict[str, list] = {}
    shells_by_atom: Dict[int, list] = {}
    for shell in molden.shells:
        shells_by_atom.setdefault(shell.atom_index, []).append(shell)
    for i, atom in enumerate(molden.atoms):
        label = f"{atom.label}{atom.serial}"
        atom_entries.append((label, atom.coords))
        bs = []
        for shell in shells_by_atom.get(i, []):
            angular = ANGULAR_MAP.get(shell.angular.lower())
            if angular is None:
                raise MissingDataError(f"Cannot map Molden shell angular momentum {shell.angular!r} for PySCF.")
            bs.append([angular] + [[float(exp), float(coef)] for exp, coef in shell.primitives])
        basis[label] = bs
    cart = bool(molden.d_count == 6 or molden.f_count == 10 or molden.g_count == 15)
    mol = gto.M(atom=atom_entries, basis=basis, unit="Angstrom", cart=cart, verbose=0)
    return mol


def _check_pyscf_same_overlap(mol, same_overlap: np.ndarray, source: Path, tol: float = 5e-5) -> None:
    s = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    ref = np.asarray(same_overlap, dtype=float)
    if s.shape != ref.shape:
        raise MissingDataError(
            f"PySCF same-geometry AO overlap shape {s.shape} disagrees with ORCA JSON {ref.shape} for {source}. "
            "AO ordering or basis mapping is uncertain."
        )
    maxdiff = float(np.max(np.abs(s - ref)))
    if maxdiff > tol:
        raise MissingDataError(
            f"PySCF same-geometry AO overlap differs from ORCA JSON by {maxdiff:.3e} for {source}. "
            "AO ordering, spherical/cartesian convention, or contraction normalization is uncertain."
        )


class MoldenCache:
    def __init__(self) -> None:
        self._cache: Dict[Path, MoldenFile] = {}

    def get(self, path: Path) -> MoldenFile:
        p = Path(path).resolve()
        if p not in self._cache:
            self._cache[p] = parse_molden(p)
        return self._cache[p]

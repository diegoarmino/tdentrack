from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .models import MissingDataError


@dataclass
class ParsedJSON:
    source: Path
    atoms: List[dict]
    overlap: Optional[np.ndarray]
    mo_coefficients: Dict[str, np.ndarray]
    nto_states: Dict[int, dict]
    tden_states: Dict[int, dict]
    raw: dict
    messages: List[str]


def load_json(path: Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def parse_orca_json(path: Path) -> ParsedJSON:
    raw = load_json(path)
    mol = _case_get(raw, "Molecule") or raw
    atoms = _parse_atoms(mol)
    overlap = _find_matrix(raw, ["S-Matrix", "SMatrix", "Overlap", "AOOverlap", "ao_overlap"])
    mo_coefficients = _find_mo_coefficients(raw)
    nto_states = _find_nto_states(raw)
    tden_states = _find_tden_states(raw)
    messages: List[str] = []
    if overlap is None:
        messages.append("No one-electron AO overlap matrix was found in JSON.")
    if not mo_coefficients:
        messages.append("No MO/NTO coefficient arrays were found in JSON.")
    if not nto_states:
        messages.append("No NTO state records were found in JSON.")
    if not tden_states:
        messages.append("No transition-density or TDDFT amplitude records were found in JSON.")
    return ParsedJSON(
        source=Path(path),
        atoms=atoms,
        overlap=overlap,
        mo_coefficients=mo_coefficients,
        nto_states=nto_states,
        tden_states=tden_states,
        raw=raw,
        messages=messages,
    )


def require_tden_json(parsed: ParsedJSON) -> None:
    if not parsed.tden_states:
        raise MissingDataError(
            f"{parsed.source} does not contain TDDFT amplitudes or transition-density data. "
            "Run orca_2json with an ORCA version/export mode that includes TDDFT/CIS/RPA amplitudes, "
            "or use --engine nto-json/nto-molden for NTO-based tracking."
        )
    if not parsed.mo_coefficients:
        raise MissingDataError(f"{parsed.source} does not contain MO coefficients required for tden-json.")


def require_nto_json(parsed: ParsedJSON) -> None:
    if not parsed.nto_states:
        raise MissingDataError(
            f"{parsed.source} does not contain JSON NTO states. "
            "Use orca_2json on root-specific .nto files if supported, or use --engine nto-molden."
        )


def _case_get(obj: dict, key: str) -> Any:
    for k, v in obj.items():
        if str(k).lower() == key.lower():
            return v
    return None


def _parse_atoms(mol: dict) -> List[dict]:
    atoms = _case_get(mol, "Atoms") or []
    out: List[dict] = []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        out.append(
            {
                "label": atom.get("ElementLabel") or atom.get("label") or atom.get("symbol"),
                "z": atom.get("ElementNumber") or atom.get("NuclearCharge") or atom.get("z"),
                "coords": atom.get("Coords") or atom.get("coords") or atom.get("xyz"),
                "idx": atom.get("Idx") or atom.get("index"),
            }
        )
    return out


def _find_matrix(raw: Any, names: Iterable[str]) -> Optional[np.ndarray]:
    wanted = {n.lower() for n in names}
    for key, value in _walk_items(raw):
        if key.lower() in wanted:
            arr = _array2(value)
            if arr is not None:
                return arr
    return None


def _find_mo_coefficients(raw: Any) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    orca_mos = _find_orca_molecular_orbitals(raw)
    if orca_mos:
        out.update(orca_mos)
    for key, value in _walk_items(raw):
        lk = key.lower().replace(" ", "").replace("-", "_")
        if lk in {"mo_coefficients", "mocoefficients", "coefficients"}:
            arr = _array2(value)
            if arr is not None:
                out.setdefault("alpha", arr)
        elif lk in {"alpha_coefficients", "alpha_mo_coefficients"}:
            arr = _array2(value)
            if arr is not None:
                out["alpha"] = arr
        elif lk in {"beta_coefficients", "beta_mo_coefficients"}:
            arr = _array2(value)
            if arr is not None:
                out["beta"] = arr
    return out


def _find_orca_molecular_orbitals(raw: Any) -> Dict[str, np.ndarray]:
    mol = _case_get(raw, "Molecule") if isinstance(raw, dict) else None
    if not isinstance(mol, dict):
        return {}
    mo_block = _case_get(mol, "MolecularOrbitals")
    if not isinstance(mo_block, dict):
        return {}
    mos = _case_get(mo_block, "MOs")
    if not isinstance(mos, list) or not mos:
        return {}
    coeff_rows = []
    for rec in mos:
        if not isinstance(rec, dict):
            return {}
        coeff = rec.get("MOCoefficients") or rec.get("mo_coefficients") or rec.get("Coefficients")
        if coeff is None:
            return {}
        try:
            coeff_rows.append(np.asarray(coeff, dtype=float))
        except Exception:
            return {}
    if not coeff_rows:
        return {}
    lengths = {row.size for row in coeff_rows}
    if len(lengths) != 1:
        return {}
    n_ao = coeff_rows[0].size
    coeff_by_mo = np.vstack(coeff_rows)
    labels = _case_get(mo_block, "OrbitalLabels") or []
    hftyp = str(_case_get(mol, "HFTyp") or "").lower()
    out: Dict[str, np.ndarray] = {}
    if coeff_by_mo.shape[0] == 2 * n_ao and (_labels_repeat(labels, n_ao) or hftyp.startswith("u")):
        out["alpha"] = coeff_by_mo[:n_ao].T.copy()
        out["beta"] = coeff_by_mo[n_ao : 2 * n_ao].T.copy()
    else:
        out["alpha"] = coeff_by_mo.T.copy()
    return out


def _labels_repeat(labels: Any, n: int) -> bool:
    if not isinstance(labels, list) or len(labels) != 2 * n:
        return False
    return labels[:n] == labels[n:]


def _find_nto_states(raw: Any) -> Dict[int, dict]:
    candidates: List[Any] = []
    for key, value in _walk_items(raw):
        lk = key.lower().replace(" ", "_").replace("-", "_")
        if lk in {"nto_states", "ntostates", "natural_transition_orbitals"}:
            candidates.append(value)
    out: Dict[int, dict] = {}
    for value in candidates:
        if isinstance(value, dict):
            iterator = value.items()
        elif isinstance(value, list):
            iterator = enumerate(value, 1)
        else:
            continue
        for k, rec in iterator:
            if not isinstance(rec, dict):
                continue
            root = rec.get("root") or rec.get("state") or rec.get("Root") or rec.get("State")
            try:
                root_i = int(root if root is not None else k)
            except Exception:
                continue
            out[root_i] = rec
    return out


def _find_tden_states(raw: Any) -> Dict[int, dict]:
    candidates: List[Any] = []
    for key, value in _walk_items(raw):
        lk = key.lower().replace(" ", "_").replace("-", "_")
        if lk in {
            "transition_density",
            "transition_densities",
            "tden_states",
            "tddft_amplitudes",
            "cis_amplitudes",
            "rpa_amplitudes",
        }:
            candidates.append(value)
    out: Dict[int, dict] = {}
    for value in candidates:
        if isinstance(value, dict):
            iterator = value.items()
        elif isinstance(value, list):
            iterator = enumerate(value, 1)
        else:
            continue
        for k, rec in iterator:
            if not isinstance(rec, dict):
                continue
            root = rec.get("root") or rec.get("state") or rec.get("Root") or rec.get("State")
            try:
                root_i = int(root if root is not None else k)
            except Exception:
                continue
            out[root_i] = rec
    return out


def _walk_items(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
            yield from _walk_items(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_items(item)


def _array2(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
        return arr
    return None

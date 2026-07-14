from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .models import NTOContribution, NTOOrbitalPair, NTOStateVector

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class MoldenAtom:
    label: str
    serial: int
    z: int
    coords: Tuple[float, float, float]


@dataclass
class MoldenShell:
    atom_index: int
    angular: str
    primitives: List[Tuple[float, float]]


@dataclass
class MoldenMO:
    vector_index: int
    spin: str
    sym: str = ""
    energy: Optional[float] = None
    occupation: Optional[float] = None
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class MoldenFile:
    source: Path
    atoms: List[MoldenAtom]
    shells: List[MoldenShell]
    aos: List[dict]
    mos: List[MoldenMO]
    d_count: int
    f_count: int
    g_count: int
    coordinate_units: str


def parse_molden(path: Path) -> MoldenFile:
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    d_count = detect_molden_count(lines, "[5D]", "[6D]", 5)
    f_count = detect_molden_count(lines, "[7F]", "[10F]", 7)
    g_count = detect_g_count(lines)
    atoms, coordinate_units = _parse_atoms(lines)
    shells = _parse_gto(lines)
    aos = _build_aos(atoms, shells, d_count=d_count, f_count=f_count, g_count=g_count)
    mos = _parse_mos(lines, len(aos))
    return MoldenFile(
        source=path,
        atoms=atoms,
        shells=shells,
        aos=aos,
        mos=mos,
        d_count=d_count,
        f_count=f_count,
        g_count=g_count,
        coordinate_units=coordinate_units,
    )


def detect_molden_count(lines: Iterable[str], spherical_tag: str, cart_tag: str, default: int) -> int:
    tags = {ln.strip().upper() for ln in lines if ln.strip().startswith("[")}
    if spherical_tag.upper() in tags:
        return int(spherical_tag.strip("[]")[:-1])
    if cart_tag.upper() in tags:
        return int(cart_tag.strip("[]")[:-1])
    return default


def detect_g_count(lines: Iterable[str]) -> int:
    tags = {ln.strip().upper() for ln in lines if ln.strip().startswith("[")}
    if "[9G]" in tags:
        return 9
    if "[15G]" in tags:
        return 15
    return 9


def _parse_atoms(lines: List[str]) -> Tuple[List[MoldenAtom], str]:
    atoms: List[MoldenAtom] = []
    units = "AU"
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.lower().startswith("[atoms]"):
            units = "AU" if "AU" in s.upper() else "Angs"
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                p = lines[i].split()
                if len(p) >= 6:
                    try:
                        coords = (float(p[3]), float(p[4]), float(p[5]))
                        if units.upper() == "AU":
                            coords = tuple(x * BOHR_TO_ANGSTROM for x in coords)
                        atoms.append(MoldenAtom(label=p[0], serial=int(p[1]), z=int(p[2]), coords=coords))
                    except Exception:
                        pass
                i += 1
            break
        i += 1
    return atoms, units


def _parse_gto(lines: List[str]) -> List[MoldenShell]:
    shells: List[MoldenShell] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().lower().startswith("[gto]"):
            i += 1
            continue
        i += 1
        current_atom: Optional[int] = None
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("["):
                break
            if not line:
                i += 1
                continue
            p = line.split()
            if p and p[0].isdigit():
                current_atom = int(p[0]) - 1
                i += 1
                continue
            if current_atom is not None and len(p) >= 2 and p[0].lower() in {"s", "p", "d", "f", "g", "h"}:
                angular = p[0].lower()
                try:
                    nprim = int(float(p[1]))
                except Exception:
                    nprim = 0
                prims: List[Tuple[float, float]] = []
                for raw in lines[i + 1 : i + 1 + nprim]:
                    q = raw.split()
                    if len(q) >= 2:
                        try:
                            prims.append((float(q[0]), float(q[1])))
                        except Exception:
                            pass
                shells.append(MoldenShell(atom_index=current_atom, angular=angular, primitives=prims))
                i += 1 + nprim
                continue
            i += 1
        break
    return shells


def _build_aos(
    atoms: List[MoldenAtom],
    shells: List[MoldenShell],
    d_count: int,
    f_count: int,
    g_count: int,
) -> List[dict]:
    aos: List[dict] = []
    for shell in shells:
        atom = atoms[shell.atom_index] if 0 <= shell.atom_index < len(atoms) else None
        for comp in shell_components(shell.angular, d_count=d_count, f_count=f_count, g_count=g_count):
            aos.append(
                {
                    "atom": shell.atom_index,
                    "element": atom.label if atom else f"atom{shell.atom_index + 1}",
                    "l": shell.angular,
                    "component": comp,
                }
            )
    return aos


def shell_components(shell_l: str, d_count: int = 5, f_count: int = 7, g_count: int = 9) -> List[str]:
    l = shell_l.lower()
    if l == "s":
        return ["s"]
    if l == "p":
        return ["px", "py", "pz"]
    if l == "d":
        if d_count == 5:
            return ["dz2", "dxz", "dyz", "dx2y2", "dxy"]
        return ["dxx", "dyy", "dzz", "dxy", "dxz", "dyz"]
    if l == "f":
        return [f"f{i + 1}" for i in range(f_count)]
    if l == "g":
        return [f"g{i + 1}" for i in range(g_count)]
    return [l]


def _parse_mos(lines: List[str], nao: int) -> List[MoldenMO]:
    mos: List[MoldenMO] = []
    in_mo = False
    current: Optional[dict] = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        arr = np.zeros(nao, dtype=float)
        for k, v in current["coeffs"].items():
            if 0 <= k < nao:
                arr[k] = v
        mos.append(
            MoldenMO(
                vector_index=len(mos),
                spin=current.get("spin", ""),
                sym=current.get("sym", ""),
                energy=current.get("energy"),
                occupation=current.get("occupation"),
                coeffs=arr,
            )
        )
        current = None

    for raw in lines:
        s = raw.strip()
        if not in_mo:
            if s.lower().startswith("[mo]"):
                in_mo = True
            continue
        if s.startswith("[") and not s.lower().startswith("[mo]"):
            break
        if re.match(r"^Sym\s*=", s, flags=re.I):
            finish_current()
            current = {"sym": s.split("=", 1)[1].strip(), "spin": "", "coeffs": {}}
            continue
        if current is None:
            continue
        if "=" in s:
            key, val = s.split("=", 1)
            key = key.strip().lower()
            val = val.strip()
            if key == "spin":
                current["spin"] = val.lower()
            elif key == "ene":
                try:
                    current["energy"] = float(val)
                except Exception:
                    pass
            elif key == "occup":
                try:
                    current["occupation"] = float(val)
                except Exception:
                    pass
            continue
        p = s.split()
        if len(p) >= 2:
            try:
                current["coeffs"][int(p[0]) - 1] = float(p[1])
            except Exception:
                pass
    finish_current()
    return mos


def mo_by_orca_index(molden: MoldenFile, orca_index: int, spin: str, shift: int = 0) -> Tuple[int, MoldenMO]:
    wanted = "alpha" if str(spin).lower().startswith("a") else "beta" if str(spin).lower().startswith("b") else ""
    spin_mos = [mo for mo in molden.mos if (not wanted or mo.spin.startswith(wanted) or not mo.spin)]
    if not spin_mos:
        spin_mos = molden.mos
    idx = int(orca_index) + int(shift) - 1
    if idx < 0 or idx >= len(spin_mos):
        raise IndexError(
            f"Printed ORCA NTO index {orca_index}{spin} maps to parsed vector index {idx}, "
            f"but only {len(spin_mos)} {wanted or 'unlabeled'} vectors were found in {molden.source}."
        )
    return idx, spin_mos[idx]


def select_nto_contributions(
    contribs: Iterable[NTOContribution],
    weight_min: float = 0.01,
    cumulative: float = 0.995,
) -> List[NTOContribution]:
    ordered = sorted(list(contribs), key=lambda c: c.weight, reverse=True)
    if not ordered:
        return []
    total = sum(max(0.0, c.weight) for c in ordered)
    selected: List[NTOContribution] = [c for c in ordered if c.weight >= weight_min]
    selected_ids = {id(c) for c in selected}
    running = sum(max(0.0, c.weight) for c in selected)
    for c in ordered:
        if total <= 0:
            break
        if running / total >= cumulative:
            break
        if id(c) in selected_ids:
            continue
        selected.append(c)
        selected_ids.add(id(c))
        running += max(0.0, c.weight)
    return sorted(selected, key=lambda c: c.weight, reverse=True)


def build_nto_state_from_molden(
    molden: MoldenFile,
    step_label: str,
    step_order: int,
    root: int,
    contribs: Iterable[NTOContribution],
    nto_index_shift: int = 0,
    weight_min: float = 0.01,
    cumulative: float = 0.995,
    pair_selection: str = "all",
) -> NTOStateVector:
    selected = select_nto_contributions(contribs, weight_min=weight_min, cumulative=cumulative)
    selected = select_nto_pair_mode(selected, pair_selection)
    vec = NTOStateVector(
        step_label=step_label,
        step_order=step_order,
        root=root,
        source_file=molden.source,
        source_type="nto-molden",
    )
    for c in selected:
        if c.donor_spin.lower()[:1] != c.acceptor_spin.lower()[:1]:
            vec.diagnostics.append(
                f"Skipped mixed-spin NTO pair {c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin}; "
                "this v1 implementation compares same-spin alpha/alpha and beta/beta pairs only."
            )
            continue
        donor_idx, donor = mo_by_orca_index(molden, c.donor, c.donor_spin, shift=nto_index_shift)
        acc_idx, acc = mo_by_orca_index(molden, c.acceptor, c.acceptor_spin, shift=nto_index_shift)
        vec.pairs.append(
            NTOOrbitalPair(
                spin="alpha" if c.donor_spin.lower().startswith("a") else "beta",
                donor_index=c.donor,
                acceptor_index=c.acceptor,
                weight=c.weight,
                donor_coeff=donor.coeffs.copy(),
                acceptor_coeff=acc.coeffs.copy(),
                source_file=molden.source,
                donor_vector_index=donor_idx,
                acceptor_vector_index=acc_idx,
            )
        )
        d_occ = "" if donor.occupation is None else f", donor Occup={donor.occupation:.8g}"
        a_occ = "" if acc.occupation is None else f", acceptor Occup={acc.occupation:.8g}"
        vec.diagnostics.append(
            f"{c.donor}{c.donor_spin}->{c.acceptor}{c.acceptor_spin}: "
            f"printed indices -> parsed vectors {donor_idx + 1}/{acc_idx + 1}{d_occ}{a_occ}"
        )
    if not vec.pairs:
        vec.diagnostics.append("No NTO pairs survived weight/cumulative selection and spin filtering.")
    return vec


def select_nto_pair_mode(contribs: Iterable[NTOContribution], pair_selection: str = "all") -> List[NTOContribution]:
    selected = list(contribs)
    if pair_selection == "all":
        return selected
    same_spin = [c for c in selected if c.donor_spin.lower()[:1] == c.acceptor_spin.lower()[:1]]
    if pair_selection == "dominant-total":
        return [max(same_spin, key=lambda c: c.weight)] if same_spin else []
    if pair_selection == "dominant-per-spin":
        out: List[NTOContribution] = []
        for spin in ("a", "b"):
            spin_contribs = [c for c in same_spin if c.donor_spin.lower().startswith(spin)]
            if spin_contribs:
                out.append(max(spin_contribs, key=lambda c: c.weight))
        return sorted(out, key=lambda c: c.weight, reverse=True)
    return selected


def coefficients_for_validation(vec: NTOStateVector) -> np.ndarray:
    cols: List[np.ndarray] = []
    for pair in vec.pairs:
        cols.append(pair.donor_coeff)
        cols.append(pair.acceptor_coeff)
    if not cols:
        return np.zeros((0, 0))
    return np.column_stack(cols)

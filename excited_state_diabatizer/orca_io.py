from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .models import CIContribution, FLOAT_RE, JobRecord, NTOContribution, StateRecord


def parse_int_range(text: str) -> List[int]:
    vals: List[int] = []
    for tok in re.split(r"[,\s]+", str(text).strip()):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            vals.extend(range(int(a), int(b) + 1))
        else:
            vals.append(int(tok))
    return sorted(set(vals))


def step_label_to_number(label: str) -> Optional[float]:
    m = re.match(r"([pm])(\d+)$", str(label).strip())
    if m:
        val = int(m.group(2))
        return float(val if m.group(1) == "p" else -val)
    try:
        return float(label)
    except Exception:
        return None


def default_step_label(step: float) -> str:
    if abs(step - round(step)) < 1e-10:
        i = int(round(step))
        return ("p" if i >= 0 else "m") + f"{abs(i):03d}"
    return str(step).replace("-", "m").replace(".", "p")


def parse_final_energy(out_path: Path) -> Optional[float]:
    if not Path(out_path).exists():
        return None
    text = Path(out_path).read_text(errors="replace")
    m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(%s)" % FLOAT_RE, text)
    if m:
        return float(m[-1])
    m = re.findall(r"Total\s+Energy\s*:\s*(%s)\s*Eh" % FLOAT_RE, text)
    return float(m[-1]) if m else None


def parse_gs_summary(workdir: Path) -> Dict[str, dict]:
    path = Path(workdir) / "gs_scan" / "gs_scan_summary.csv"
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or row.get("step_label") or "").strip()

            def ffloat(x: object) -> Optional[float]:
                try:
                    return float(str(x).strip())
                except Exception:
                    return None

            step = ffloat(row.get("step") or row.get("step_number") or "")
            if not label and step is not None:
                label = default_step_label(step)
            if label:
                out[label] = {
                    "step": step,
                    "energy_eh": ffloat(row.get("energy_Eh") or row.get("energy_eh") or ""),
                    "final_q": ffloat(row.get("final_Q_A") or row.get("final_q_A") or ""),
                    "converged": str(row.get("converged", "")).lower() in {"1", "true", "yes"},
                    "normal": str(row.get("normal", row.get("terminated_normally", ""))).lower()
                    in {"1", "true", "yes"},
                }
    return out


def parse_tddft_output(
    out_path: Path,
    label: str = "",
    step_order: int = 0,
    scan_step: Optional[float] = None,
) -> Tuple[Optional[float], Dict[int, StateRecord], bool]:
    text = Path(out_path).read_text(errors="replace") if Path(out_path).exists() else ""
    ref = parse_final_energy(out_path)
    normal = "ORCA TERMINATED NORMALLY" in text
    state_re = re.compile(
        r"STATE\s+(\d+)\s*:\s*E=\s*"
        r"(%s)\s*au\s+"
        r"(%s)\s*eV\s+"
        r"(%s)\s*cm\*\*-1"
        r"(?:.*?<S\*\*2>\s*=\s*(%s))?"
        r"(?:.*?Mult\s+(\d+))?" % (FLOAT_RE, FLOAT_RE, FLOAT_RE, FLOAT_RE),
        re.I,
    )
    states: Dict[int, StateRecord] = {}
    for m in state_re.finditer(text):
        root = int(m.group(1))
        exc_au = float(m.group(2))
        states[root] = StateRecord(
            step_label=label,
            step_order=step_order,
            scan_step=scan_step,
            root=root,
            exc_au=exc_au,
            exc_ev=float(m.group(3)),
            cm1=float(m.group(4)),
            s2=None if m.group(5) is None else float(m.group(5)),
            multiplicity=None if m.group(6) is None else int(m.group(6)),
            ref_energy_eh=ref,
            abs_energy_eh=None if ref is None else ref + exc_au,
        )

    nto_blocks = parse_nto_blocks(text)
    for root, contribs in nto_blocks.items():
        if root in states:
            states[root].contribs = contribs
        else:
            states[root] = StateRecord(
                step_label=label,
                step_order=step_order,
                scan_step=scan_step,
                root=root,
                ref_energy_eh=ref,
                contribs=contribs,
            )
    ci_blocks = parse_ci_coefficients(text)
    for root, contribs in ci_blocks.items():
        if root in states:
            states[root].ci_contribs = contribs
        else:
            states[root] = StateRecord(
                step_label=label,
                step_order=step_order,
                scan_step=scan_step,
                root=root,
                ref_energy_eh=ref,
                ci_contribs=contribs,
            )
    return ref, states, normal


def parse_nto_blocks(text: str) -> Dict[int, List[NTOContribution]]:
    header_re = re.compile(r"NATURAL\s+TRANSITION\s+ORBITALS\s+FOR\s+STATE\s+(\d+)", re.I)
    hits = list(header_re.finditer(text))
    out: Dict[int, List[NTOContribution]] = {}
    for idx, h in enumerate(hits):
        root = int(h.group(1))
        end = hits[idx + 1].start() if idx + 1 < len(hits) else len(text)
        block = text[h.start() : end]
        contribs: List[NTOContribution] = []
        for dm, ds, am, aps, wt in re.findall(
            r"(\d+)\s*([abAB]?)\s*->\s*(\d+)\s*([abAB]?)\s*:\s*n\s*=\s*(%s)" % FLOAT_RE,
            block,
        ):
            contribs.append(
                NTOContribution(
                    donor=int(dm),
                    donor_spin=(ds.lower() or "a"),
                    acceptor=int(am),
                    acceptor_spin=(aps.lower() or "a"),
                    weight=float(wt),
                )
            )
        out[root] = contribs
    return out


def parse_ci_coefficients(text: str) -> Dict[int, List[CIContribution]]:
    state_re = re.compile(r"^\s*STATE\s+(\d+)\s*:\s*E=", re.I | re.M)
    hits = list(state_re.finditer(text))
    out: Dict[int, List[CIContribution]] = {}
    coeff_re = re.compile(
        r"^\s*(\d+)\s*([abAB]?)\s*->\s*(\d+)\s*([abAB]?)\s*:\s*"
        r"(%s)\s*\(c=\s*(%s)\s*\)" % (FLOAT_RE, FLOAT_RE),
        re.M,
    )
    for idx, h in enumerate(hits):
        root = int(h.group(1))
        end = hits[idx + 1].start() if idx + 1 < len(hits) else len(text)
        block = text[h.start() : end]
        # Stop before printed NTO sections, whose "n=" lines share the same donor->acceptor shape.
        nto_pos = re.search(r"NATURAL\s+TRANSITION\s+ORBITALS", block, re.I)
        if nto_pos:
            block = block[: nto_pos.start()]
        contribs: List[CIContribution] = []
        for dm, ds, am, aps, wt, coeff in coeff_re.findall(block):
            donor_spin = (ds.lower() or aps.lower() or "a")
            acceptor_spin = (aps.lower() or donor_spin)
            contribs.append(
                CIContribution(
                    donor=int(dm),
                    donor_spin=donor_spin,
                    acceptor=int(am),
                    acceptor_spin=acceptor_spin,
                    weight=float(wt),
                    coefficient=float(coeff),
                )
            )
        if contribs:
            out[root] = contribs
    return out


def find_workdir_jobs(workdir: Path) -> List[JobRecord]:
    workdir = Path(workdir).resolve()
    tddft_dir = workdir / "tddft"
    if not tddft_dir.exists():
        raise FileNotFoundError(f"TDDFT directory not found: {tddft_dir}")
    jobs: List[JobRecord] = []
    for d in sorted(tddft_dir.glob("tddft_step_*")):
        if not d.is_dir():
            continue
        label = d.name.replace("tddft_step_", "", 1)
        scan_step = step_label_to_number(label)
        jobs.append(
            JobRecord(
                order=0,
                label=label,
                scan_step=scan_step,
                step_dir=d.resolve(),
                out_path=(d / "job.out").resolve(),
                geom_path=(d / "geom.xyz").resolve(),
                gbw_path=(d / "job.gbw").resolve(),
                uno_path=(d / "job.uno").resolve(),
                nto_pattern=str((d / "job.s{state}.nto").resolve()),
            )
        )
    jobs.sort(key=lambda j: (float("inf") if j.scan_step is None else j.scan_step, j.label))
    for i, job in enumerate(jobs):
        job.order = i
    return jobs


def read_jobs_csv(path: Path) -> List[JobRecord]:
    path = Path(path).resolve()
    rows: List[JobRecord] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            order = int(row.get("order") or len(rows))
            label = (row.get("label") or f"step_{order:04d}").strip()
            step_txt = row.get("step") or row.get("scan_step") or ""
            scan_step = step_label_to_number(step_txt) if step_txt != "" else step_label_to_number(label)
            out_path = _csv_path(row.get("out"), path.parent)
            geom_path = _csv_path(row.get("geom"), path.parent)
            gbw_path = _csv_path(row.get("gbw"), path.parent, optional=True)
            uno_path = _csv_path(row.get("uno"), path.parent, optional=True)
            step_dir = out_path.parent if out_path is not None else path.parent
            nto_pattern = (row.get("nto_pattern") or str(step_dir / "job.s{state}.nto")).strip()
            if not Path(nto_pattern).is_absolute():
                nto_pattern = str((path.parent / nto_pattern).resolve())
            rows.append(
                JobRecord(
                    order=order,
                    label=label,
                    scan_step=scan_step,
                    step_dir=step_dir.resolve(),
                    out_path=out_path.resolve(),
                    geom_path=geom_path.resolve(),
                    gbw_path=None if gbw_path is None else gbw_path.resolve(),
                    uno_path=None if uno_path is None else uno_path.resolve(),
                    nto_pattern=nto_pattern,
                )
            )
    rows.sort(key=lambda j: (j.order, j.label))
    return rows


def _csv_path(value: object, base: Path, optional: bool = False) -> Optional[Path]:
    text = "" if value is None else str(value).strip()
    if not text:
        if optional:
            return None
        raise ValueError("jobs CSV is missing a required path column")
    p = Path(text)
    return p if p.is_absolute() else base / p


def nto_path_for_job(job: JobRecord, root: int) -> Path:
    try:
        text = job.nto_pattern.format(state=root, root=root)
    except Exception:
        text = job.nto_pattern.replace("{state}", str(root)).replace("{root}", str(root))
    return Path(text)


def filter_states(states: Dict[int, StateRecord], roots: Iterable[int]) -> Dict[int, StateRecord]:
    wanted = set(roots)
    return {r: s for r, s in states.items() if r in wanted}

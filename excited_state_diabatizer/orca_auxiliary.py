from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cis_io import write_orca_2json_overlap_config
from .json_io import parse_orca_json
from .models import ExtractionStatus, JobRecord, MissingDataError


def orca_auxiliary_cross_overlap(
    job_a: JobRecord,
    job_b: JobRecord,
    same_a: np.ndarray,
    same_b: np.ndarray,
    outdir: Path,
    orca_exe: str,
    orca_2json_exe: str,
    statuses: List[ExtractionStatus],
    force: bool = False,
    timeout: int = 900,
) -> np.ndarray:
    outdir = Path(outdir)
    cache = outdir / "json_cache" / f"cross_{job_a.label}_{job_b.label}.json"
    if cache.exists() and not force:
        return _load_cross_block(cache, same_a, same_b, job_a.label, job_b.label, statuses)

    work = outdir / "json_cache" / f"orca_aux_{job_a.label}_{job_b.label}"
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "cross.inp"
    _write_cross_input(inp, job_a, job_b)
    write_orca_2json_overlap_config(work / "cross.json.conf")

    orca_log = outdir / "utility_logs" / f"orca_aux_cross_{job_a.label}_{job_b.label}.log"
    proc = subprocess.run([orca_exe, "cross.inp"], cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    orca_log.write_text(
        "COMMAND: "
        + f"{orca_exe} cross.inp"
        + f"\nreturncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    gbw = work / "cross.gbw"
    if not gbw.exists():
        statuses.append(
            ExtractionStatus(
                job_a.label,
                None,
                "orca-auxiliary-cross",
                source_file=inp,
                json_file=cache,
                status="failed",
                message=f"Auxiliary ORCA job did not produce cross.gbw; see {orca_log}.",
            )
        )
        raise MissingDataError(f"ORCA auxiliary cross-overlap job failed for {job_a.label}->{job_b.label}; see {orca_log}.")

    json_log = outdir / "utility_logs" / f"orca_2json_aux_cross_{job_a.label}_{job_b.label}.log"
    proc_json = subprocess.run([orca_2json_exe, "cross.gbw"], cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    json_log.write_text(
        "COMMAND: "
        + f"{orca_2json_exe} cross.gbw"
        + "\nCONFIG:\n"
        + (work / "cross.json.conf").read_text()
        + f"\nreturncode={proc_json.returncode}\n\nSTDOUT:\n{proc_json.stdout}\n\nSTDERR:\n{proc_json.stderr}"
    )
    produced = work / "cross.json"
    if proc_json.returncode != 0 or not produced.exists():
        statuses.append(
            ExtractionStatus(
                job_a.label,
                None,
                "orca-auxiliary-cross",
                source_file=gbw,
                json_file=cache,
                status="failed",
                message=f"orca_2json did not export auxiliary cross-overlap JSON; see {json_log}.",
            )
        )
        raise MissingDataError(f"orca_2json auxiliary cross-overlap export failed for {job_a.label}->{job_b.label}; see {json_log}.")
    shutil.copy2(produced, cache)
    _cleanup_auxiliary_workdir(work)
    statuses.append(
        ExtractionStatus(
            f"{job_a.label}->{job_b.label}",
            None,
            "orca-auxiliary-cross",
            source_file=inp,
            json_file=cache,
            status="ok",
            message="ORCA-generated cross-geometry AO overlap extracted and cached.",
        )
    )
    return _load_cross_block(cache, same_a, same_b, job_a.label, job_b.label, statuses)


def _load_cross_block(
    json_file: Path,
    same_a: np.ndarray,
    same_b: np.ndarray,
    label_a: str,
    label_b: str,
    statuses: List[ExtractionStatus],
    tol: float = 5.0e-8,
) -> np.ndarray:
    parsed = parse_orca_json(json_file)
    if parsed.overlap is None:
        raise MissingDataError(f"{json_file} does not contain an auxiliary ORCA S-Matrix.")
    s = np.asarray(parsed.overlap, dtype=float)
    n_a = same_a.shape[0]
    n_b = same_b.shape[0]
    if s.shape != (n_a + n_b, n_a + n_b):
        raise MissingDataError(
            f"Auxiliary S-Matrix shape {s.shape} is incompatible with block sizes {n_a} and {n_b} "
            f"for {label_a}->{label_b}."
        )
    block_a = s[:n_a, :n_a]
    block_b = s[n_a:, n_a:]
    err_a = float(np.max(np.abs(block_a - same_a)))
    err_b = float(np.max(np.abs(block_b - same_b)))
    sym_err = float(np.max(np.abs(s[:n_a, n_a:] - s[n_a:, :n_a].T)))
    if err_a > tol or err_b > tol:
        raise MissingDataError(
            f"Auxiliary S-Matrix block validation failed for {label_a}->{label_b}: "
            f"A block error {err_a:.3g}, B block error {err_b:.3g}, tolerance {tol:.3g}."
        )
    statuses.append(
        ExtractionStatus(
            f"{label_a}->{label_b}",
            None,
            "orca-auxiliary-cross-validation",
            json_file=json_file,
            status="ok",
            message=f"Diagonal S blocks validated: max errors {err_a:.3g}, {err_b:.3g}; cross symmetry error {sym_err:.3g}.",
        )
    )
    return s[:n_a, n_a:]


def _write_cross_input(path: Path, job_a: JobRecord, job_b: JobRecord) -> None:
    charge_a, _ = _parse_charge_mult(job_a.out_path, job_a.step_dir / "job.inp")
    charge_b, _ = _parse_charge_mult(job_b.out_path, job_b.step_dir / "job.inp")
    basis_tokens, basis_block = _extract_basis_directives(job_a.step_dir / "job.inp")
    atoms = _read_xyz(job_a.geom_path) + _read_xyz(job_b.geom_path)
    with Path(path).open("w") as f:
        f.write("! HF " + " ".join(basis_tokens) + " NoIter MiniPrint\n\n")
        if basis_block:
            f.write(basis_block.rstrip() + "\n\n")
        f.write("%maxcore 2000\n\n")
        f.write(f"* xyz {charge_a + charge_b} 1\n")
        for element, x, y, z in atoms:
            f.write(f"{element:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")
        f.write("*\n")


def _read_xyz(path: Path) -> List[Tuple[str, float, float, float]]:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise MissingDataError(f"Empty XYZ file: {path}")
    n_atoms = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise MissingDataError(f"Malformed XYZ line in {path}: {line}")
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if len(atoms) != n_atoms:
        raise MissingDataError(f"XYZ atom count mismatch in {path}: expected {n_atoms}, got {len(atoms)}.")
    return atoms


def _parse_charge_mult(out_path: Path, inp_path: Path) -> Tuple[int, int]:
    if inp_path.exists():
        for line in inp_path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("* xyz"):
                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        return int(parts[2]), int(parts[3])
                    except Exception:
                        pass
    return 0, 1


def _extract_basis_directives(inp_path: Path) -> Tuple[List[str], str]:
    text = Path(inp_path).read_text(errors="replace") if Path(inp_path).exists() else ""
    basis_tokens: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            for tok in stripped[1:].split():
                if _looks_like_orca_basis_token(tok):
                    basis_tokens.append(tok)
            break
    if not basis_tokens:
        raise MissingDataError(f"Could not infer a main ORCA basis token from {inp_path}.")
    block = _extract_percent_block(text, "basis")
    return basis_tokens, block


def _looks_like_orca_basis_token(token: str) -> bool:
    low = token.lower()
    if "/" in low or low in {"rijcosx", "cosx"}:
        return False
    markers = ("def2", "cc-pv", "aug-cc", "ano", "sarc", "ma-", "dhf-", "jorge")
    return any(m in low for m in markers)


def _extract_percent_block(text: str, name: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    in_block = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if not in_block and stripped.lower().startswith("%" + name.lower()):
            in_block = True
            depth = 1
            out.append(line)
            continue
        if in_block:
            out.append(line)
            if stripped.lower() == "end":
                depth -= 1
                if depth <= 0:
                    break
    return "\n".join(out)


def _cleanup_auxiliary_workdir(work: Path) -> None:
    for pattern in ("*.tmp", "*.tmp.*", "*.grid.tmp", "*.shark_grid.tmp", "*.propint.tmp.*", "*.cpscfdata.tmp.*"):
        for path in Path(work).glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass

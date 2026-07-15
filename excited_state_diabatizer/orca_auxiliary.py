from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cis_io import write_orca_2json_overlap_config
from .json_io import parse_orca_json
from .models import ExtractionStatus, JobRecord, MissingDataError


_ORCA_NORMAL_TERMINATION = "ORCA TERMINATED NORMALLY"


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
    cache_dir = outdir / "json_cache"
    log_dir = outdir / "utility_logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # The labels are useful to a human reading the cache, but are not a cache
    # identity: the input digest changes when either geometry, charge, or basis
    # specification changes.  In particular, reusing labels for a restarted
    # scan can no longer silently reuse an overlap from an older geometry.
    cross_input = _render_cross_input(job_a, job_b)
    input_digest = _auxiliary_input_digest(cross_input)
    pair_name = f"{_safe_path_component(job_a.label)}_{_safe_path_component(job_b.label)}"
    cache = cache_dir / f"cross_{pair_name}_{input_digest}.json"
    if cache.exists() and not force:
        return _load_cross_block(cache, same_a, same_b, job_a.label, job_b.label, statuses)

    work = cache_dir / f"orca_aux_{pair_name}_{input_digest}"
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "cross.inp"
    inp.write_text(cross_input)
    write_orca_2json_overlap_config(work / "cross.json.conf")

    # Do not let products from an interrupted run make a subsequent failed run
    # appear successful.
    for stale_name in ("cross.gbw", "cross.json"):
        stale = work / stale_name
        if stale.exists():
            stale.unlink()

    orca_log = log_dir / f"orca_aux_cross_{pair_name}_{input_digest}.log"
    proc = subprocess.run([orca_exe, "cross.inp"], cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    orca_log.write_text(
        "COMMAND: "
        + f"{orca_exe} cross.inp"
        + f"\nreturncode={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
    )
    gbw = work / "cross.gbw"
    terminated_normally = _orca_terminated_normally(proc.stdout)
    if proc.returncode != 0 or not terminated_normally or not gbw.exists():
        reasons = []
        if proc.returncode != 0:
            reasons.append(f"return code {proc.returncode}")
        if not terminated_normally:
            reasons.append(f"missing '{_ORCA_NORMAL_TERMINATION}' marker")
        if not gbw.exists():
            reasons.append("missing cross.gbw")
        reason = "; ".join(reasons)
        statuses.append(
            ExtractionStatus(
                f"{job_a.label}->{job_b.label}",
                None,
                "orca-auxiliary-cross",
                source_file=inp,
                json_file=cache,
                status="failed",
                message=f"Auxiliary ORCA job failed ({reason}); see {orca_log}.",
            )
        )
        raise MissingDataError(
            f"ORCA auxiliary cross-overlap job failed for {job_a.label}->{job_b.label} "
            f"({reason}); see {orca_log}."
        )

    json_log = log_dir / f"orca_2json_aux_cross_{pair_name}_{input_digest}.log"
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
    # Validate before publishing the cache.  A malformed or inconsistent
    # export must not poison future non-forced runs for this input digest.
    cross_block = _load_cross_block(produced, same_a, same_b, job_a.label, job_b.label, statuses)
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
    return cross_block


def _load_cross_block(
    json_file: Path,
    same_a: np.ndarray,
    same_b: np.ndarray,
    label_a: str,
    label_b: str,
    statuses: List[ExtractionStatus],
    tol: float = 5.0e-8,
) -> np.ndarray:
    same_a = _validated_square_matrix(same_a, "same-geometry A S-Matrix")
    same_b = _validated_square_matrix(same_b, "same-geometry B S-Matrix")
    parsed = parse_orca_json(json_file)
    if parsed.overlap is None:
        raise MissingDataError(f"{json_file} does not contain an auxiliary ORCA S-Matrix.")
    s = np.asarray(parsed.overlap, dtype=float)
    if not np.all(np.isfinite(s)):
        raise MissingDataError(f"Auxiliary S-Matrix in {json_file} contains non-finite values.")
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
    errors = []
    if err_a > tol or err_b > tol:
        errors.append(f"A block error {err_a:.3g}, B block error {err_b:.3g}")
    if sym_err > tol:
        errors.append(f"cross-block transpose-symmetry error {sym_err:.3g}")
    if errors:
        message = (
            f"Auxiliary S-Matrix block validation failed for {label_a}->{label_b}: "
            + "; ".join(errors)
            + f", tolerance {tol:.3g}."
        )
        statuses.append(
            ExtractionStatus(
                f"{label_a}->{label_b}",
                None,
                "orca-auxiliary-cross-validation",
                json_file=json_file,
                status="failed",
                message=message,
            )
        )
        raise MissingDataError(
            message
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
    Path(path).write_text(_render_cross_input(job_a, job_b))


def _render_cross_input(job_a: JobRecord, job_b: JobRecord) -> str:
    charge_a, _ = _parse_charge_mult(job_a.out_path, job_a.step_dir / "job.inp")
    charge_b, _ = _parse_charge_mult(job_b.out_path, job_b.step_dir / "job.inp")
    inp_a = job_a.step_dir / "job.inp"
    inp_b = job_b.step_dir / "job.inp"
    basis_tokens, basis_block = _extract_basis_directives(inp_a)
    if inp_b.exists():
        basis_tokens_b, basis_block_b = _extract_basis_directives(inp_b)
        if _canonical_basis_spec(basis_tokens, basis_block) != _canonical_basis_spec(basis_tokens_b, basis_block_b):
            raise MissingDataError(
                f"Auxiliary cross-overlap requires identical basis specifications, but {inp_a} and {inp_b} differ. "
                "Use the same AO basis at both geometries or provide a validated cross-overlap backend for unequal bases."
            )
    atoms = _read_xyz(job_a.geom_path) + _read_xyz(job_b.geom_path)
    lines = [" ".join(["!", "HF", *basis_tokens, "NoIter", "MiniPrint"]), ""]
    if basis_block:
        lines.extend([basis_block.rstrip(), ""])
    lines.extend(["%maxcore 2000", "", f"* xyz {charge_a + charge_b} 1"])
    lines.extend(f"{element:2s} {x:16.10f} {y:16.10f} {z:16.10f}" for element, x, y, z in atoms)
    lines.extend(["*", ""])
    return "\n".join(lines)


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
    block = _extract_percent_block(text, "basis")
    basis_tokens: List[str] = []
    seen = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("!"):
            # ORCA accepts multiple simple-input lines.  Strip its common '#'
            # comments and inspect all such lines rather than only the first.
            simple = stripped[1:].split("#", 1)[0]
            for tok in simple.split():
                if _looks_like_orca_basis_token(tok):
                    key = tok.casefold()
                    if key not in seen:
                        seen.add(key)
                        basis_tokens.append(tok)
    if not basis_tokens and not block:
        raise MissingDataError(f"Could not infer an ORCA basis specification from {inp_path}.")
    return basis_tokens, block


def _looks_like_orca_basis_token(token: str) -> bool:
    low = token.strip(",").lower()
    if "/" in low or low in {"rijcosx", "cosx", "autoaux", "autoauxri"}:
        return False
    markers = (
        "def2",
        "cc-pv",
        "cc-pwcv",
        "aug-cc",
        "jun-cc",
        "may-cc",
        "ano",
        "sarc",
        "ma-",
        "dhf-",
        "jorge",
        "pcseg",
        "x2c-",
        "zora-",
        "dkh-",
        "iglo",
        "epr-",
        "wachters",
        "lanl",
        "cep-",
    )
    if any(marker in low for marker in markers):
        return True
    if low in {"svp", "sv(p)", "tzv", "tzvp", "tzvpp", "qzv", "qzvp", "qzvpp", "minao"}:
        return True
    return bool(re.match(r"^(?:sto-\d+g|[346]-\d{2,3}(?:\+{0,2})g(?:\*{0,2}|\(d(?:,p)?\))?)$", low))


def _extract_percent_block(text: str, name: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if re.match(rf"^%{re.escape(name)}\b", stripped, flags=re.IGNORECASE):
            start = index
            break
    if start is None:
        return ""

    out: List[str] = []
    depth = 1
    for index in range(start, len(lines)):
        line = lines[index]
        out.append(line)
        words = _orca_words(line)
        if index == start and words and words[0].casefold() == f"%{name}".casefold():
            words = words[1:]
        nested = sum(1 for word in words if _is_nested_basis_directive(word))
        ends = sum(1 for word in words if word.casefold() == "end")
        depth += nested - ends
        if depth <= 0:
            return "\n".join(out)
    raise MissingDataError(f"Unterminated %{name} block in ORCA input.")


def _orca_words(line: str) -> List[str]:
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError:
        return line.split("#", 1)[0].split()


def _is_nested_basis_directive(word: str) -> bool:
    low = word.casefold()
    return low == "newecp" or ((low.startswith("new") or low.startswith("add")) and "gto" in low)


def _canonical_basis_spec(tokens: Sequence[str], block: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    canonical_tokens = tuple(token.casefold() for token in tokens)
    canonical_block = tuple(" ".join(_orca_words(line)) for line in block.splitlines() if _orca_words(line))
    return canonical_tokens, canonical_block


def _auxiliary_input_digest(cross_input: str) -> str:
    return hashlib.sha256(("tdentrack-orca-cross-v2\0" + cross_input).encode("utf-8")).hexdigest()[:16]


def _safe_path_component(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
    return (safe or "step")[:80]


def _orca_terminated_normally(stdout: str) -> bool:
    return _ORCA_NORMAL_TERMINATION in (stdout or "").upper()


def _validated_square_matrix(matrix: np.ndarray, description: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise MissingDataError(f"{description} must be a non-empty square matrix; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise MissingDataError(f"{description} contains non-finite values.")
    return array


def _cleanup_auxiliary_workdir(work: Path) -> None:
    for pattern in ("*.tmp", "*.tmp.*", "*.grid.tmp", "*.shark_grid.tmp", "*.propint.tmp.*", "*.cpscfdata.tmp.*"):
        for path in Path(work).glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass

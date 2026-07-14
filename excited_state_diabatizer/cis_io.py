from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import CISAmplitudeCheck, MissingDataError
from .orca_io import parse_ci_coefficients

HEADER_INTS = 15
HEADER_BYTES = HEADER_INTS * 4
LEADING_DOUBLES = 2
INTER_ROOT_DOUBLES = 5


@dataclass(frozen=True)
class CISHeader:
    nroots: int
    alpha_occ_start: int
    alpha_occ_end: int
    alpha_virt_start: int
    alpha_virt_end: int
    beta_occ_start: int
    beta_occ_end: int
    beta_virt_start: int
    beta_virt_end: int
    vector_length: int
    raw_ints: Tuple[int, ...]

    @property
    def alpha_nocc(self) -> int:
        return self.alpha_occ_end - self.alpha_occ_start + 1

    @property
    def alpha_nvirt(self) -> int:
        return self.alpha_virt_end - self.alpha_virt_start + 1

    @property
    def beta_nocc(self) -> int:
        return self.beta_occ_end - self.beta_occ_start + 1

    @property
    def beta_nvirt(self) -> int:
        return self.beta_virt_end - self.beta_virt_start + 1

    @property
    def alpha_size(self) -> int:
        return self.alpha_nocc * self.alpha_nvirt

    @property
    def beta_size(self) -> int:
        return self.beta_nocc * self.beta_nvirt

    @property
    def expected_vector_length(self) -> int:
        return self.alpha_size + self.beta_size

    def as_jsonable(self) -> dict:
        out = asdict(self)
        out["raw_ints"] = list(self.raw_ints)
        out.update(
            {
                "alpha_shape": [self.alpha_nocc, self.alpha_nvirt],
                "beta_shape": [self.beta_nocc, self.beta_nvirt],
                "alpha_size": self.alpha_size,
                "beta_size": self.beta_size,
            }
        )
        return out


def read_cis_header(path: Path) -> CISHeader:
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        raise MissingDataError(f"{path} is too small to contain an ORCA CIS header.")
    ints = struct.unpack("<" + "i" * HEADER_INTS, raw)
    header = CISHeader(
        nroots=ints[0],
        alpha_occ_start=ints[1],
        alpha_occ_end=ints[2],
        alpha_virt_start=ints[3],
        alpha_virt_end=ints[4],
        beta_occ_start=ints[5],
        beta_occ_end=ints[6],
        beta_virt_start=ints[7],
        beta_virt_end=ints[8],
        vector_length=ints[9],
        raw_ints=tuple(ints),
    )
    _validate_header(path, header)
    return header


def parse_cis_amplitudes(path: Path, roots: Optional[Iterable[int]] = None) -> Tuple[CISHeader, Dict[int, dict]]:
    """Parse ORCA 6-style unrestricted TDA/CIS amplitude vectors from job.cis.

    This reader intentionally supports the layout observed for ORCA TD-DFT/TDA
    binary CI-vector files: a 15-int32 little-endian header, two leading doubles,
    one active occupied-virtual amplitude vector per root, and five separator
    doubles between adjacent root vectors. It fails if the file-size arithmetic
    does not match those assumptions.
    """

    path = Path(path)
    header = read_cis_header(path)
    data = _read_payload(path, header)
    wanted = list(range(1, header.nroots + 1)) if roots is None else sorted(set(int(r) for r in roots))
    out: Dict[int, dict] = {}
    for root in wanted:
        if root < 1 or root > header.nroots:
            continue
        vec = _root_vector(data, header, root)
        alpha = vec[: header.alpha_size].reshape((header.alpha_nocc, header.alpha_nvirt)).copy()
        beta = vec[header.alpha_size : header.alpha_size + header.beta_size].reshape((header.beta_nocc, header.beta_nvirt)).copy()
        out[root] = {
            "root": root,
            "alpha": alpha,
            "beta": beta,
            "alpha_occ_range": (header.alpha_occ_start, header.alpha_occ_end),
            "alpha_virt_range": (header.alpha_virt_start, header.alpha_virt_end),
            "beta_occ_range": (header.beta_occ_start, header.beta_occ_end),
            "beta_virt_range": (header.beta_virt_start, header.beta_virt_end),
            "source_type": "orca-cis-binary",
            "source_file": path,
        }
    return header, out


def coefficient_from_amplitudes(header: CISHeader, amplitudes: dict, donor: int, donor_spin: str, acceptor: int, acceptor_spin: str) -> float:
    spin = _normalize_spin(donor_spin or acceptor_spin)
    acceptor_spin_norm = _normalize_spin(acceptor_spin or donor_spin)
    if spin != acceptor_spin_norm:
        raise MissingDataError(
            f"Spin-changing CIS contribution {donor}{donor_spin}->{acceptor}{acceptor_spin} is not supported by this overlap engine."
        )
    if spin == "alpha":
        i = donor - header.alpha_occ_start
        a = acceptor - header.alpha_virt_start
        arr = amplitudes["alpha"]
        if i < 0 or i >= header.alpha_nocc or a < 0 or a >= header.alpha_nvirt:
            raise MissingDataError(
                f"Alpha transition {donor}{donor_spin}->{acceptor}{acceptor_spin} lies outside CIS active ranges "
                f"{header.alpha_occ_start}..{header.alpha_occ_end} to {header.alpha_virt_start}..{header.alpha_virt_end}."
            )
        return float(arr[i, a])
    i = donor - header.beta_occ_start
    a = acceptor - header.beta_virt_start
    arr = amplitudes["beta"]
    if i < 0 or i >= header.beta_nocc or a < 0 or a >= header.beta_nvirt:
        raise MissingDataError(
            f"Beta transition {donor}{donor_spin}->{acceptor}{acceptor_spin} lies outside CIS active ranges "
            f"{header.beta_occ_start}..{header.beta_occ_end} to {header.beta_virt_start}..{header.beta_virt_end}."
        )
    return float(arr[i, a])


def validate_cis_against_output(
    cis_path: Path,
    out_path: Path,
    step_label: str,
    roots: Sequence[int],
    max_checks_per_root: int = 12,
    tolerance: float = 5.0e-7,
) -> Tuple[CISHeader, Dict[int, dict], List[CISAmplitudeCheck]]:
    header, amplitudes = parse_cis_amplitudes(cis_path, roots)
    text = Path(out_path).read_text(errors="replace") if Path(out_path).exists() else ""
    printed = parse_ci_coefficients(text)
    rows: List[CISAmplitudeCheck] = []
    for root in roots:
        root_amps = amplitudes.get(root)
        if root_amps is None:
            continue
        for contrib in printed.get(root, [])[:max_checks_per_root]:
            try:
                binary = coefficient_from_amplitudes(
                    header,
                    root_amps,
                    contrib.donor,
                    contrib.donor_spin,
                    contrib.acceptor,
                    contrib.acceptor_spin,
                )
                err = abs(binary - contrib.coefficient)
                passed = err <= tolerance
                msg = "" if passed else f"|binary-printed|={err:.3g} exceeds {tolerance:.3g}"
            except Exception as exc:
                binary = None
                err = None
                passed = False
                msg = str(exc)
            rows.append(
                CISAmplitudeCheck(
                    step_label=step_label,
                    root=root,
                    source_file=Path(cis_path),
                    donor=contrib.donor,
                    donor_spin=contrib.donor_spin,
                    acceptor=contrib.acceptor,
                    acceptor_spin=contrib.acceptor_spin,
                    printed_coefficient=contrib.coefficient,
                    binary_coefficient=binary,
                    abs_error=err,
                    passed=passed,
                    message=msg,
                )
            )
    return header, amplitudes, rows


def write_cis_manifest(path: Path, cis_path: Path, header: CISHeader, roots: Sequence[int], checks: Sequence[CISAmplitudeCheck]) -> None:
    path = Path(path)
    n_checks = len(checks)
    n_pass = sum(1 for row in checks if row.passed)
    payload = {
        "format": "orca-cis-tden-manifest-v1",
        "source_file": str(Path(cis_path)),
        "description": "Manifest for ORCA job.cis TDDFT/TDA amplitudes parsed by excited_state_diabatizer. Dense amplitude arrays are kept in memory, not duplicated into JSON.",
        "header": header.as_jsonable(),
        "roots": list(map(int, roots)),
        "validation": {
            "checked_coefficients": n_checks,
            "passed_coefficients": n_pass,
            "failed_coefficients": n_checks - n_pass,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_orca_2json_full_config_template(path: Path) -> None:
    payload = {
        "MOCoefficients": True,
        "Basisset": True,
        "1elIntegrals": ["S"],
        "JSONFormats": ["json"],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_orca_2json_overlap_config(path: Path) -> None:
    payload = {
        "MOCoefficients": False,
        "Basisset": False,
        "1elIntegrals": ["S"],
        "JSONFormats": ["json"],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_orca_2json_mo_only_config(path: Path) -> None:
    payload = {
        "JSONFormats": ["json"],
        "MOCoefficients": True,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_header(path: Path, header: CISHeader) -> None:
    if header.nroots <= 0:
        raise MissingDataError(f"{path} has an invalid CIS root count in its header: {header.nroots}.")
    ranges = [
        ("alpha occupied", header.alpha_occ_start, header.alpha_occ_end),
        ("alpha virtual", header.alpha_virt_start, header.alpha_virt_end),
        ("beta occupied", header.beta_occ_start, header.beta_occ_end),
        ("beta virtual", header.beta_virt_start, header.beta_virt_end),
    ]
    for name, start, end in ranges:
        if end < start:
            raise MissingDataError(f"{path} has an invalid {name} CIS orbital range: {start}..{end}.")
    if header.expected_vector_length <= 0:
        raise MissingDataError(f"{path} has an empty CIS vector length implied by its active orbital ranges.")
    if header.vector_length != header.expected_vector_length:
        raise MissingDataError(
            f"{path} CIS vector length mismatch: header says {header.vector_length}, "
            f"active alpha/beta ranges imply {header.expected_vector_length}."
        )


def _read_payload(path: Path, header: CISHeader) -> np.ndarray:
    size = Path(path).stat().st_size
    expected_doubles = LEADING_DOUBLES + header.nroots * header.vector_length + max(0, header.nroots - 1) * INTER_ROOT_DOUBLES
    expected_size = HEADER_BYTES + 8 * expected_doubles
    if size != expected_size:
        raise MissingDataError(
            f"{path} size does not match the supported ORCA CIS layout: got {size} bytes, expected {expected_size} bytes "
            f"from header ranges and {header.nroots} roots."
        )
    with Path(path).open("rb") as f:
        f.seek(HEADER_BYTES)
        data = np.fromfile(f, dtype="<f8", count=expected_doubles)
    if data.size != expected_doubles:
        raise MissingDataError(f"{path} ended before all CIS amplitude data could be read.")
    return data


def _root_vector(data: np.ndarray, header: CISHeader, root: int) -> np.ndarray:
    start = LEADING_DOUBLES + (root - 1) * (header.vector_length + INTER_ROOT_DOUBLES)
    end = start + header.vector_length
    return data[start:end]


def _normalize_spin(spin: object) -> str:
    text = str(spin or "a").strip().lower()
    return "alpha" if text.startswith("a") else "beta"

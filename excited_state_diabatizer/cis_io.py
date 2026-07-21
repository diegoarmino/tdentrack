from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import CISAmplitudeCheck, MissingDataError
from .orca_io import parse_ci_coefficients, parse_tddft_output

HEADER_INTS = 15
HEADER_BYTES = HEADER_INTS * 4
LEADING_DOUBLES = 2
INTER_ROOT_DOUBLES = 5
ORBITAL_INDEX_BASE = 0
STANDARD_HEADER_INTS = 9
STANDARD_HEADER_BYTES = STANDARD_HEADER_INTS * 4
STANDARD_VECTOR_HEADER_INTS = 6
STANDARD_VECTOR_HEADER_BYTES = STANDARD_VECTOR_HEADER_INTS * 4
STANDARD_VECTOR_PREFIX_BYTES = STANDARD_VECTOR_HEADER_BYTES + 2 * 8


class _RecognizedStandardCISError(MissingDataError):
    """Semantic failure after a complete standard vector stream was decoded."""


@dataclass(frozen=True)
class CISHeader:
    """Normalized header of a supported ORCA ``.cis`` binary layout.

    ORCA stores the four active-orbital ranges as zero-based, inclusive MO
    indices.  TDenTrack keeps that convention internally so the ranges can be
    used directly to slice MO coefficient matrices (with ``end + 1``).  For a
    restricted standard ORCA file the beta ranges are absent on disk (all four
    values are ``-1``); the normalized header mirrors the alpha ranges so that
    the reconstructed alpha and beta transition-density blocks have explicit,
    usable ranges.
    """

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
    layout: str = "orca-legacy-15int-tda"
    restricted: bool = False
    tda: bool = True
    multiplicities: Tuple[int, ...] = ()
    stored_vectors: Optional[int] = None

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
        # A restricted standard ORCA file stores one spatial excitation vector.
        # The two normalized spin blocks are reconstructed after reading it.
        return self.alpha_size if self.restricted else self.alpha_size + self.beta_size

    def as_jsonable(self) -> dict:
        out = asdict(self)
        out["raw_ints"] = list(self.raw_ints)
        out["orbital_index_base"] = ORBITAL_INDEX_BASE
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
    """Read and normalize either supported ORCA ``.cis`` header variant."""

    path = Path(path)
    try:
        header, _ = _parse_standard_cis(path)
        return header
    except _RecognizedStandardCISError:
        raise
    except MissingDataError:
        legacy = _read_legacy_header_if_exact(path)
        if legacy is not None:
            return legacy
        raise


def parse_cis_amplitudes(
    path: Path,
    roots: Optional[Iterable[int]] = None,
    multiplicity: Optional[int] = None,
    tda: Optional[bool] = None,
) -> Tuple[CISHeader, Dict[int, dict]]:
    """Parse supported ORCA 6 CIS/TDA amplitude vectors from ``job.cis``.

    Two layouts are recognized without a user switch:

    * the historical 15-int header/unrestricted-TDA layout already used by
      TDenTrack; and
    * ORCA's standard vector-record layout, including the restricted TDA files
      written by ORCA 6.1.1.

    Root keys always follow the global ``STATE N`` order printed by ORCA, even
    when the file contains consecutive singlet and triplet blocks.  Passing
    ``multiplicity`` filters those global roots after parsing and also provides
    an explicit guard against mixing spin manifolds. The optional tda hint
    resolves the otherwise binary-ambiguous case of one non-TDA X+Y/X-Y pair
    versus two exactly degenerate TDA records. Dictionary keys and global_root
    are printed mixed-manifold STATE labels. For unrestricted open-shell
    gradients this global label is ORCA's IRoot. The separate
    multiplicity-local orca_gradient_iroot value is used only when selecting a
    spin-adapted multiplicity block, such as restricted-reference triplets
    with IRootMult triplet.
    """

    path = Path(path)
    if multiplicity is not None and int(multiplicity) <= 0:
        raise ValueError(f"CIS multiplicity must be a positive integer, got {multiplicity!r}.")
    multiplicity = None if multiplicity is None else int(multiplicity)
    try:
        header, parsed = _parse_standard_cis(path, tda_hint=tda)
        wanted = None if roots is None else {int(root) for root in roots}
        out = {
            root: state
            for root, state in parsed.items()
            if (wanted is None or root in wanted)
            and (multiplicity is None or state["multiplicity"] == multiplicity)
        }
        return header, out
    except _RecognizedStandardCISError:
        raise
    except MissingDataError:
        legacy = _read_legacy_header_if_exact(path)
        if legacy is None:
            raise

    header = legacy
    if tda is False:
        raise MissingDataError(
            f"{path} matches the historical TDA-only CIS layout, but the caller "
            "declared a non-TDA calculation."
        )
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
            "orbital_index_base": ORBITAL_INDEX_BASE,
            "source_type": "orca-cis-binary",
            "source_file": path,
            "cis_layout": header.layout,
            # The legacy 15-int layout does not encode multiplicity. Keep the
            # caller's requested filter separately instead of presenting it as
            # a binary fact.
            "multiplicity": None,
            "requested_multiplicity": multiplicity,
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
        return float(arr[i, a]) * float(amplitudes.get("printed_coefficient_scale", 1.0))
    i = donor - header.beta_occ_start
    a = acceptor - header.beta_virt_start
    arr = amplitudes["beta"]
    if i < 0 or i >= header.beta_nocc or a < 0 or a >= header.beta_nvirt:
        raise MissingDataError(
            f"Beta transition {donor}{donor_spin}->{acceptor}{acceptor_spin} lies outside CIS active ranges "
            f"{header.beta_occ_start}..{header.beta_occ_end} to {header.beta_virt_start}..{header.beta_virt_end}."
        )
    return float(arr[i, a]) * float(amplitudes.get("printed_coefficient_scale", 1.0))


def validate_cis_against_output(
    cis_path: Path,
    out_path: Path,
    step_label: str,
    roots: Sequence[int],
    max_checks_per_root: int = 12,
    tolerance: float = 5.0e-7,
    energy_tolerance_eh: float = 5.0e-6,
) -> Tuple[CISHeader, Dict[int, dict], List[CISAmplitudeCheck]]:
    text = Path(out_path).read_text(errors="replace") if Path(out_path).exists() else ""
    _, output_states, _ = parse_tddft_output(out_path, step_label) if text else (None, {}, False)
    output_multiplicities = {
        output_states[root].multiplicity
        for root in roots
        if root in output_states and output_states[root].multiplicity is not None
    }
    multiplicity = next(iter(output_multiplicities)) if len(output_multiplicities) == 1 else None
    tda_hint = _tda_hint_from_output(text)
    header, amplitudes = parse_cis_amplitudes(
        cis_path,
        roots,
        multiplicity=multiplicity,
        tda=tda_hint,
    )
    missing = sorted(set(map(int, roots)) - set(amplitudes))
    if missing:
        qualifier = ""
        if multiplicity is not None:
            qualifier = f" in multiplicity {multiplicity}"
        raise MissingDataError(
            f"{cis_path} does not contain requested global ORCA root(s) {missing}{qualifier}; "
            f"available multiplicities are {list(header.multiplicities) or 'not encoded'} and the file contains {header.nroots} root(s)."
        )
    for root, root_amps in amplitudes.items():
        binary_mult = root_amps.get("multiplicity")
        output_mult = output_states.get(root).multiplicity if root in output_states else None
        if binary_mult is not None and output_mult is not None and int(binary_mult) != int(output_mult):
            raise MissingDataError(
                f"{cis_path} maps global root {root} to multiplicity {binary_mult}, but {out_path} prints multiplicity {output_mult}."
            )
        binary_energy = root_amps.get("excitation_energy_eh")
        output_energy = output_states.get(root).exc_au if root in output_states else None
        if (
            binary_energy is not None
            and output_energy is not None
            and abs(float(binary_energy) - float(output_energy)) > energy_tolerance_eh
        ):
            raise MissingDataError(
                f"{cis_path} maps global root {root} to excitation energy "
                f"{float(binary_energy):.12g} Eh, but {out_path} prints "
                f"{float(output_energy):.12g} Eh (tolerance {energy_tolerance_eh:.3g} Eh)."
            )
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


def write_cis_manifest(
    path: Path,
    cis_path: Path,
    header: CISHeader,
    roots: Sequence[int],
    checks: Sequence[CISAmplitudeCheck],
    amplitudes: Optional[Dict[int, dict]] = None,
) -> None:
    path = Path(path)
    n_checks = len(checks)
    n_pass = sum(1 for row in checks if row.passed)
    payload = {
        "format": "orca-cis-tden-manifest-v1",
        "source_file": str(Path(cis_path)),
        "description": "Manifest for ORCA job.cis TDDFT/TDA amplitudes parsed by excited_state_diabatizer. Dense amplitude arrays are kept in memory, not duplicated into JSON.",
        "header": header.as_jsonable(),
        "roots": list(map(int, roots)),
        "states": [
            {
                key: state.get(key)
                for key in (
                    "root",
                    "global_root",
                    "multiplicity",
                    "root_within_multiplicity",
                    "orca_gradient_iroot",
                    "excitation_energy_eh",
                    "orca_root_index",
                    "vector_record_index",
                    "tda",
                    "response_component",
                    "restricted",
                    "restricted_spin_reconstruction",
                    "cis_layout",
                    "orbital_index_base",
                )
                if key in state
            }
            for _, state in sorted((amplitudes or {}).items())
        ],
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


@dataclass(frozen=True)
class _StandardVectorRecord:
    record_index: int
    ncoeffs: int
    multiplicity: int
    orca_root_index: int
    energy_eh: float
    auxiliary_value: float
    raw_ints: Tuple[int, ...]
    coefficients: np.ndarray


def _read_legacy_header_if_exact(path: Path) -> Optional[CISHeader]:
    """Return a legacy header only when its complete size arithmetic matches.

    Standard ORCA vector-record files also begin with integer orbital ranges,
    so recognizing a legacy file from a prefix alone is unsafe. The exact byte
    count is an intentional part of the legacy format discriminator.
    """

    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MissingDataError(f"Cannot stat ORCA CIS file {path}: {exc}") from exc
    if size < HEADER_BYTES:
        return None
    with path.open("rb") as handle:
        raw = handle.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        return None
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
        layout="orca-legacy-15int-tda",
        restricted=False,
        tda=True,
        stored_vectors=ints[0],
    )
    try:
        _validate_header(path, header)
    except MissingDataError:
        return None
    expected_doubles = (
        LEADING_DOUBLES
        + header.nroots * header.vector_length
        + max(0, header.nroots - 1) * INTER_ROOT_DOUBLES
    )
    expected_size = HEADER_BYTES + 8 * expected_doubles
    return header if size == expected_size else None


def _parse_standard_cis(
    path: Path,
    tda_hint: Optional[bool] = None,
) -> Tuple[CISHeader, Dict[int, dict]]:
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MissingDataError(f"Cannot stat ORCA CIS file {path}: {exc}") from exc
    if size < STANDARD_HEADER_BYTES + STANDARD_VECTOR_PREFIX_BYTES:
        raise MissingDataError(
            f"{path} is too small for either supported ORCA CIS layout ({size} bytes)."
        )

    with path.open("rb") as handle:
        header_raw = handle.read(STANDARD_HEADER_BYTES)
        if len(header_raw) != STANDARD_HEADER_BYTES:
            raise MissingDataError(f"{path} ended while reading its standard ORCA CIS header.")
        raw_ints = struct.unpack("<" + "i" * STANDARD_HEADER_INTS, header_raw)
        nvec = raw_ints[0]
        if nvec <= 0:
            raise MissingDataError(f"{path} has an invalid standard CIS vector count: {nvec}.")
        # Every vector has at least a 40-byte prefix and one coefficient. This
        # rejects corrupt counts before they can drive a large loop.
        max_vectors = (size - STANDARD_HEADER_BYTES) // (STANDARD_VECTOR_PREFIX_BYTES + 8)
        if nvec > max_vectors:
            raise MissingDataError(
                f"{path} declares {nvec} standard CIS vectors, but at most {max_vectors} can fit in {size} bytes."
            )

        alpha_ranges = tuple(raw_ints[1:5])
        beta_ranges_raw = tuple(raw_ints[5:9])
        _validate_raw_range(path, "alpha occupied", alpha_ranges[0], alpha_ranges[1])
        _validate_raw_range(path, "alpha virtual", alpha_ranges[2], alpha_ranges[3])
        restricted = beta_ranges_raw == (-1, -1, -1, -1)
        if not restricted:
            if any(value == -1 for value in beta_ranges_raw):
                raise _RecognizedStandardCISError(
                    f"{path} has a partially absent beta CIS header {beta_ranges_raw}; "
                    "expected four valid ranges or four -1 sentinels."
                )
            _validate_raw_range(path, "beta occupied", beta_ranges_raw[0], beta_ranges_raw[1])
            _validate_raw_range(path, "beta virtual", beta_ranges_raw[2], beta_ranges_raw[3])

        alpha_size = (
            (alpha_ranges[1] - alpha_ranges[0] + 1)
            * (alpha_ranges[3] - alpha_ranges[2] + 1)
        )
        if restricted:
            beta_ranges = alpha_ranges
            stored_vector_length = alpha_size
        else:
            beta_ranges = beta_ranges_raw
            beta_size = (
                (beta_ranges[1] - beta_ranges[0] + 1)
                * (beta_ranges[3] - beta_ranges[2] + 1)
            )
            stored_vector_length = alpha_size + beta_size

        records = _read_standard_vector_records(
            path,
            handle,
            nvec,
            stored_vector_length,
            size,
        )

    tda = bool(tda_hint) if tda_hint is not None else _infer_standard_tda(path, records)
    state_records = _group_standard_state_records(path, records, tda)
    multiplicities = tuple(sorted({primary.multiplicity for primary, _ in state_records}))
    header = CISHeader(
        nroots=len(state_records),
        alpha_occ_start=alpha_ranges[0],
        alpha_occ_end=alpha_ranges[1],
        alpha_virt_start=alpha_ranges[2],
        alpha_virt_end=alpha_ranges[3],
        beta_occ_start=beta_ranges[0],
        beta_occ_end=beta_ranges[1],
        beta_virt_start=beta_ranges[2],
        beta_virt_end=beta_ranges[3],
        vector_length=stored_vector_length,
        raw_ints=tuple(raw_ints),
        layout="orca-standard-vector-records",
        restricted=restricted,
        tda=tda,
        multiplicities=multiplicities,
        stored_vectors=nvec,
    )
    _validate_header(path, header)

    states: Dict[int, dict] = {}
    roots_within_multiplicity: Dict[int, int] = {}
    restricted_scale = float(1.0 / np.sqrt(2.0))
    for global_root, (primary, secondary) in enumerate(state_records, start=1):
        roots_within_multiplicity[primary.multiplicity] = (
            roots_within_multiplicity.get(primary.multiplicity, 0) + 1
        )
        root_within_multiplicity = roots_within_multiplicity[primary.multiplicity]
        alpha, beta = _split_standard_coefficients(primary.coefficients, header)
        alpha_minus_y = beta_minus_y = None
        if secondary is not None:
            alpha_minus_y, beta_minus_y = _split_standard_coefficients(
                secondary.coefficients, header
            )
        printed_scale = 1.0
        restricted_spin_reconstruction = None
        if restricted:
            # ORCA stores a normalized spin-adapted spatial vector. Construct
            # two equal spin blocks with total norm one; coefficient validation
            # rescales either block back to ORCA's printed spatial coefficient.
            alpha = alpha * restricted_scale
            beta = alpha.copy()
            if alpha_minus_y is not None:
                alpha_minus_y = alpha_minus_y * restricted_scale
                beta_minus_y = alpha_minus_y.copy()
            printed_scale = float(np.sqrt(2.0))
            restricted_spin_reconstruction = (
                "alpha/beta copies of the spatial vector, each scaled by 1/sqrt(2)"
            )

        state = {
            "root": global_root,
            "global_root": global_root,
            "alpha": alpha,
            "beta": beta,
            "alpha_occ_range": (header.alpha_occ_start, header.alpha_occ_end),
            "alpha_virt_range": (header.alpha_virt_start, header.alpha_virt_end),
            "beta_occ_range": (header.beta_occ_start, header.beta_occ_end),
            "beta_virt_range": (header.beta_virt_start, header.beta_virt_end),
            "orbital_index_base": ORBITAL_INDEX_BASE,
            "source_type": "orca-cis-binary",
            "source_file": path,
            "cis_layout": header.layout,
            "multiplicity": primary.multiplicity,
            "root_within_multiplicity": root_within_multiplicity,
            "orca_gradient_iroot": root_within_multiplicity,
            "excitation_energy_eh": primary.energy_eh,
            "orca_root_index": primary.orca_root_index,
            "vector_record_index": primary.record_index,
            "tda": tda,
            "response_component": "tda-x" if tda else "x-plus-y",
            "restricted": restricted,
            "restricted_spin_reconstruction": restricted_spin_reconstruction,
            "printed_coefficient_scale": printed_scale,
        }
        if secondary is not None:
            state.update(
                {
                    "alpha_x_minus_y": alpha_minus_y,
                    "beta_x_minus_y": beta_minus_y,
                    "x_minus_y_vector_record_index": secondary.record_index,
                }
            )
        states[global_root] = state
    return header, states


def _read_standard_vector_records(
    path: Path,
    handle,
    nvec: int,
    expected_ncoeffs: int,
    file_size: int,
) -> List[_StandardVectorRecord]:
    records: List[_StandardVectorRecord] = []
    for record_index in range(nvec):
        raw = handle.read(STANDARD_VECTOR_PREFIX_BYTES)
        if len(raw) != STANDARD_VECTOR_PREFIX_BYTES:
            raise MissingDataError(
                f"{path} ended while reading standard CIS vector record "
                f"{record_index + 1}/{nvec}."
            )
        raw_record_ints = struct.unpack(
            "<" + "i" * STANDARD_VECTOR_HEADER_INTS,
            raw[:STANDARD_VECTOR_HEADER_BYTES],
        )
        ncoeffs, _, multiplicity, _, orca_root_index, _ = raw_record_ints
        energy_eh, auxiliary_value = struct.unpack(
            "<2d", raw[STANDARD_VECTOR_HEADER_BYTES:]
        )
        if ncoeffs != expected_ncoeffs:
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} declares "
                f"{ncoeffs} coefficients; the active orbital ranges require "
                f"{expected_ncoeffs}. Spin-flip and mixed-layout vectors are not supported."
            )
        if multiplicity <= 0:
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} has invalid "
                f"multiplicity {multiplicity}."
            )
        if orca_root_index < 0:
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} has invalid "
                f"ORCA root index {orca_root_index}."
            )
        # The second double is undocumented and is not interpreted; ORCA has
        # written bit-pattern-like/subnormal values there. Validate only the
        # excitation energy that TDenTrack actually consumes.
        if not np.isfinite(energy_eh):
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} contains a "
                "non-finite excitation energy."
            )
        needed = ncoeffs * 8
        remaining = file_size - handle.tell()
        if needed > remaining:
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} needs {needed} "
                f"coefficient bytes, but only {remaining} remain."
            )
        coeff_raw = handle.read(needed)
        if len(coeff_raw) != needed:
            raise MissingDataError(
                f"{path} ended while reading coefficients for standard CIS "
                f"vector {record_index + 1}."
            )
        coefficients = np.frombuffer(coeff_raw, dtype="<f8").astype(float, copy=True)
        if coefficients.size != ncoeffs or not np.all(np.isfinite(coefficients)):
            raise MissingDataError(
                f"{path} standard CIS vector {record_index + 1} has missing or "
                "non-finite coefficients."
            )
        records.append(
            _StandardVectorRecord(
                record_index=record_index,
                ncoeffs=ncoeffs,
                multiplicity=multiplicity,
                orca_root_index=orca_root_index,
                energy_eh=float(energy_eh),
                auxiliary_value=float(auxiliary_value),
                raw_ints=tuple(raw_record_ints),
                coefficients=coefficients,
            )
        )
    trailing = handle.read(1)
    if trailing:
        raise _RecognizedStandardCISError(
            f"{path} contains trailing bytes after its declared {nvec} standard "
            "CIS vectors; refusing an ambiguous layout."
        )
    return records


def _infer_standard_tda(
    path: Path, records: Sequence[_StandardVectorRecord]
) -> bool:
    if len(records) == 1:
        return True
    # Prefer a multiplicity block with at least two vectors; ORCA 6.1.1 may stop
    # updating its stored triplet iroot after a preceding singlet block, so iroot
    # alone is not a safe discriminator there.
    for index, first in enumerate(records[:-1]):
        second = records[index + 1]
        if first.multiplicity != second.multiplicity:
            continue
        if second.orca_root_index == first.orca_root_index + 1:
            return True
        same_energy = np.isclose(
            first.energy_eh,
            second.energy_eh,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        if second.orca_root_index == first.orca_root_index and same_energy:
            if len(records) == 2:
                raise _RecognizedStandardCISError(
                    f"{path} contains two same-energy records with the same "
                    "multiplicity and stored iroot. This can be either one "
                    "non-TDA X+Y/X-Y pair or two exactly degenerate TDA roots; "
                    "pass tda=True/False or validate against job.out."
                )
            return False
        if second.orca_root_index == first.orca_root_index and not same_energy:
            return True
        raise _RecognizedStandardCISError(
            f"{path} has an unsupported standard CIS root sequence: vector "
            f"records {index} and {index + 1} store iroot "
            f"{first.orca_root_index} and {second.orca_root_index}."
        )
    # One vector per multiplicity can only represent TDA states; an RPA/TDDFT
    # state requires adjacent X+Y and X-Y records of the same multiplicity.
    return True


def _group_standard_state_records(
    path: Path,
    records: Sequence[_StandardVectorRecord],
    tda: bool,
) -> List[Tuple[_StandardVectorRecord, Optional[_StandardVectorRecord]]]:
    if tda:
        return [(record, None) for record in records]
    if len(records) % 2:
        raise _RecognizedStandardCISError(
            f"{path} contains an odd number ({len(records)}) of non-TDA CIS "
            "vectors; expected X+Y/X-Y pairs."
        )
    states: List[Tuple[_StandardVectorRecord, Optional[_StandardVectorRecord]]] = []
    for index in range(0, len(records), 2):
        plus = records[index]
        minus = records[index + 1]
        if plus.multiplicity != minus.multiplicity:
            raise _RecognizedStandardCISError(
                f"{path} non-TDA CIS vector pair {index // 2 + 1} crosses "
                f"multiplicities {plus.multiplicity} and {minus.multiplicity}."
            )
        if plus.orca_root_index != minus.orca_root_index:
            raise _RecognizedStandardCISError(
                f"{path} non-TDA CIS vector pair {index // 2 + 1} has different "
                f"ORCA root indices {plus.orca_root_index} and "
                f"{minus.orca_root_index}."
            )
        if not np.isclose(
            plus.energy_eh,
            minus.energy_eh,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise _RecognizedStandardCISError(
                f"{path} non-TDA CIS vector pair {index // 2 + 1} has "
                f"inconsistent energies {plus.energy_eh:.16g} and "
                f"{minus.energy_eh:.16g} Eh."
            )
        states.append((plus, minus))
    return states


def _split_standard_coefficients(
    coefficients: np.ndarray, header: CISHeader
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = coefficients[: header.alpha_size].reshape(
        (header.alpha_nocc, header.alpha_nvirt)
    ).copy()
    if header.restricted:
        return alpha, alpha.copy()
    beta = coefficients[
        header.alpha_size : header.alpha_size + header.beta_size
    ]
    beta = beta.reshape((header.beta_nocc, header.beta_nvirt)).copy()
    return alpha, beta


def _validate_raw_range(path: Path, name: str, start: int, end: int) -> None:
    if start < ORBITAL_INDEX_BASE or end < start:
        raise MissingDataError(
            f"{path} has an invalid {name} CIS orbital range: {start}..{end}."
        )


def _tda_hint_from_output(text: str) -> Optional[bool]:
    hint: Optional[bool] = None
    for line in text.splitlines():
        if "tamm-dancoff approximation" not in line.lower():
            continue
        status = line.split("...", 1)[-1].strip().lower()
        if status.startswith(("not operative", "off", "false")):
            hint = False
        elif status.startswith(("operative", "on", "true")):
            hint = True
    return hint


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
        if start < ORBITAL_INDEX_BASE or end < start:
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

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .json_io import parse_orca_json
from .models import NTOOrbitalPair, NTOVisualizationRecord, StepData
from .orca_io import parse_int_range

ANGSTROM_TO_BOHR = 1.889725988578923
SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}


class _PyscfContext:
    def __init__(
        self,
        mol,
        orca_to_pyscf: np.ndarray,
        orca_signs: np.ndarray,
        overlap_max_abs_error: float,
        overlap_rms_error: float,
    ) -> None:
        self.mol = mol
        self.orca_to_pyscf = np.asarray(orca_to_pyscf, dtype=int)
        self.orca_signs = np.asarray(orca_signs, dtype=float)
        self.overlap_max_abs_error = float(overlap_max_abs_error)
        self.overlap_rms_error = float(overlap_rms_error)


def render_self_nto_visualizations(
    steps: Sequence[StepData],
    roots: Sequence[int],
    outdir: Path,
    args,
    statuses: Optional[list] = None,
) -> List[NTOVisualizationRecord]:
    """Write validated self-derived NTO cube files and optional Jmol PNGs.

    The orbitals rendered here are the NTO coefficient vectors obtained by SVD
    of the same ORCA job.cis transition-amplitude matrices used by the tden-json
    tracking engine. ORCA .nto files are intentionally not read.
    """

    records: List[NTOVisualizationRecord] = []
    outdir = Path(outdir)
    cube_dir = outdir / "self_nto_cubes"
    image_dir = outdir / "self_nto_images"
    script_dir = outdir / "generated_inputs"
    log_dir = outdir / "utility_logs"
    for d in (cube_dir, image_dir, script_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    try:
        from pyscf import gto  # noqa: F401
    except Exception as exc:
        msg = (
            "PySCF is required for rigorous self-derived NTO cube generation. "
            "Install pyscf, then rerun with --render-self-nto-cubes."
        )
        records.append(
            NTOVisualizationRecord(
                step_label="",
                root=None,
                spin="",
                pair_index=None,
                side="",
                weight=None,
                status="failed",
                message=f"{msg} ({type(exc).__name__}: {exc})",
            )
        )
        return records

    selected_roots = _selected_roots(getattr(args, "self_nto_plot_roots", None), roots)
    selected_steps = _selected_steps(getattr(args, "self_nto_plot_steps", None), steps)
    grid = tuple(int(x) for x in getattr(args, "self_nto_cube_grid", (64, 64, 64)))
    max_pairs = int(getattr(args, "self_nto_plot_max_pairs", 2))
    pair_weight_min = float(getattr(args, "self_nto_plot_weight_min", 0.10))
    force = bool(getattr(args, "force_cubes", False))
    render_png = not bool(getattr(args, "self_nto_no_render_png", False))

    contexts: Dict[int, _PyscfContext] = {}
    for step in steps:
        if step.job.label not in selected_steps:
            continue
        context = _context_for_step(step, contexts, args)
        if isinstance(context, NTOVisualizationRecord):
            records.append(context)
            continue
        for root in selected_roots:
            vec = step.self_nto_vectors.get(root)
            if vec is None or not vec.pairs:
                records.append(
                    NTOVisualizationRecord(
                        step_label=step.job.label,
                        root=root,
                        spin="",
                        pair_index=None,
                        side="",
                        weight=None,
                        status="missing",
                        message="No in-memory self-derived NTO vectors are available for this root.",
                    )
                )
                continue
            pairs = _select_pairs(vec.pairs, pair_weight_min, max_pairs)
            if not pairs:
                records.append(
                    NTOVisualizationRecord(
                        step_label=step.job.label,
                        root=root,
                        spin="",
                        pair_index=None,
                        side="",
                        weight=None,
                        status="skipped",
                        message=f"No self-derived NTO pair passed visualization threshold {pair_weight_min:g}.",
                    )
                )
                continue
            for pair in pairs:
                records.extend(
                    _render_pair(
                        step,
                        root,
                        pair,
                        context,
                        cube_dir,
                        image_dir,
                        script_dir,
                        log_dir,
                        grid,
                        float(getattr(args, "self_nto_cube_margin_ang", 3.0)),
                        render_png,
                        force,
                        args,
                    )
                )
    return records


def _context_for_step(step: StepData, contexts: Dict[int, _PyscfContext], args):
    if step.job.order in contexts:
        return contexts[step.job.order]
    if step.json_file is None:
        return NTOVisualizationRecord(
            step_label=step.job.label,
            root=None,
            spin="",
            pair_index=None,
            side="",
            weight=None,
            status="failed",
            message="No ORCA job.json file is attached to this step; cannot build a validated AO grid.",
        )
    try:
        parsed = parse_orca_json(step.json_file)
        context = _build_pyscf_context(parsed.raw, parsed.overlap, float(getattr(args, "self_nto_cube_overlap_tol", 1.0e-6)))
    except Exception as exc:
        return NTOVisualizationRecord(
            step_label=step.job.label,
            root=None,
            spin="",
            pair_index=None,
            side="",
            weight=None,
            status="failed",
            message=str(exc),
        )
    contexts[step.job.order] = context
    return context


def _build_pyscf_context(raw: dict, orca_overlap: Optional[np.ndarray], tolerance: float) -> _PyscfContext:
    if orca_overlap is None:
        raise RuntimeError("ORCA S-Matrix is missing; refusing to generate self-derived NTO cubes without AO validation.")
    try:
        from pyscf import gto
    except Exception as exc:
        raise RuntimeError(f"PySCF is not available: {exc}") from exc

    mol_block = raw.get("Molecule", raw)
    atoms = mol_block.get("Atoms") or []
    atom_specs = []
    basis = {}
    for idx, atom in enumerate(atoms):
        symbol = atom.get("ElementLabel") or atom.get("label") or atom.get("symbol")
        coords = atom.get("Coords") or atom.get("coords")
        shells = atom.get("Basis") or []
        if not symbol or coords is None or not shells:
            raise RuntimeError("ORCA JSON atom records lack ElementLabel, Coords, or Basis entries.")
        label = f"{symbol}{idx}"
        atom_specs.append([label, [float(x) for x in coords]])
        basis[label] = [_shell_to_pyscf(shell) for shell in shells]

    charge = int(mol_block.get("Charge", 0) or 0)
    try:
        total_z = sum(int(gto.charge(_element_symbol(atom[0]))) for atom in atom_specs)
        nelec = total_z - charge
        spin = 0 if nelec % 2 == 0 else 1
    except Exception:
        spin = 0

    mol = gto.Mole()
    mol.atom = atom_specs
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.cart = False
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build(parse_arg=False)
    s_pyscf = np.asarray(mol.intor("int1e_ovlp"), dtype=float)
    s_orca = np.asarray(orca_overlap, dtype=float)
    if s_pyscf.shape != s_orca.shape:
        raise RuntimeError(
            f"PySCF AO overlap shape {s_pyscf.shape} does not match ORCA S-Matrix shape {s_orca.shape}; "
            "AO ordering or spherical/cartesian convention is not validated."
        )
    orca_to_pyscf = _shell_sequence_orca_to_pyscf_indices(atoms)
    if orca_to_pyscf.size != s_orca.shape[0]:
        raise RuntimeError(
            f"Internal AO mapping has {orca_to_pyscf.size} functions, but ORCA S-Matrix has {s_orca.shape[0]}."
        )
    s_reordered = s_pyscf[np.ix_(orca_to_pyscf, orca_to_pyscf)]
    orca_signs = _solve_orca_to_pyscf_signs(s_orca, s_reordered)
    s_mapped = (orca_signs[:, None] * s_reordered) * orca_signs[None, :]
    diff = s_mapped - s_orca
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    if max_abs > tolerance:
        raise RuntimeError(
            f"Mapped PySCF AO overlap does not reproduce ORCA S-Matrix: max |dS|={max_abs:.3e}, rms={rms:.3e}, "
            f"tolerance={tolerance:.3e}. Refusing cube generation because AO ordering, normalization, or "
            "solid-harmonic convention is not validated."
        )
    return _PyscfContext(mol, orca_to_pyscf, orca_signs, max_abs, rms)


def _shell_to_pyscf(shell: dict) -> list:
    label = str(shell.get("Shell") or shell.get("shell") or "").lower()[:1]
    if label not in SHELL_L:
        raise RuntimeError(f"Unsupported shell label in ORCA JSON basis: {shell.get('Shell')!r}")
    exponents = shell.get("Exponents") or shell.get("exponents")
    coefficients = shell.get("Coefficients") or shell.get("coefficients")
    if exponents is None or coefficients is None or len(exponents) != len(coefficients):
        raise RuntimeError("ORCA JSON basis shell has inconsistent Exponents/Coefficients arrays.")
    out = [SHELL_L[label]]
    for exp, coeff in zip(exponents, coefficients):
        out.append([float(exp), float(coeff)])
    return out


def _shell_sequence_orca_to_pyscf_indices(atoms: Sequence[dict]) -> np.ndarray:
    pyscf_components = {
        "s": [""],
        "p": ["x", "y", "z"],
        "d": ["xy", "yz", "z2", "xz", "x2y2"],
        "f": ["-3", "-2", "-1", "0", "+1", "+2", "+3"],
    }
    orca_components = {
        "s": [""],
        "p": ["z", "x", "y"],
        "d": ["z2", "xz", "yz", "x2y2", "xy"],
        "f": ["0", "+1", "-1", "+2", "-2", "+3", "-3"],
    }
    indices: List[int] = []
    pyscf_start = 0
    for atom in atoms:
        for shell in atom.get("Basis", []):
            label = str(shell.get("Shell") or shell.get("shell") or "").lower()[:1]
            if label not in pyscf_components or label not in orca_components:
                raise RuntimeError(
                    f"Self-derived NTO cube writer does not yet know the ORCA/PySCF spherical ordering for {label!r} shells."
                )
            pyscf_order = pyscf_components[label]
            orca_order = orca_components[label]
            block = {component: pyscf_start + i for i, component in enumerate(pyscf_order)}
            indices.extend(block[component] for component in orca_order)
            pyscf_start += len(pyscf_order)
    return np.asarray(indices, dtype=int)


def _solve_orca_to_pyscf_signs(s_orca: np.ndarray, s_reordered: np.ndarray) -> np.ndarray:
    n = int(s_orca.shape[0])
    signs = np.ones(n, dtype=float)
    fixed = np.zeros(n, dtype=bool)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a = float(s_orca[i, j])
            b = float(s_reordered[i, j])
            if abs(a) <= 1.0e-4 or abs(b) <= 1.0e-4:
                continue
            ratio = a / b
            if abs(abs(ratio) - 1.0) <= 0.1:
                edges.append((abs(a), i, j, 1.0 if ratio > 0.0 else -1.0))
    edges.sort(reverse=True, key=lambda row: row[0])
    remaining = set(range(n))
    while remaining:
        seed = min(remaining)
        fixed[seed] = True
        remaining.remove(seed)
        changed = True
        while changed:
            changed = False
            for _, i, j, sign_ij in edges:
                if fixed[i] and not fixed[j]:
                    signs[j] = signs[i] * sign_ij
                    fixed[j] = True
                    remaining.discard(j)
                    changed = True
                elif fixed[j] and not fixed[i]:
                    signs[i] = signs[j] * sign_ij
                    fixed[i] = True
                    remaining.discard(i)
                    changed = True
    return signs


def _render_pair(
    step: StepData,
    root: int,
    pair: NTOOrbitalPair,
    context: _PyscfContext,
    cube_dir: Path,
    image_dir: Path,
    script_dir: Path,
    log_dir: Path,
    grid: Tuple[int, int, int],
    margin_ang: float,
    render_png: bool,
    force: bool,
    args,
) -> List[NTOVisualizationRecord]:
    out: List[NTOVisualizationRecord] = []
    for side, coeff in (("donor", pair.donor_coeff), ("acceptor", pair.acceptor_coeff)):
        pair_index = int(pair.donor_index if side == "donor" else pair.acceptor_index)
        spin_tag = "a" if pair.spin == "alpha" else "b"
        stem = f"{step.job.label}_root{root:03d}_{pair.spin}_pair{pair_index:02d}_{side}"
        cube = cube_dir / f"{stem}.cube"
        png = image_dir / f"{stem}.png"
        status = "ok"
        message = "self-derived NTO cube generated from CIS/TDA amplitude SVD"
        try:
            coeff_arr = _orca_coefficients_to_pyscf(np.asarray(coeff, dtype=float).reshape(-1), context)
            if coeff_arr.size != context.mol.nao_nr():
                raise RuntimeError(
                    f"{side} coefficient length {coeff_arr.size} does not match validated AO count {context.mol.nao_nr()}."
                )
            if force or not cube.exists() or cube.stat().st_size == 0:
                _write_cube(
                    cube,
                    context.mol,
                    coeff_arr,
                    grid,
                    margin_ang,
                    f"{step.job.label} root {root} {pair.spin} pair {pair_index} {side} self-derived NTO",
                )
            else:
                message = "reused cached self-derived NTO cube"
            if render_png:
                png_status, png_message = _render_png_with_jmol(
                    step.job.geom_path,
                    cube,
                    png,
                    script_dir / f"jmol_{stem}.spt",
                    log_dir / f"jmol_{stem}.log",
                    args,
                    force,
                )
                if png_status != "ok":
                    status = "cube_ok_png_failed"
                    message = message + "; " + png_message
                else:
                    message = message + "; Jmol PNG rendered"
        except Exception as exc:
            status = "failed"
            message = str(exc)
        out.append(
            NTOVisualizationRecord(
                step_label=step.job.label,
                root=root,
                spin=spin_tag,
                pair_index=pair_index,
                side=side,
                weight=float(pair.weight),
                cube_file=cube,
                png_file=png if png.exists() else None,
                status=status,
                message=message,
                overlap_max_abs_error=context.overlap_max_abs_error,
                overlap_rms_error=context.overlap_rms_error,
            )
        )
    return out


def _orca_coefficients_to_pyscf(coefficients: np.ndarray, context: _PyscfContext) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
    if coeffs.size != context.orca_to_pyscf.size:
        raise RuntimeError(
            f"ORCA-order coefficient vector has length {coeffs.size}, but the validated AO mapping has "
            f"{context.orca_to_pyscf.size} functions."
        )
    out = np.zeros(context.mol.nao_nr(), dtype=float)
    out[context.orca_to_pyscf] = context.orca_signs * coeffs
    return out


def _write_cube(
    path: Path,
    mol,
    coeff: np.ndarray,
    grid: Tuple[int, int, int],
    margin_ang: float,
    comment: str,
) -> None:
    nx, ny, nz = grid
    if min(nx, ny, nz) < 2:
        raise RuntimeError("--self-nto-cube-grid values must all be >= 2.")
    coords = np.asarray(mol.atom_coords(), dtype=float)
    margin = float(margin_ang) * ANGSTROM_TO_BOHR
    origin = coords.min(axis=0) - margin
    upper = coords.max(axis=0) + margin
    steps = (upper - origin) / np.asarray([nx - 1, ny - 1, nz - 1], dtype=float)
    vectors = np.diag(steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(comment[:78] + "\n")
        f.write("Generated by orca_diabatize.py from self-derived CIS/TDA NTO coefficients\n")
        f.write(f"{mol.natm:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n")
        f.write(f"{nx:5d}{vectors[0,0]:12.6f}{vectors[0,1]:12.6f}{vectors[0,2]:12.6f}\n")
        f.write(f"{ny:5d}{vectors[1,0]:12.6f}{vectors[1,1]:12.6f}{vectors[1,2]:12.6f}\n")
        f.write(f"{nz:5d}{vectors[2,0]:12.6f}{vectors[2,1]:12.6f}{vectors[2,2]:12.6f}\n")
        for i in range(mol.natm):
            z = int(mol.atom_charge(i))
            x, y, zc = coords[i]
            f.write(f"{z:5d}{float(z):12.6f}{x:12.6f}{y:12.6f}{zc:12.6f}\n")
        total = nx * ny * nz
        vals_on_line = 0
        chunk = 20000
        for start in range(0, total, chunk):
            stop = min(total, start + chunk)
            idx = np.arange(start, stop, dtype=np.int64)
            iz = idx % nz
            iy = (idx // nz) % ny
            ix = idx // (ny * nz)
            points = origin + np.column_stack((ix * steps[0], iy * steps[1], iz * steps[2]))
            ao = mol.eval_gto("GTOval_sph", points)
            values = np.asarray(ao @ coeff, dtype=float)
            for val in values:
                f.write(f" {val:13.5e}")
                vals_on_line += 1
                if vals_on_line == 6:
                    f.write("\n")
                    vals_on_line = 0
        if vals_on_line:
            f.write("\n")


def _render_png_with_jmol(
    geom: Path,
    cube: Path,
    png: Path,
    script: Path,
    log: Path,
    args,
    force: bool,
) -> Tuple[str, str]:
    if png.exists() and png.stat().st_size > 0 and not force:
        return "ok", "reused cached Jmol PNG"
    exe = _resolve_executable(getattr(args, "jmol", "jmol"))
    if exe is None:
        return "failed", f"Jmol executable {getattr(args, 'jmol', 'jmol')!r} was not found."
    if not Path(geom).exists():
        return "failed", f"Geometry file for Jmol rendering is missing: {geom}"
    png.parent.mkdir(parents=True, exist_ok=True)
    _write_jmol_script(
        script,
        Path(geom),
        Path(cube),
        Path(png),
        float(getattr(args, "jmol_cutoff", 0.05)),
        float(getattr(args, "jmol_rotate_y", -90.0)),
        float(getattr(args, "jmol_rotate_x", -50.0)),
        int(getattr(args, "jmol_zoom", 150)),
        int(getattr(args, "jmol_width", 420)),
        int(getattr(args, "jmol_height", 315)),
    )
    commands = [[exe, "-n", "-s", str(script)], [exe, "-s", str(script)]]
    chunks = []
    timeout = int(getattr(args, "jmol_timeout", 120))
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except Exception as exc:
            chunks.append(f"$ {' '.join(cmd)}\nERROR: {exc}\n")
            continue
        chunks.append(
            f"$ {' '.join(cmd)}\nreturncode={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n"
        )
        if proc.returncode == 0 and png.exists() and png.stat().st_size > 0:
            log.write_text("\n".join(chunks))
            return "ok", "Jmol PNG rendered"
    log.write_text("\n".join(chunks))
    return "failed", f"Jmol did not produce a PNG; see {log}."


def _write_jmol_script(
    path: Path,
    geom: Path,
    cube: Path,
    png: Path,
    cutoff: float,
    rotate_y: float,
    rotate_x: float,
    zoom: int,
    width: int,
    height: int,
) -> None:
    lines = [
        "background white",
        "set showHydrogens off",
        "wireframe only",
        f"load {_quote(geom.resolve())}",
        f"isosurface sign red blue cutoff {cutoff:g} {_quote(cube.resolve())}",
        f"rotate y {rotate_y:g}",
        f"rotate x {rotate_x:g}",
        f"zoom {int(zoom)}",
        "refresh",
        "delay 1",
        f"write image {int(width)} {int(height)} png {_quote(png.resolve())}",
        "quit",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _select_pairs(pairs: Sequence[NTOOrbitalPair], weight_min: float, max_pairs: int) -> List[NTOOrbitalPair]:
    eligible = [p for p in pairs if float(p.weight) >= float(weight_min)]
    eligible.sort(key=lambda p: float(p.weight), reverse=True)
    if max_pairs > 0:
        eligible = eligible[:max_pairs]
    return eligible


def _selected_roots(text: Optional[str], roots: Sequence[int]) -> List[int]:
    if not text:
        return list(roots)
    allowed = set(roots)
    return [r for r in parse_int_range(text) if r in allowed]


def _selected_steps(text: Optional[str], steps: Sequence[StepData]) -> Set[str]:
    labels = {step.job.label for step in steps}
    if not text:
        return labels
    out: Set[str] = set()
    by_order = {step.job.order: step.job.label for step in steps}
    for token in _split_tokens(text):
        if token in labels:
            out.add(token)
            continue
        clean = token[1:] if token.startswith("p") and token[1:].isdigit() else token
        if clean.isdigit():
            label = by_order.get(int(clean))
            if label is not None:
                out.add(label)
            continue
        if "-" in clean:
            try:
                a, b = clean.split("-", 1)
                for i in range(int(a), int(b) + 1):
                    label = by_order.get(i)
                    if label is not None:
                        out.add(label)
            except Exception:
                pass
    return out


def _split_tokens(text: str) -> Iterable[str]:
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if token:
            yield token


def _element_symbol(label: str) -> str:
    return "".join(ch for ch in str(label) if ch.isalpha())


def _resolve_executable(value: str) -> Optional[str]:
    p = Path(value)
    if p.is_absolute() or "/" in value:
        return str(p) if p.exists() else None
    return shutil.which(value)


def _quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'

#!/usr/bin/env python3
"""Generate an audited report for the Ru-CO TDenTrack/pysisyphus run."""

from __future__ import annotations

import argparse
import csv
import gc
import html
import importlib.util
import json
import math
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from excited_state_diabatizer.cis_io import parse_cis_amplitudes
from excited_state_diabatizer.nto_from_tden import derive_self_nto_state
from excited_state_diabatizer.self_nto_cubes import (
    _build_pyscf_context,
    _orca_coefficients_to_pyscf,
)
from pysisyphus.wavefunction import Wavefunction


EH_TO_EV = 27.211386245988
EH_TO_KCAL = 627.5094740631
BOHR_TO_ANG = 0.529177210903

ENERGY_JOBS = (
    ("committed start", "opt_job/calculator_000.000.orca", 3),
    ("restart bridge 1", "opt_job_retry5/calculator_000.000.orca", 2),
    ("restart bridge 2", "opt_job_retry6/calculator_000.000.orca", 2),
    ("retry7 cycle 1", "opt_job_retry7/calculator_000.000.orca", 2),
    ("retry7 cycle 2", "opt_job_retry7/calculator_000.001.orca", 2),
    ("retry7 cycle 3", "opt_job_retry7/calculator_000.002.orca", 2),
    ("retry7 cycle 4", "opt_job_retry7/calculator_000.003.orca", 2),
    ("retry7 cycle 5", "opt_job_retry7/calculator_000.004.orca", 2),
    ("retry7 cycle 6", "opt_job_retry7/calculator_000.005.orca", 2),
    ("latest evaluated", "opt_job_retry7/calculator_000.006.orca", 2),
)

ACCEPTED_SURVEYS = (
    ("opt_job_retry5", 0, 1.0),
    ("opt_job_retry6", 0, 1.0),
    ("opt_job_retry7", 0, 1.0),
    ("opt_job_retry7", 1, 1.0),
    ("opt_job_retry7", 2, 0.5),
    ("opt_job_retry7", 3, 1.0),
    ("opt_job_retry7", 4, 0.5),
    ("opt_job_retry7", 5, 1.0),
    ("opt_job_retry7", 6, 0.5),
)

COLORS = {
    "H": "#e7e7e7",
    "C": "#363636",
    "N": "#2359c4",
    "O": "#d62828",
    "Ru": "#3db7a3",
}
RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "Ru": 1.46}


def final_energy(path: Path) -> float:
    matches = re.findall(
        r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)",
        path.read_text(errors="replace"),
    )
    if not matches:
        raise RuntimeError(f"No FINAL SINGLE POINT ENERGY in {path}")
    return float(matches[-1].replace("D", "E").replace("d", "e"))


def tden_overlap(left, right, s_ao: np.ndarray) -> np.ndarray:
    wf_l, header_l, states_l = left
    wf_r, header_r, states_r = right
    roots_l = sorted(states_l)
    roots_r = sorted(states_r)
    result = np.zeros((len(roots_l), len(roots_r)))
    for spin, key in enumerate(("alpha", "beta")):
        s_mo = wf_l.C[spin].T @ s_ao @ wf_r.C[spin]
        lo = slice(
            getattr(header_l, f"{key}_occ_start"),
            getattr(header_l, f"{key}_occ_end") + 1,
        )
        lv = slice(
            getattr(header_l, f"{key}_virt_start"),
            getattr(header_l, f"{key}_virt_end") + 1,
        )
        ro = slice(
            getattr(header_r, f"{key}_occ_start"),
            getattr(header_r, f"{key}_occ_end") + 1,
        )
        rv = slice(
            getattr(header_r, f"{key}_virt_start"),
            getattr(header_r, f"{key}_virt_end") + 1,
        )
        trans_l = np.stack([states_l[root][key] for root in roots_l])
        trans_r = np.stack([states_r[root][key] for root in roots_r])
        result += np.einsum(
            "ria,ij,ab,sjb->rs",
            trans_l,
            s_mo[lo, ro],
            s_mo[lv, rv],
            trans_r,
            optimize=True,
        )
    return result


def load_electronic_state(stem: Path):
    wf = Wavefunction.from_file(Path(f"{stem}.bson"))
    header, states = parse_cis_amplitudes(Path(f"{stem}.cis"), tda=True)
    return wf, header, states


def normalized_tden_matrix(left, right):
    s_cross = left[0].S_with(right[0])
    cross = tden_overlap(left, right, s_cross)
    gram_l = tden_overlap(left, left, left[0].S)
    gram_r = tden_overlap(right, right, right[0].S)
    norms = np.sqrt(np.diag(gram_l)[:, None] * np.diag(gram_r)[None, :])
    return cross, np.abs(cross) / norms, s_cross


def metrics_from_h5(base: Path):
    rows = {}

    def one(path: Path, index: int):
        with h5py.File(path) as handle:
            group = handle["opt"]
            return {
                "max_force": float(group["max_forces"][index]),
                "rms_force": float(group["rms_forces"][index]),
                "max_step": float(group["max_steps"][index]),
                "rms_step": float(group["rms_steps"][index]),
                "is_converged": bool(group.attrs["is_converged"]),
                "coord_type": str(group.attrs["coord_type"]),
                "coord_size": int(group.attrs["coord_size"]),
            }

    # Retry 5 contains the corrected committed-start record. Retry 6 contains
    # the first bridge result. Retry 7 starts at the second bridge result.
    rows[0] = one(base / "opt_job_retry5/optimization.h5", 0)
    rows[1] = one(base / "opt_job_retry6/optimization.h5", 0)
    for index in range(7):
        rows[index + 2] = one(base / "opt_job_retry7/optimization.h5", index)
    return rows


def accepted_manifest_rows(base: Path):
    rows = []
    for retry, revision, factor in ACCEPTED_SURVEYS:
        candidates = []
        for path in (base / retry / "tdentrack_surveys").glob(
            f"survey-r{revision:04d}-*/state_survey.json"
        ):
            payload = json.loads(path.read_text())
            if math.isclose(float(payload["factor"]), factor, abs_tol=1.0e-12):
                candidates.append((path, payload))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one accepted survey for {retry} revision {revision} "
                f"factor {factor}, found {len(candidates)}"
            )
        path, payload = candidates[0]
        reference_roots = [int(x) for x in payload["reference_roots"]]
        candidate_roots = [int(x) for x in payload["candidate_roots"]]
        reference_root = int(payload["reference_root"])
        matrix = np.asarray(payload["signed_overlap_matrix"], dtype=float)
        gram_ref = np.asarray(payload["reference_gram"], dtype=float)
        gram_cur = np.asarray(payload["candidate_gram"], dtype=float)
        ref_index = reference_roots.index(reference_root)
        scores = np.abs(matrix[ref_index]) / np.sqrt(
            gram_ref[ref_index, ref_index] * np.diag(gram_cur)
        )
        order = np.argsort(scores)[::-1]
        rows.append(
            {
                "path": path,
                "retry": retry,
                "revision": revision,
                "factor": factor,
                "reference_root": reference_root,
                "selected_root": candidate_roots[int(order[0])],
                "similarity": float(scores[order[0]]),
                "runner_up_root": candidate_roots[int(order[1])],
                "runner_up": float(scores[order[1]]),
                "margin": float(scores[order[0]] - scores[order[1]]),
            }
        )
    return rows


def dominant_nto(label: str, root: int, electronic_state):
    wf, _, states = electronic_state
    vec, weight_rows, reconstruction = derive_self_nto_state(
        label,
        0,
        root,
        states[root],
        {"alpha": wf.C[0], "beta": wf.C[1]},
        weight_min=0.0,
        cumulative=1.0,
    )
    pair = max(vec.pairs, key=lambda item: item.weight)
    state_weight = sum(row.weight for row in weight_rows)
    return pair, weight_rows, reconstruction, state_weight


def bson_loader():
    path = Path("/home/diegoa/dev/pysisyphus/pysisyphus/io/bson.py")
    spec = importlib.util.spec_from_file_location("_pysis_bson_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_orbital(context, coefficients, grid=(40, 40, 40), margin_ang=3.0):
    coeff = _orca_coefficients_to_pyscf(np.asarray(coefficients), context)
    mol = context.mol
    coords = np.asarray(mol.atom_coords(), dtype=float)
    margin = margin_ang / BOHR_TO_ANG
    origin = coords.min(axis=0) - margin
    upper = coords.max(axis=0) + margin
    grid = np.asarray(grid, dtype=int)
    steps = (upper - origin) / (grid - 1)
    total = int(np.prod(grid))
    values = np.empty(total)
    chunk = 20000
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        idx = np.arange(start, stop)
        iz = idx % grid[2]
        iy = (idx // grid[2]) % grid[1]
        ix = idx // (grid[1] * grid[2])
        points = origin + np.column_stack(
            (ix * steps[0], iy * steps[1], iz * steps[2])
        )
        values[start:stop] = mol.eval_gto("GTOval_sph", points) @ coeff
    return values.reshape(tuple(grid)), origin, steps


def write_cube(path: Path, mol, volume, origin, steps, comment):
    nx, ny, nz = volume.shape
    coords = np.asarray(mol.atom_coords())
    with path.open("w") as handle:
        handle.write(comment[:78] + "\n")
        handle.write("CIS/TDA NTO; PySCF AO grid validated against ORCA S matrix\n")
        handle.write(
            f"{mol.natm:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n"
        )
        for n, axis, step in zip((nx, ny, nz), range(3), steps):
            vector = [0.0, 0.0, 0.0]
            vector[axis] = float(step)
            handle.write(
                f"{n:5d}{vector[0]:12.6f}{vector[1]:12.6f}{vector[2]:12.6f}\n"
            )
        for i in range(mol.natm):
            charge = int(mol.atom_charge(i))
            x, y, z = coords[i]
            handle.write(
                f"{charge:5d}{float(charge):12.6f}{x:12.6f}{y:12.6f}{z:12.6f}\n"
            )
        flat = volume.ravel()
        for start in range(0, flat.size, 6):
            handle.write(" ".join(f"{value:13.5e}" for value in flat[start : start + 6]))
            handle.write("\n")


def atom_symbol(mol, index: int) -> str:
    symbol = re.sub(r"\d", "", str(mol.atom_symbol(index))).capitalize()
    return symbol


def draw_molecule(ax, mol):
    coords = np.asarray(mol.atom_coords())
    symbols = [atom_symbol(mol, i) for i in range(mol.natm)]
    for i in range(mol.natm):
        for j in range(i):
            cutoff_ang = 1.22 * (RADII.get(symbols[i], 0.75) + RADII.get(symbols[j], 0.75))
            if np.linalg.norm(coords[i] - coords[j]) * BOHR_TO_ANG <= cutoff_ang:
                xyz = np.vstack((coords[i], coords[j]))
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#747474", lw=1.2, alpha=0.8)
    for symbol in sorted(set(symbols), key=lambda value: value != "Ru"):
        inds = [i for i, value in enumerate(symbols) if value == symbol]
        xyz = coords[inds]
        size = 105 if symbol == "Ru" else (35 if symbol == "H" else 58)
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=size,
            c=COLORS.get(symbol, "#adadad"),
            edgecolors="#202020",
            linewidths=0.35,
            depthshade=True,
        )


def draw_orbital(ax, mol, volume, origin, steps, title, cutoff=0.05):
    level = min(float(cutoff), 0.35 * float(np.max(np.abs(volume))))
    for sign, color in ((1.0, "#1565c0"), (-1.0, "#d32f2f")):
        field = sign * volume
        if float(field.max()) <= level:
            continue
        vertices, faces, _, _ = marching_cubes(field, level=level, spacing=tuple(steps))
        vertices += origin
        mesh = Poly3DCollection(
            vertices[faces], alpha=0.58, facecolor=color, edgecolor="none"
        )
        ax.add_collection3d(mesh)
    draw_molecule(ax, mol)
    coords = np.asarray(mol.atom_coords())
    center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    radius = 0.58 * float(np.max(coords.max(axis=0) - coords.min(axis=0))) + 1.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-72)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=2)
    return level


def nto_assets(outdir: Path, endpoints, pairs, bson):
    image_info = []
    for label, stem, root, state, pair in endpoints:
        wf = state[0]
        raw = bson.loads(Path(f"{stem}.bson").read_bytes())
        context = _build_pyscf_context(raw, wf.S, 1.0e-6)
        volumes = {}
        cutoffs = {}
        for side, coeff in (
            ("donor", pair.donor_coeff),
            ("acceptor", pair.acceptor_coeff),
        ):
            volume, origin, steps = evaluate_orbital(context, coeff)
            cube = outdir / f"nto_{label}_{side}.cube"
            write_cube(
                cube,
                context.mol,
                volume,
                origin,
                steps,
                f"{label} root {root} dominant {pair.spin} NTO {side}",
            )
            volumes[side] = (volume, origin, steps)
            script = outdir / f"jmol_nto_{label}_{side}.spt"
            png = outdir / f"nto_{label}_{side}_jmol.png"
            script.write_text(
                f'load "{cube.resolve()}";\n'
                "background white;\n"
                "set antialiasDisplay true;\n"
                f'isosurface cutoff 0.05 sign [x1565c0] [xd32f2f] "{cube.resolve()}";\n'
                "wireframe 0.12; spacefill 18%;\n"
                f'write IMAGE 900 700 PNG "{png.resolve()}";\nquit;\n'
            )
        fig = plt.figure(figsize=(10.5, 5.2), constrained_layout=True)
        for panel, side in enumerate(("donor", "acceptor"), start=1):
            ax = fig.add_subplot(1, 2, panel, projection="3d")
            volume, origin, steps = volumes[side]
            cutoffs[side] = draw_orbital(
                ax,
                context.mol,
                volume,
                origin,
                steps,
                f"{side.capitalize()} NTO",
            )
        fig.suptitle(
            f"{label.capitalize()} state: root {root}, dominant {pair.spin} pair "
            f"(weight {pair.weight:.6f})",
            fontsize=13,
        )
        figure = outdir / f"nto_{label}_pair.png"
        fig.savefig(figure, dpi=180, facecolor="white")
        plt.close(fig)
        image_info.append(
            {
                "label": label,
                "root": root,
                "figure": figure,
                "weight": float(pair.weight),
                "spin": pair.spin,
                "overlap_max": context.overlap_max_abs_error,
                "overlap_rms": context.overlap_rms_error,
                "cutoffs": cutoffs,
            }
        )
        del raw, context, volumes, volume, origin, steps
        gc.collect()
    return image_info


def make_plots(outdir, energies, metric_rows, tracking, direct_scores):
    x = np.arange(len(energies))
    values = np.array([row["energy_eh"] for row in energies])

    fig, ax = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    ax.plot(x, values, marker="o", color="#0d5c75", lw=2)
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.set_xlabel("Accepted/evaluated optimizer geometry")
    ax.set_ylabel("Absolute energy / Eh")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.set_title("Tracked-state absolute energy")
    fig.savefig(outdir / "absolute_energy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    rel = (values - values[0]) * EH_TO_EV
    ax.plot(x, rel, marker="o", color="#6a4c93", lw=2)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Accepted/evaluated optimizer geometry")
    ax.set_ylabel("Energy relative to committed start / eV")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.set_title("Energy lowering along the stitched restart trajectory")
    fig.savefig(outdir / "relative_energy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    specs = (
        ("max_force", 0.0025, "Maximum force", "Eh a0$^{-1}$ (rad$^{-1}$)"),
        ("rms_force", 0.0017, "RMS force", "Eh a0$^{-1}$ (rad$^{-1}$)"),
        ("max_step", 0.0100, "Maximum step", "a0 (rad)"),
        ("rms_step", 0.0067, "RMS step", "a0 (rad)"),
    )
    mx = sorted(metric_rows)
    for ax, (key, threshold, title, ylabel) in zip(axes.ravel(), specs):
        vals = [metric_rows[index][key] for index in mx]
        ax.semilogy(mx, vals, marker="o", color="#c44536")
        ax.axhline(threshold, color="#1b7f3a", ls="--", label="threshold")
        ax.set_title(title)
        ax.set_xlabel("Geometry")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22, which="both")
        ax.legend(frameon=False)
    fig.suptitle("Optimizer convergence metrics (completed rows only)")
    fig.savefig(outdir / "convergence_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)
    tx = np.arange(1, len(tracking) + 1)
    best = [row["similarity"] for row in tracking]
    second = [row["runner_up"] for row in tracking]
    ax.plot(tx, best, "o-", lw=2, label="selected root")
    ax.plot(tx, second, "o--", lw=1.4, label="runner-up")
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Accepted electronic-state transition")
    ax.set_ylabel("Normalized transition-density overlap")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Adjacent-geometry state-tracking confidence")
    fig.savefig(outdir / "adjacent_state_similarity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    roots = np.arange(1, len(direct_scores) + 1)
    colors = ["#0d5c75" if root == int(np.argmax(direct_scores)) + 1 else "#9eb5bd" for root in roots]
    ax.bar(roots, direct_scores, color=colors)
    ax.set_xticks(roots)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Root at latest evaluated geometry")
    ax.set_ylabel("|normalized overlap| with committed-start root 3")
    ax.grid(alpha=0.2, axis="y")
    ax.set_title("Direct committed-start → latest electronic comparison")
    fig.savefig(outdir / "initial_to_latest_root_overlap.png", dpi=180)
    plt.close(fig)


def write_csvs(outdir, energies, tracking, metric_rows, direct_scores):
    with (outdir / "accepted_geometry_energies.csv").open("w", newline="") as handle:
        fields = [
            "geometry",
            "label",
            "root",
            "energy_eh",
            "relative_ev",
            "relative_kcal_mol",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in energies:
            writer.writerow({key: row[key] for key in fields})
    with (outdir / "convergence_metrics.csv").open("w", newline="") as handle:
        fields = ["geometry", "max_force", "rms_force", "max_step", "rms_step"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, values in metric_rows.items():
            writer.writerow({"geometry": index, **{key: values[key] for key in fields[1:]}})
    with (outdir / "accepted_state_transitions.csv").open("w", newline="") as handle:
        fields = [
            "transition",
            "reference_root",
            "selected_root",
            "factor",
            "similarity",
            "runner_up_root",
            "runner_up",
            "margin",
            "manifest",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(tracking, 1):
            writer.writerow(
                {
                    "transition": index,
                    **{key: row[key] for key in fields[1:-1]},
                    "manifest": row["path"],
                }
            )
    with (outdir / "initial_to_latest_root_overlap.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["latest_root", "normalized_overlap_with_initial_root3"])
        for root, score in enumerate(direct_scores, 1):
            writer.writerow([root, score])


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def write_html(
    outdir,
    base,
    energies,
    metric_rows,
    tracking,
    direct_scores,
    direct_signed,
    nto,
    nto_images,
    geom,
):
    initial_pair, final_pair, donor_overlap, acceptor_overlap = nto
    best_root = int(np.argmax(direct_scores)) + 1
    best = float(np.max(direct_scores))
    runner = float(np.partition(direct_scores, -2)[-2])
    total_drop = energies[-1]["energy_eh"] - energies[0]["energy_eh"]
    latest_complete = metric_rows[max(metric_rows)]
    energy_rows = []
    for row in energies:
        metrics = metric_rows.get(row["geometry"])
        energy_rows.append(
            (
                row["geometry"],
                html.escape(row["label"]),
                row["root"],
                f"{row['energy_eh']:.12f}",
                f"{row['relative_ev']:.6f}",
                "—" if metrics is None else f"{metrics['max_force']:.6f}",
                "—" if metrics is None else f"{metrics['rms_force']:.6f}",
                "—" if metrics is None else f"{metrics['max_step']:.6f}",
                "—" if metrics is None else f"{metrics['rms_step']:.6f}",
            )
        )
    tracking_rows = [
        (
            index,
            f"{row['reference_root']} → {row['selected_root']}",
            f"{row['factor']:.2f}",
            f"{row['similarity']:.7f}",
            f"{row['runner_up_root']} ({row['runner_up']:.7f})",
            f"{row['margin']:.7f}",
        )
        for index, row in enumerate(tracking, 1)
    ]
    image_blocks = []
    for info in nto_images:
        image_blocks.append(
            f"""
            <article class="nto-card">
              <h3>{info['label'].capitalize()} root {info['root']}</h3>
              <img src="{info['figure'].name}" alt="{info['label']} donor and acceptor NTO">
              <p>Dominant {info['spin']}-spin pair weight: <strong>{info['weight']:.6f}</strong>.
              ORCA/PySCF AO-overlap validation: max |ΔS| =
              {info['overlap_max']:.2e}, RMS = {info['overlap_rms']:.2e}.
              Rendered isovalue: ±{info['cutoffs']['donor']:.3f} (donor) and
              ±{info['cutoffs']['acceptor']:.3f} (acceptor).</p>
            </article>
            """
        )

    css = """
    :root { --ink:#17242b; --muted:#5a6a72; --blue:#0d5c75; --pale:#eef4f5;
            --green:#1b7f3a; --amber:#a65f00; --red:#a12821; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
           background:#f7f8f7; }
    main { max-width:1180px; margin:auto; padding:38px 28px 70px; }
    h1 { font-size:2.25rem; line-height:1.1; margin:0 0 8px; }
    h2 { margin-top:44px; border-bottom:2px solid #d7e1e3; padding-bottom:7px; }
    h3 { margin:8px 0 10px; }
    .subtitle { color:var(--muted); margin-bottom:28px; }
    .verdict { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }
    .card,.nto-card { background:white; border:1px solid #dbe3e5; border-radius:10px;
                      padding:18px; box-shadow:0 2px 10px #24343b10; }
    .card strong { display:block; font-size:1.35rem; margin-top:5px; }
    .fail strong { color:var(--red); } .pass strong { color:var(--green); }
    .warn { border-left:5px solid var(--amber); background:#fff9ed; padding:15px 18px; }
    .good { border-left:5px solid var(--green); background:#f0f8f2; padding:15px 18px; }
    figure { margin:26px 0; background:white; border:1px solid #dbe3e5; border-radius:10px; padding:12px; }
    figure img { width:100%; height:auto; display:block; }
    figcaption { color:var(--muted); padding:8px 6px 2px; }
    .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:18px; }
    .nto-card img { width:100%; }
    .table-wrap { overflow:auto; border:1px solid #dbe3e5; border-radius:8px; background:white; }
    table { border-collapse:collapse; width:100%; font-size:.9rem; }
    th,td { padding:9px 10px; border-bottom:1px solid #e3e8e9; text-align:right; white-space:nowrap; }
    th { background:var(--pale); position:sticky; top:0; }
    th:nth-child(2),td:nth-child(2) { text-align:left; }
    code { background:#edf1f2; padding:2px 5px; border-radius:4px; }
    a { color:var(--blue); }
    .small { font-size:.9rem; color:var(--muted); }
    """
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TDenTrack/pysisyphus excited-state optimization audit</title><style>{css}</style></head>
<body><main>
<h1>Excited-state optimization audit</h1>
<p class="subtitle">Ru–CO complex · ORCA/TDA state following · generated from retained BSON, CIS,
EnGrad, audit-manifest and optimizer HDF5 artifacts</p>

<section class="verdict">
  <div class="card fail">Geometry status<strong>Not converged</strong>HDF5 flag is false; the latest
  completed row still fails maximum-force and maximum-step criteria.</div>
  <div class="card pass">Electronic identity<strong>Supported</strong>Direct initial-root-3 → latest-root-2
  transition-density similarity = {best:.4f}, margin = {best-runner:.4f}.</div>
  <div class="card">Absolute energy change<strong>{total_drop:.6f} Eh</strong>
  {total_drop*EH_TO_EV:.4f} eV; {total_drop*EH_TO_KCAL:.2f} kcal mol⁻¹.</div>
  <div class="card">Latest completed metrics<strong>max |F| = {latest_complete['max_force']:.6f}</strong>
  max step = {latest_complete['max_step']:.6f}; thresholds are 0.002500 and 0.010000.</div>
</section>

<h2>1. Scope and trajectory reconstruction</h2>
<p>The energy curve contains the ten committed or subsequently evaluated gradient geometries that
form the stitched restart trajectory: the committed start in <code>opt_job</code>, bridge gradients in
<code>opt_job_retry5</code> and <code>opt_job_retry6</code>, and seven gradients in
<code>opt_job_retry7</code>. Alternative all-root survey displacements are state-selection trials,
not optimizer geometries, and are therefore not mixed into this energy curve.</p>
<div class="warn"><strong>Important provenance correction.</strong> The top-level
<code>initial_job/p000_000.000</code> is not the electronic reference retained by the successful
state-following chain. Its old BSON coordinate metadata are inconsistent with the actual optimizer
frame, and its printed final energy is also on the pre-fix path. The first accepted manifest names
<code>opt_job/calculator_000.000</code> as its reference. That committed root-3 calculation is used
as “initial” throughout this report.</div>

<figure><img src="absolute_energy.png" alt="Absolute energy plot">
<figcaption>Absolute ORCA FINAL SINGLE POINT ENERGY for every retained gradient evaluation on the
accepted trajectory. All production gradients after the committed start target root 2.</figcaption></figure>
<figure><img src="relative_energy.png" alt="Relative energy plot">
<figcaption>Same energies relative to the committed start. This auxiliary view exposes changes that
are visually compressed on an absolute Hartree scale.</figcaption></figure>
{table(("Geom.","Label","Root","Energy / Eh","ΔE / eV","max |F|","RMS F","max step","RMS step"), energy_rows)}

<h2>2. Geometry convergence</h2>
<p class="warn"><strong>The geometry is not converged.</strong> The latest serialized HDF5 row is
geometry 8 (retry7 cycle 6), where RMS force and RMS step pass, but maximum force
({latest_complete['max_force']:.6f}) exceeds 0.002500 and maximum step
({latest_complete['max_step']:.6f}) exceeds 0.010000. Geometry 9 has a normally terminated ORCA
energy/gradient ({energies[-1]['energy_eh']:.12f} Eh), and the optimizer log reports RMS force
approximately 0.000836, but the download ends while the next RFO/state-screened step is being built.
No completed row supplies its maximum force or step tests, and no “Optimization finished” marker is
present. It cannot be certified as converged.</p>
<figure><img src="convergence_metrics.png" alt="Convergence metrics">
<figcaption>Completed pysisyphus rows only. Green dashed lines are the configured thresholds.
Convergence requires all relevant tests, not merely RMS force.</figcaption></figure>
<p>The committed-start to latest displacement is {geom['rms_ang']:.3f} Å RMS per atom
({geom['max_ang']:.3f} Å maximum). The Ru–C distance changes from {geom['ru_c_initial']:.4f} to
{geom['ru_c_latest']:.4f} Å; C–O changes from {geom['c_o_initial']:.4f} to
{geom['c_o_latest']:.4f} Å. These are endpoint changes, not convergence tests.</p>

<h2>3. State-following audit</h2>
<p>Every accepted transition selects the same evolving state, although ORCA’s adiabatic root label
changes from 3 at the committed start to 2 at the first displacement. Adjacent normalized
transition-density similarities range from {min(row['similarity'] for row in tracking):.4f} to
{max(row['similarity'] for row in tracking):.4f}. The accepted displacement factor was reduced to
0.5 at three transitions; these are the transactional step-controller decisions, not an
always-on interpolation of the energy curve.</p>
<figure><img src="adjacent_state_similarity.png" alt="Adjacent state similarities">
<figcaption>Production overlap criterion: full signed α+β CIS/TDA transition-density overlap,
contracted with exact analytic inter-geometry AO overlaps reconstructed from the retained BSON
basis data.</figcaption></figure>
{table(("Transition","Root mapping","Factor","Selected similarity","Runner-up","Margin"), tracking_rows)}

<h2>4. Direct initial-to-latest comparison</h2>
<div class="good">The direct, non-chain comparison identifies latest root {best_root} as the
continuation of committed-start root 3 with normalized overlap <strong>{best:.7f}</strong>. The
runner-up is {runner:.7f}, giving a margin of {best-runner:.7f}. This is lower than the adjacent
overlaps—as expected after a sizable geometry/electronic relaxation—but it is unique and remains
large enough to support state continuity.</div>
<figure><img src="initial_to_latest_root_overlap.png" alt="Initial to latest root overlap">
<figcaption>Direct endpoint comparison against all 15 latest CIS roots; it does not multiply
adjacent scores and therefore avoids accumulated-score interpretation.</figcaption></figure>

<h2>5. Self-derived NTO analysis</h2>
<p>NTOs were reconstructed by SVD of the same validated <code>.cis</code> transition-amplitude
matrices used for state selection. The committed-start root-3 dominant β pair carries
{initial_pair.weight:.6f} of the transition norm; the latest root-2 pair carries
{final_pair.weight:.6f}. Using the exact endpoint AO cross-overlap, the dominant donor-orbital
overlap is {donor_overlap:.6f}, the acceptor-orbital overlap is {acceptor_overlap:.6f}, and their
product is {donor_overlap*acceptor_overlap:.6f}. The particularly high acceptor overlap and a still
large donor overlap are consistent with the direct full transition-density result.</p>
<div class="grid2">{''.join(image_blocks)}</div>
<p class="small">Blue/red surfaces are opposite orbital phases at ±0.05 (or the automatically
reduced cutoff recorded above). Jmol was not installed on this host and the attempted full
distribution download timed out. The report therefore uses a validated PySCF AO grid plus marching
cubes for the PNGs. Four Gaussian cube files and matching <code>jmol_*.spt</code> scripts are
included, so the identical orbitals can be rendered with Jmol later. This is a visualization-only
fallback; numerical overlap and NTO conclusions do not depend on the renderer.</p>

<h2>6. Verdict and recommended continuation</h2>
<ul>
  <li><strong>Electronic tracking:</strong> successful through the retained path. Adjacent overlap,
  direct endpoint transition density, and dominant NTOs agree that latest root 2 is the relaxed
  continuation of committed-start root 3.</li>
  <li><strong>Optimization:</strong> unfinished, not failed electronically and not converged
  geometrically. Resume from the latest committed/restartable state if the corresponding remote
  restart snapshot exists; otherwise restart from the latest complete serialized geometry 8.</li>
  <li><strong>Final validation:</strong> after convergence, run one all-root survey and repeat the
  direct endpoint/NTO audit. A converged label should only be applied once pysisyphus records all
  force/step criteria and <code>is_converged=True</code>.</li>
</ul>

<h2>7. Machine-readable outputs</h2>
<ul>
  <li><a href="accepted_geometry_energies.csv">accepted_geometry_energies.csv</a></li>
  <li><a href="convergence_metrics.csv">convergence_metrics.csv</a></li>
  <li><a href="accepted_state_transitions.csv">accepted_state_transitions.csv</a></li>
  <li><a href="initial_to_latest_root_overlap.csv">initial_to_latest_root_overlap.csv</a></li>
  <li><a href="analysis_summary.json">analysis_summary.json</a></li>
</ul>
<p class="small">Input directory: {html.escape(str(base))}</p>
</main></body></html>"""
    (outdir / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = args.input.resolve()
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    energies = []
    for index, (label, rel_stem, root) in enumerate(ENERGY_JOBS):
        stem = base / rel_stem
        energy = final_energy(Path(f"{stem}.out"))
        energies.append(
            {
                "geometry": index,
                "label": label,
                "root": root,
                "energy_eh": energy,
                "source": str(Path(f"{stem}.out")),
            }
        )
    initial_energy = energies[0]["energy_eh"]
    for row in energies:
        row["relative_ev"] = (row["energy_eh"] - initial_energy) * EH_TO_EV
        row["relative_kcal_mol"] = (
            row["energy_eh"] - initial_energy
        ) * EH_TO_KCAL

    metric_rows = metrics_from_h5(base)
    tracking = accepted_manifest_rows(base)

    initial_stem = base / ENERGY_JOBS[0][1]
    final_stem = base / ENERGY_JOBS[-1][1]
    initial_state = load_electronic_state(initial_stem)
    final_state = load_electronic_state(final_stem)
    signed, normalized, s_cross = normalized_tden_matrix(initial_state, final_state)
    direct_scores = normalized[2]

    initial_pair, initial_weights, initial_recon, initial_norm = dominant_nto(
        "initial", 3, initial_state
    )
    final_pair, final_weights, final_recon, final_norm = dominant_nto(
        "latest", 2, final_state
    )
    donor_overlap = abs(initial_pair.donor_coeff @ s_cross @ final_pair.donor_coeff)
    acceptor_overlap = abs(
        initial_pair.acceptor_coeff @ s_cross @ final_pair.acceptor_coeff
    )

    wf_i = initial_state[0]
    wf_f = final_state[0]
    coords_i = wf_i.coords3d * BOHR_TO_ANG
    coords_f = wf_f.coords3d * BOHR_TO_ANG
    displacement = coords_f - coords_i
    geom = {
        "rms_ang": float(np.sqrt(np.mean(np.sum(displacement * displacement, axis=1)))),
        "max_ang": float(np.max(np.linalg.norm(displacement, axis=1))),
        "ru_c_initial": float(np.linalg.norm(coords_i[1] - coords_i[0])),
        "ru_c_latest": float(np.linalg.norm(coords_f[1] - coords_f[0])),
        "c_o_initial": float(np.linalg.norm(coords_i[2] - coords_i[1])),
        "c_o_latest": float(np.linalg.norm(coords_f[2] - coords_f[1])),
    }

    make_plots(outdir, energies, metric_rows, tracking, direct_scores)
    write_csvs(outdir, energies, tracking, metric_rows, direct_scores)
    bson = bson_loader()
    nto_images = nto_assets(
        outdir,
        (
            ("initial", initial_stem, 3, initial_state, initial_pair),
            ("latest", final_stem, 2, final_state, final_pair),
        ),
        (initial_pair, final_pair),
        bson,
    )

    summary = {
        "input_directory": str(base),
        "geometry_converged": False,
        "serialized_is_converged": False,
        "completed_geometry_rows": 9,
        "gradient_evaluations": len(energies),
        "committed_start": str(initial_stem),
        "latest_evaluated": str(final_stem),
        "initial_root": 3,
        "latest_root": 2,
        "initial_energy_eh": energies[0]["energy_eh"],
        "latest_energy_eh": energies[-1]["energy_eh"],
        "energy_change_eh": energies[-1]["energy_eh"] - energies[0]["energy_eh"],
        "energy_change_ev": energies[-1]["relative_ev"],
        "direct_initial_latest_similarity": float(direct_scores[1]),
        "direct_best_latest_root": int(np.argmax(direct_scores)) + 1,
        "direct_assignment_margin": float(
            np.sort(direct_scores)[-1] - np.sort(direct_scores)[-2]
        ),
        "dominant_nto": {
            "spin": initial_pair.spin,
            "initial_weight": initial_pair.weight,
            "latest_weight": final_pair.weight,
            "donor_overlap": float(donor_overlap),
            "acceptor_overlap": float(acceptor_overlap),
            "overlap_product": float(donor_overlap * acceptor_overlap),
            "initial_state_norm": float(initial_norm),
            "latest_state_norm": float(final_norm),
        },
        "geometry": geom,
        "latest_completed_metrics": metric_rows[max(metric_rows)],
        "thresholds": {
            "max_force": 0.0025,
            "rms_force": 0.0017,
            "max_step": 0.0100,
            "rms_step": 0.0067,
        },
        "nto_images": [
            {
                key: (
                    value.name
                    if key == "figure" and isinstance(value, Path)
                    else (str(value) if isinstance(value, Path) else value)
                )
                for key, value in info.items()
            }
            for info in nto_images
        ],
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_html(
        outdir,
        base,
        energies,
        metric_rows,
        tracking,
        direct_scores,
        signed,
        (initial_pair, final_pair, donor_overlap, acceptor_overlap),
        nto_images,
        geom,
    )
    print(outdir / "index.html")


if __name__ == "__main__":
    main()

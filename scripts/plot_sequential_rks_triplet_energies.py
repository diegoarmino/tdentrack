#!/usr/bin/env python3
"""Plot absolute RKS-reference and triplet TDDFT scan energies.

ORCA's ``FINAL SINGLE POINT ENERGY`` belongs to the selected triplet root and
contains state-independent contributions that are absent from the ``Total
Energy`` printed in the SCF section.  For a calculation selecting local
triplet root m, a consistent corrected reference is

    E(S0) = E(FINAL) - omega(T_m)

and every triplet surface is reconstructed as

    E(T_n) = E(S0) + omega(T_n).
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCF_RE = re.compile(
    r"^\s*Total Energy\s*:\s*([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+Eh",
    re.MULTILINE,
)
FINAL_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)"
)
STATE_RE = re.compile(
    r"^\s*STATE\s+(\d+):\s+E=\s*"
    r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+au"
)
IROOT_RE = re.compile(r"^\s*iroot\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Point:
    label: str
    coordinate: float
    scf_energy: float
    final_energy: float
    selected_local_root: int
    correction: float
    s0_energy: float
    triplet_excitations: tuple[float, ...]

    @property
    def triplet_energies(self) -> np.ndarray:
        return self.s0_energy + np.asarray(self.triplet_excitations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--first-roots", type=int, default=10)
    return parser.parse_args()


def load_manifest(scan_dir: Path) -> dict[str, dict[str, str]]:
    path = scan_dir / "sequential_tddft_manifest.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["label"]: row for row in rows}


def parse_output(jobdir: Path, coordinate: float) -> Point:
    out_path = jobdir / "job.out"
    inp_path = jobdir / "job.inp"
    text = out_path.read_text(errors="replace")
    if "ORCA TERMINATED NORMALLY" not in text:
        raise RuntimeError("ORCA did not terminate normally")

    scf_matches = [float(match.group(1)) for match in SCF_RE.finditer(text)]
    final_matches = [float(match.group(1)) for match in FINAL_RE.finditer(text)]
    if not scf_matches or not final_matches:
        raise RuntimeError("missing SCF or FINAL energy")

    triplet_excitations: list[float] = []
    in_triplets = False
    for line in text.splitlines():
        if "TD-DFT/TDA EXCITED STATES (TRIPLETS)" in line:
            in_triplets = True
            continue
        if in_triplets and (
            "ABSORPTION SPECTRUM" in line
            or "CD SPECTRUM" in line
            or "TD-DFT/TDA EXCITED STATES (SINGLETS)" in line
        ):
            in_triplets = False
        if in_triplets and (match := STATE_RE.match(line)):
            triplet_excitations.append(float(match.group(2)))

    if not triplet_excitations:
        raise RuntimeError("no triplet TDDFT roots found")

    input_text = inp_path.read_text(errors="replace")
    iroot_match = IROOT_RE.search(input_text)
    selected_local_root = int(iroot_match.group(1)) if iroot_match else 1
    if not 1 <= selected_local_root <= len(triplet_excitations):
        raise RuntimeError(
            f"selected local root {selected_local_root} is outside the "
            f"{len(triplet_excitations)} parsed triplet roots"
        )

    scf_energy = scf_matches[-1]
    final_energy = final_matches[-1]
    selected_excitation = triplet_excitations[selected_local_root - 1]
    correction = final_energy - (scf_energy + selected_excitation)
    s0_energy = scf_energy + correction
    return Point(
        label=jobdir.name.removeprefix("tddft_step_"),
        coordinate=coordinate,
        scf_energy=scf_energy,
        final_energy=final_energy,
        selected_local_root=selected_local_root,
        correction=correction,
        s0_energy=s0_energy,
        triplet_excitations=tuple(triplet_excitations),
    )


def collect_points(scan_dir: Path):
    manifest = load_manifest(scan_dir)
    points: list[Point] = []
    excluded: list[tuple[str, str]] = []
    for jobdir in sorted(
        scan_dir.glob("tddft_step_*"),
        key=lambda path: float(path.name.removeprefix("tddft_step_")),
    ):
        label = jobdir.name.removeprefix("tddft_step_")
        row = manifest.get(label, {})
        coordinate = float(row.get("coordinate_A", label))
        try:
            points.append(parse_output(jobdir, coordinate))
        except Exception as exc:
            excluded.append((label, str(exc)))

    if not points:
        raise RuntimeError("No complete TDDFT points were parsed.")
    root_counts = {len(point.triplet_excitations) for point in points}
    if len(root_counts) != 1:
        raise RuntimeError(f"Inconsistent triplet-root counts: {root_counts}")
    return points, excluded


def save_csv(points: list[Point], output_dir: Path) -> None:
    root_count = len(points[0].triplet_excitations)
    fields = [
        "label",
        "coordinate_A",
        "raw_rks_scf_Eh",
        "orca_final_selected_triplet_Eh",
        "selected_local_triplet_root",
        "state_independent_correction_Eh",
        "S0_absolute_Eh",
    ] + [f"T{root}_absolute_Eh" for root in range(1, root_count + 1)]
    with (output_dir / "absolute_energies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = {
                "label": point.label,
                "coordinate_A": f"{point.coordinate:.12f}",
                "raw_rks_scf_Eh": f"{point.scf_energy:.12f}",
                "orca_final_selected_triplet_Eh": f"{point.final_energy:.12f}",
                "selected_local_triplet_root": point.selected_local_root,
                "state_independent_correction_Eh": f"{point.correction:.12f}",
                "S0_absolute_Eh": f"{point.s0_energy:.12f}",
            }
            row.update(
                {
                    f"T{root}_absolute_Eh": f"{energy:.12f}"
                    for root, energy in enumerate(point.triplet_energies, 1)
                }
            )
            writer.writerow(row)


def plot_states(
    points: list[Point],
    output_dir: Path,
    roots_to_plot: int,
    stem: str,
    title: str,
) -> None:
    coordinates = np.asarray([point.coordinate for point in points])
    s0 = np.asarray([point.s0_energy for point in points])
    triplets = np.vstack([point.triplet_energies for point in points])
    roots_to_plot = min(roots_to_plot, triplets.shape[1])

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.plot(
        coordinates,
        s0,
        color="black",
        marker="o",
        markersize=3.5,
        linewidth=2.5,
        label=r"$S_0$ (RKS)",
        zorder=10,
    )
    cmap = plt.get_cmap("turbo")
    denominator = max(roots_to_plot - 1, 1)
    for root in range(roots_to_plot):
        ax.plot(
            coordinates,
            triplets[:, root],
            color=cmap(root / denominator),
            marker="o",
            markersize=2.3,
            linewidth=1.15,
            label=rf"$T_{{{root + 1}}}$",
        )

    ax.set_xlabel(r"Ru–CO scan coordinate / $\mathrm{\AA}$")
    ax.set_ylabel(r"Absolute energy / $E_\mathrm{h}$")
    ax.set_title(title)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.22)
    columns = 2 if roots_to_plot <= 12 else 3
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        ncol=columns,
        fontsize=8.5,
        columnspacing=0.8,
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=210)
    plt.close(fig)


def max_jumps(points: list[Point]):
    coords = np.asarray([point.coordinate for point in points])
    energies = np.column_stack(
        (
            np.asarray([point.s0_energy for point in points]),
            np.vstack([point.triplet_energies for point in points]),
        )
    )
    jumps = np.abs(np.diff(energies, axis=0))
    rows = []
    for state_index in range(energies.shape[1]):
        jump_index = int(np.argmax(jumps[:, state_index]))
        rows.append(
            (
                "S0" if state_index == 0 else f"T{state_index}",
                float(jumps[jump_index, state_index]),
                float(coords[jump_index]),
                float(coords[jump_index + 1]),
            )
        )
    return rows


def save_html(
    points: list[Point],
    excluded: list[tuple[str, str]],
    output_dir: Path,
    first_roots: int,
) -> None:
    jumps = max_jumps(points)
    correction_values = np.asarray([point.correction for point in points])
    jump_rows = "\n".join(
        "<tr>"
        f"<td>{state}</td><td>{jump:.8f}</td>"
        f"<td>{left:.6f} → {right:.6f}</td>"
        "</tr>"
        for state, jump, left, right in jumps
    )
    excluded_rows = (
        "\n".join(
            f"<li>{html.escape(label)}: {html.escape(reason)}</li>"
            for label, reason in excluded
        )
        or "<li>None</li>"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Partial sequential RKS-triplet scan</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1200px;
       padding: 0 1rem; color: #18212b; }}
h1, h2 {{ color: #153752; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr));
          gap: .8rem; }}
.card {{ background: #eef4f7; border-radius: 8px; padding: .8rem 1rem; }}
.card strong {{ display: block; font-size: 1.25rem; margin-top: .2rem; }}
figure {{ margin: 1.5rem 0; }}
img {{ width: 100%; height: auto; border: 1px solid #d6dde2; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: .35rem .55rem; border-bottom: 1px solid #d6dde2; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
code {{ background: #eef1f3; padding: .1rem .25rem; }}
</style>
</head>
<body>
<h1>Partial sequential RKS → triplet TDDFT scan</h1>
<div class="cards">
  <div class="card">Completed points<strong>{len(points)}</strong></div>
  <div class="card">Coordinate range<strong>{points[0].coordinate:.6f}–{points[-1].coordinate:.6f} Å</strong></div>
  <div class="card">Triplet roots per point<strong>{len(points[0].triplet_excitations)}</strong></div>
  <div class="card">Correction range<strong>{correction_values.min():.6f}–{correction_values.max():.6f} Eh</strong></div>
</div>
<p>Absolute energies use <code>E(S0) = E(FINAL) − ω(T_selected)</code> and
<code>E(Tn) = E(S0) + ω(Tn)</code>. Curves are energy-ordered adiabatic ORCA
roots; no transition-density state tracking has yet been applied.</p>
<h2>Readable view: S0 and the first {first_roots} triplet roots</h2>
<figure><img src="absolute_energy_S0_T1-T{first_roots}.svg"
alt="Absolute energy plot for S0 and the first triplet roots"></figure>
<h2>All triplet roots</h2>
<figure><img src="absolute_energy_S0_all_triplets.svg"
alt="Absolute energy plot for S0 and all triplet roots"></figure>
<h2>Largest adjacent-point absolute-energy changes</h2>
<table><thead><tr><th>State</th><th>|ΔE|max / Eh</th><th>Interval / Å</th></tr></thead>
<tbody>{jump_rows}</tbody></table>
<h2>Excluded jobs</h2><ul>{excluded_rows}</ul>
<p>Machine-readable values: <a href="absolute_energies.csv">absolute_energies.csv</a>.</p>
</body></html>
"""
    (output_dir / "partial_energy_report.html").write_text(document)


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else scan_dir / "partial_absolute_energy_report"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    points, excluded = collect_points(scan_dir)
    save_csv(points, output_dir)
    first_roots = min(args.first_roots, len(points[0].triplet_excitations))
    plot_states(
        points,
        output_dir,
        first_roots,
        f"absolute_energy_S0_T1-T{first_roots}",
        f"Partial sequential scan: $S_0$ and $T_1$–$T_{{{first_roots}}}$",
    )
    plot_states(
        points,
        output_dir,
        len(points[0].triplet_excitations),
        "absolute_energy_S0_all_triplets",
        "Partial sequential scan: $S_0$ and all triplet TDDFT roots",
    )
    save_html(points, excluded, output_dir, first_roots)
    print(f"Parsed {len(points)} complete points.")
    print(f"Excluded {len(excluded)} incomplete/invalid jobs: {excluded}")
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()

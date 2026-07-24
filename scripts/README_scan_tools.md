# Sequential ORCA TDDFT rerun

`rerun_orca_tddft_sequential.py` reuses existing scan geometries without
rerunning the ground-state optimizations. It measures the collective coordinate
from every XYZ file and sorts from smaller to larger coordinate. The first
TDDFT point reads the corresponding ground-state `job.gbw`. Later points read
the preceding TDDFT `job.gbw` as the converged SCF-orbital guess with:

```text
%scf
  Guess MORead
  MOInp "previous.gbw"
  GuessMode CMatrix
  AutoStart false
end
```

For every point after the first, the same preceding TDDFT GBW also restarts the
response amplitudes:

```text
%tddft
  ...
  Restart "previous.gbw"
end
```

The first point does not request a TDDFT restart because a ground-state GBW has
no preceding response amplitudes. The source calculation must have the same
number of roots as the target or more; the sequential runner uses one unchanged
template for every point.

Validate the intended 57-point run without writing anything:

```bash
python3 /path/to/tdentrack/scripts/rerun_orca_tddft_sequential.py \
  --source-tddft-dir gs_scan_then_tddft/tddft \
  --source-gs-dir gs_scan_then_tddft/gs_scan \
  --jobs-csv jobs_ruco_intended_57.csv \
  --bonds-file collective_bonds.dat \
  --template /path/to/tdentrack/scripts/tddft_rks_triplets_sequential.inp \
  --charge 2 \
  --orca /absolute/path/to/orca \
  --output-dir gs_scan_then_tddft/tddft_rks_triplets_sequential \
  --dry-run
```

Remove only `--dry-run` to execute. Use `--resume` after a scheduler interruption.
Resume accepts a point only when `job.out` contains normal ORCA termination and
`job.gbw` is nonempty. The new output tree is separate from the old UKS results.

The charge is intentionally mandatory. The archived Ru–CO inputs inspected
during development contain charge `2`, although the later command record says
charge `1`; resolve that chemical/provenance discrepancy before production.
The runner validates the charge and singlet multiplicity recorded in the first
GS `job.inp`, and rejects an incompatible seed. The example therefore uses
charge `2` for the archived dataset. A charge-1 rerun requires a corresponding
charge-1 ground-state GBW.

The archived GS input also explicitly says `UKS` despite multiplicity 1. The
new calculation is explicitly `RKS`; the archived orbitals are used only as its
initial guess, after which the restricted SCF is converged. Inspect the first
point carefully before committing the full sequence.

# Legacy directory renaming

`rename_scan_steps_to_coordinate.py` renames only legacy `pNNN`/`mNNN`
directories. It derives labels from the geometry and collective-coordinate
definition rather than reconstructing them from a step counter.

Preview:

```bash
python3 /path/to/tdentrack/scripts/rename_scan_steps_to_coordinate.py \
  --workdir gs_scan_then_tddft \
  --bonds-file collective_bonds.dat
```

Apply only after inspecting the complete preview:

```bash
python3 /path/to/tdentrack/scripts/rename_scan_steps_to_coordinate.py \
  --workdir gs_scan_then_tddft \
  --bonds-file collective_bonds.dat \
  --apply
```

The command refuses existing destinations, rounded-label collisions, and
disagreement between paired GS/TDDFT geometries. It writes
`scan_directory_rename_manifest.csv` when changes are applied. It does not edit
the contents of completed ORCA output files.

# Continue the relaxed GS scan and sequential RKS-triplet TDDFT chain

`orca_gs_scan_then_tddft_sequential.py` extends a completed endpoint without
rerunning the existing scan. It propagates two independent restart chains:

- each constrained GS optimization reads the preceding GS `job.gbw`;
- each RKS-reference triplet TDDFT calculation reads the preceding TDDFT
  `job.gbw` for both its SCF orbitals and response-amplitude restart.

The driver validates the two seed geometries, charge/multiplicity, normal
termination, GS optimization convergence, triplet-root count, and nonempty
GBW/CIS artifacts. Every new job records SHA-256 provenance for its geometry,
template, and upstream GBW. With `--resume`, an incompatible job is archived
and recomputed; a changed upstream GBW therefore invalidates downstream reuse.

For the present Ru–CO continuation, the validated seed is 2.055247562115 Å.
A 0.01 Å grid followed by an exact 2.8000 Å endpoint contains 75 new points.
Validate the plan without writing anything:

```bash
python3 /path/to/tdentrack/scripts/orca_gs_scan_then_tddft_sequential.py \
  --seed-gs-job gs_scan_then_tddft/gs_scan/gs_scan_point_p020 \
  --seed-tddft-job gs_scan_then_tddft/tddft_rks_triplets_sequential/tddft_step_2.0552 \
  --bonds-file collective_bonds.dat \
  --gs-template /path/to/tdentrack/scripts/gs_scan_sequential_template.inp \
  --tddft-template /path/to/tdentrack/scripts/tddft_rks_triplets_sequential.inp \
  --gs-charge 2 \
  --gs-mult 1 \
  --tddft-charge 2 \
  --stop-coordinate 2.8 \
  --coordinate-step 0.01 \
  --orca /absolute/path/to/orca \
  --workdir gs_scan_then_tddft_extension_2p8 \
  --dry-run
```

Remove `--dry-run` to execute. After an interruption, rerun the same command
with `--resume`. The supplied GS template deliberately preserves the archived
UKS/multiplicity-1 ground-state methodology, whereas the TDDFT template uses
the successful restricted singlet reference with `triplets true`.

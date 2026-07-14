# TDenTrack Manual

Version: 0.1.0

TDenTrack is a post-processing tool for numerical state following and approximate diabatization of completed ORCA TDDFT scans. It was developed for relaxed ground-state scan plus vertical TDDFT workflows, but it also supports generic job lists through a CSV input.

The central design rule is simple: state identity is assigned from numerical wavefunction-like overlap data, not orbital pictures, fragment labels, or cube similarity.

## 1. Scope

TDenTrack reads completed ORCA TDDFT calculations and builds state-state similarity matrices between scan points. It then assigns adiabatic roots into diabatic tracks using global one-to-one matching, while explicitly reporting low-confidence assignments, mixed manifolds, and locally stable branch segments.

The tool does not rerun TDDFT calculations. It may call ORCA utilities, and for the trusted overlap backend it may run small auxiliary ORCA jobs used only to extract cross-geometry AO overlap matrices.

Supported layout:

```text
WORKDIR/
  gs_scan/
    gs_scan_summary.csv
  tddft/
    tddft_step_p000/
      job.out
      geom.xyz
      job.gbw
      job.cis
      job.json
      job.s1.nto
      ...
    tddft_step_p001/
      ...
```

Generic job CSV input is also supported:

```text
order,label,step,out,geom,gbw,uno,nto_pattern
```

where `nto_pattern` may contain `{state}`, for example `/path/to/job.s{state}.nto`.

## 2. Software Name

The software is named **TDenTrack**.

The name is intentionally descriptive: the preferred engine follows states through transition-density overlap. This helps keep the project scientifically honest: NTO images are diagnostic and interpretive, while the default state-tracking metric is numerical transition-density similarity.

The Python package remains `excited_state_diabatizer` for compatibility. The command-line entry point can be run either as:

```bash
python orca_diabatize.py
```

or, after installation:

```bash
tdentrack
```

## 3. Installation

From the repository root:

```bash
python -m pip install -e .
```

Required Python packages:

- `numpy`
- `scipy`

Optional packages/tools:

- `pyscf`, required for self-derived NTO cube generation and the Molden/PySCF fallback route
- `jmol`, required to render cube files to PNG images
- ORCA executables, especially `orca` and `orca_2json`
- `orca_2mkl`, only for the Molden fallback engine

For the current Ru scan environment:

```bash
/home/diegoa/miniconda3/bin/python -m pip install -e .
```

## 4. Recommended Command

For the current ORCA 6.1.1 scan directory, the recommended numerical tracking run is:

```bash
/home/diegoa/miniconda3/bin/python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine tden-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --reference-mode previous \
  --consensus-tracking \
  --outdir state_tracking_tden_cis_consensus \
  --html
```

`--ao-overlap-mode` defaults to `auto`. For `tden-json` and `nto-json`, `auto` uses ORCA auxiliary cross-overlap jobs. This is currently the trusted route because it preserves ORCA's AO ordering and spherical/cartesian conventions. For `nto-molden`, `auto` uses the PySCF cross-overlap backend.

## 5. Engines

### 5.1 `tden-json`

This is the recommended engine.

Despite the historical name, the current trusted path is:

1. Parse `job.out` for final energies, TDDFT roots, spin diagnostics, CI coefficient checks, and printed NTO weights.
2. Load ORCA `job.cis` binary amplitudes.
3. Validate selected `.cis` amplitudes against printed `c=` coefficients from `job.out` when available.
4. Load MO coefficients and same-geometry AO overlap matrices from `job.json`, generating richer `orca_2json` exports when needed.
5. Generate cross-geometry AO overlap matrices using auxiliary ORCA calculations, then extract the corresponding S matrix with `orca_2json`.
6. Build root-root transition-density similarity matrices.

For a state `I` at geometry `A` and state `J` at geometry `B`, the similarity is conceptually:

```text
S_IJ = | <T_I(A) | T_J(B)> | / sqrt(<T_I(A)|T_I(A)> <T_J(B)|T_J(B)>)
```

For unrestricted calculations, alpha and beta spin blocks are handled separately and summed only within matching spin channels.

In the implementation, for each spin:

```text
S_MO_AB = C_A^T S_AO_AB C_B
```

The occupied and virtual subblocks of `S_MO_AB` contract the CIS/TDA amplitude matrices:

```text
sum_ia,jb T_A(ia) S_occ_AB(i,j) S_virt_AB(a,b) T_B(jb)
```

The final value is normalized and absolute-valued. This makes the metric insensitive to arbitrary global phase changes in the transition vector.

### 5.2 `nto-json`

This engine is kept for diagnostics and comparison. It tries to read NTO orbital coefficients from JSON exports of ORCA `.nto` files. Because ORCA `.nto` JSON orbital indexing can be ambiguous, this route should not be treated as the primary scientific result unless the mapping is independently validated.

Options relevant to this engine:

```bash
--nto-index-shift 0
--nto-pair-selection all|dominant-per-spin|dominant-total
--nto-json-index-window N
```

The index window mode is explicitly diagnostic. It can reveal nearby vector-index candidates, but it is not proof of the correct ORCA index convention.

### 5.3 `nto-molden`

This engine uses `orca_2mkl -molden -anyorbs` as a fallback. Molden is useful for cross-checking early development but is not the preferred route because AO ordering, normalization, spherical harmonics, and orbital indexing can become ambiguous.

For this engine, `--ao-overlap-mode auto` maps to `pyscf-cross`. Same-geometry validation should be used to make sure the PySCF AO representation agrees with ORCA before trusting cross-geometry overlaps.

## 6. Cross-Geometry AO Overlaps

A same-geometry AO overlap matrix is not enough for state tracking across geometries. TDenTrack needs:

```text
S_AO_AB(mu,nu) = < chi_mu(R_A) | chi_nu(R_B) >
```

The trusted implementation uses `orca-auxiliary` for `tden-json` and `nto-json`:

1. Create a small auxiliary ORCA input containing the atoms of geometry A followed by geometry B.
2. Use the same basis directives inferred from the original ORCA input.
3. Run ORCA to obtain a combined `cross.gbw`.
4. Run `orca_2json` with overlap export enabled.
5. Extract the off-diagonal AO overlap block.
6. Validate that the diagonal blocks reproduce the same-geometry AO overlap matrices already exported for A and B.

The extracted cross block is cached in:

```text
OUTDIR/json_cache/cross_pXXX_pYYY.json
```

Auxiliary logs are saved in:

```text
OUTDIR/utility_logs/
```

## 7. Same-Geometry Orthonormality Validation

For each parsed MO or NTO orbital source, TDenTrack validates:

```text
M = C^T S_AO_AA C
```

The diagnostics report:

- maximum diagonal deviation from 1
- maximum absolute off-diagonal element
- RMS off-diagonal element
- pass/fail

The output file is:

```text
OUTDIR/diagnostics/orthonormality_checks.csv
```

If validation fails, the run stops unless `--allow-failed-orthonormality` is passed. That override should be used only for dangerous debugging. A failure usually means AO ordering, spherical/cartesian convention, normalization, or orbital index mapping is inconsistent.

## 8. Assignment Algorithm

For every adjacent step pair, TDenTrack builds a complete root-root similarity matrix. It then assigns roots with a global Hungarian maximum-weight matching:

```text
scipy.optimize.linear_sum_assignment(-S)
```

This is important. Greedy row-wise matching can assign two diabatic tracks to the same current root and can fail badly near crossings.

Each assigned edge receives:

- best similarity
- second-best similarity
- margin
- ratio
- confidence label
- reason string
- previous-step similarity
- anchor/reference similarity, when applicable

Default confidence logic:

```text
reliable       best >= 0.70 and margin >= 0.10
low_confidence best >= 0.45 but below one reliable threshold
failed         best < 0.45
ambiguous      subspace/purity/rescue logic marks the edge provisional
```

Thresholds are CLI-configurable.

## 9. Reference Modes

### `previous`

Compare each track's current root to roots at the next geometry. This is the most local tracking mode and is the recommended baseline for smooth scans.

### `first`

Compare every step to the initial root assignment. This is strict but can fail if the state character evolves substantially.

### `adaptive`

Use the last stable anchor for each track. The anchor updates only when the assignment is reliable and not inside an ambiguous manifold.

### `hybrid`

Combine previous-step and anchor-reference scores:

```text
S_combined = previous_weight * S_previous + anchor_weight * S_anchor
```

The `predictor_weight` CLI option is reserved for a future two-step predictor. It is currently kept for API stability.

## 10. NTO Purity Guard and Ambiguity Rescue

For `tden-json`, TDenTrack derives NTOs itself by SVD of the validated CIS/TDA amplitude matrices. This avoids relying on ORCA `.nto` JSON orbital indexing.

For each root, the diagnostic purity is:

```text
dominant_pair_weight / sum_selected_pair_weights
```

If `--nto-purity-guard` is active, a current root whose dominant pair fraction is below `--nto-purity-threshold` is treated as mixed/provisional. The default threshold is 0.80.

If `--ambiguity-rescue` is active, a broken track does not keep comparing to the immediately previous mixed state. Instead, it compares future roots to the last stable pre-ambiguity anchor. This avoids poisoning a track after a mixed point.

Important limitation:

A later pure state is not automatically assigned a new identity just because it looks internally simple. It must either reconnect numerically to a stable anchor or appear as a locally stable segment. This is intentional: local purity is not the same as diabatic identity.

## 11. Subspace and Manifold Detection

TDenTrack does not force a unique root label when the data indicate a mixed manifold.

For each adjacent pair, it checks for close competitors:

- second_best / best greater than `--subspace-ratio-threshold`
- or best - second_best below `--assignment-margin-threshold`
- and roots close in energy within `--subspace-gap-ev`

Candidate manifolds are reported with a subspace score:

```text
S_subspace = || S[P,Q] ||_F / sqrt(size)
```

Outputs:

```text
ambiguous_regions.csv
manifold_episodes.csv
```

The HTML report summarizes these episodes and marks assignments inside them as provisional.

## 12. Reverse and Consensus Tracking

`--bidirectional` performs a local consistency check on each adjacent similarity matrix:

- run Hungarian matching forward on `S`
- run Hungarian matching backward on `S^T`
- report mismatches

It writes:

```text
bidirectional_disagreements.csv
```

`--reverse-tracking` runs a full right-to-left tracking pass and writes reverse diagnostics:

```text
reverse_diabatic_assignments.csv
reverse_assignment_confidence.csv
reverse_ambiguous_regions.csv
reverse_manifold_episodes.csv
```

`--consensus-tracking` enables reverse tracking and reports locally stable branch segments. These segments are useful when a track is ambiguous on one side of the scan but becomes well defined later.

Segment criteria:

```text
--segment-min-purity 0.90
--segment-min-adjacent-similarity 0.85
--segment-min-length 2
```

Output:

```text
track_segments.csv
```

A segment is evidence for local continuity, not proof that the branch connects to a pre-ambiguity anchor.

## 13. Self-Derived NTO Visualization

TDenTrack can render NTOs derived from the same CIS/TDA amplitudes used for tracking:

```bash
--render-self-nto-cubes
--self-nto-plot-steps 0-15
--self-nto-plot-roots 4-10
--self-nto-plot-weight-min 0.10
--self-nto-plot-max-pairs 2
--self-nto-cube-grid 64 64 64
--jmol /snap/bin/jmol
```

The code uses PySCF to evaluate the self-derived NTO orbitals on a grid. Before cube generation, it compares the PySCF same-geometry AO overlap against the ORCA same-geometry AO overlap. If the maximum absolute error exceeds `--self-nto-cube-overlap-tol`, visualization fails rather than producing misleading images.

Outputs:

```text
OUTDIR/self_nto_cubes/
OUTDIR/self_nto_pngs/
OUTDIR/diagnostics/self_nto_visualizations.csv
```

These cubes and PNGs are for interpretation and consistency checks. They are never used as the primary tracking metric.

## 14. Outputs

Primary files:

- `diabatic_assignments.csv`: one row per track and step
- `tracked_state_energies.csv`: tracked absolute energies and excitation energies
- `adjacent_similarity_matrices.csv`: long-format adjacent similarity matrices
- `adjacent_similarity_matrices.npz`: NumPy archive of adjacent matrices
- `assignment_confidence.csv`: edge-level confidence diagnostics
- `ambiguous_regions.csv`: adjacent mixed-subspace detections
- `manifold_episodes.csv`: coalesced manifold episodes
- `bidirectional_disagreements.csv`: local forward/backward mismatches
- `track_segments.csv`: locally stable branch segments
- `state_tracking_report.html`: human-readable report when `--html` is passed

Diagnostics:

- `diagnostics/extraction_status.csv`
- `diagnostics/orthonormality_checks.csv`
- `diagnostics/cis_amplitude_checks.csv`
- `diagnostics/self_nto_weights.csv`
- `diagnostics/self_nto_reconstruction.csv`
- `diagnostics/nto_purity.csv`
- `diagnostics/self_nto_visualizations.csv`
- `diagnostics/nto_index_mapping.csv`, mainly for NTO JSON/Molden fallback work

Caches and logs:

- `json_cache/`
- `molden_cache/`
- `utility_logs/`

## 15. Reading the HTML Report

The report includes:

- executive summary
- absolute adiabatic energies
- absolute diabatic tracked energies
- adiabatic excitation energies
- diabatic tracked excitation energies
- similarity heatmaps
- per-track root sequences
- manifold and ambiguity summaries
- stable local segment table
- self-derived NTO purity diagnostics
- self-derived NTO visualization gallery, if requested
- extraction and orthonormality diagnostics

Grey/dashed energy-plot segments indicate low-confidence, ambiguous, failed, or manifold-tagged assignments. They are scientific warnings, not styling errors.

## 16. Common Commands

Trusted tracking:

```bash
python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine tden-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --reference-mode previous \
  --consensus-tracking \
  --outdir state_tracking_tden_cis_consensus \
  --html
```

Force fresh JSON and cross-overlap extraction:

```bash
python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine tden-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --force-json \
  --outdir state_tracking_fresh \
  --html
```

Render self-derived NTOs:

```bash
python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine tden-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --consensus-tracking \
  --render-self-nto-cubes \
  --self-nto-plot-steps 0-15 \
  --self-nto-plot-roots 4-10 \
  --self-nto-plot-weight-min 0.10 \
  --self-nto-plot-max-pairs 2 \
  --jmol /snap/bin/jmol \
  --outdir state_tracking_self_nto \
  --html
```

Run the diagnostic NTO JSON route:

```bash
python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine nto-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --outdir state_tracking_nto_json \
  --html
```

## 17. Known Limitations

- `tden-json` is the trusted engine, but it currently relies on ORCA `.cis` amplitudes plus JSON MO coefficients. If either is missing, tracking cannot proceed.
- `json-cross` is not implemented as an independent cross-overlap source. Same-geometry JSON S matrices are not cross-geometry overlaps.
- The PySCF cross-overlap backend is available for Molden fallback workflows, but the trusted transition-density path uses ORCA auxiliary overlap jobs.
- Self-derived NTOs are diagnostic. A pure NTO pair does not by itself prove diabatic identity across a previous ambiguous region.
- The two-step predictor option is reserved for future development.
- Automatic insertion and execution of intermediate scan points is planned but not implemented in this version.
- Wavefunction-overlap tracking for critical regions is planned but not implemented in this version.

## 18. Development Checks

Run tests:

```bash
python -m unittest discover -s excited_state_diabatizer/tests
```

Compile check:

```bash
python -m compileall -q excited_state_diabatizer orca_diabatize.py
```

Show CLI help:

```bash
python orca_diabatize.py --help
```

## 19. Recommended Scientific Workflow

1. Run the trusted `tden-json` workflow with `--consensus-tracking`.
2. Inspect `assignment_confidence.csv`, `ambiguous_regions.csv`, and `manifold_episodes.csv`.
3. Inspect stable segments in `track_segments.csv`.
4. Render self-derived NTOs for ambiguous or chemically important roots.
5. Treat grey/dashed tracks and low similarity as warnings, not as inconveniences to hide.
6. If a critical state disappears into a manifold or beyond the root window, add intermediate scan points or include more TDDFT roots.
7. Use NTO images to interpret state character, not to override failed numerical continuity.

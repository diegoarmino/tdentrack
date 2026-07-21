# TDenTrack Manual

Version: 0.1.0

TDenTrack is a post-processing tool for numerical state following and approximate diabatization of completed ORCA TDDFT scans. It was developed for relaxed ground-state scan plus vertical TDDFT workflows, but it also supports generic job lists through a CSV input.

The central design rule is simple: state identity is assigned from numerical wavefunction-like overlap data, not orbital pictures, fragment labels, or cube similarity.

## 1. Scope

TDenTrack reads completed ORCA TDDFT calculations and builds state-state similarity matrices between scan points. It then assigns adiabatic roots into diabatic tracks using global one-to-one matching, while explicitly reporting low-confidence assignments, mixed manifolds, and locally stable branch segments. It also provides an experimental transactional selection API used by the accompanying pysisyphus integration for live excited-state optimization.

The established scan-analysis CLI does not rerun the production TDDFT calculations. It may call ORCA utilities, and for the trusted overlap backend it may run small auxiliary ORCA jobs used only to extract cross-geometry AO overlap matrices. The optimizer-facing Python API is different: its pysisyphus adapter deliberately launches isolated all-root surveys and selected-root gradient calculations.

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

The active occupied and virtual ranges stored in an ORCA `.cis` header are
**zero-based and inclusive**. TDenTrack preserves that convention internally;
for example, the range `6..20` selects MO coefficient columns `6:21`. This is
the same numbering used for CI contributions printed by ORCA and must not be
shifted when reconstructing NTOs.

The binary reader auto-detects both the historical 15-integer unrestricted-TDA
layout and ORCA's standard vector-record layout. The latter has been regression
tested against a real restricted singlet-plus-triplet .cis file produced by
ORCA 6.1.1. In a standard file, multiplicity is read from every vector record
and roots are exposed in the global STATE N order printed in job.out.
This ordering is deliberate: ORCA 6.1.1 can repeat the stored internal iroot
value in the triplet block, so iroot alone cannot identify the printed root.
Each parsed state therefore also records a stable one-based
root_within_multiplicity ordinal for translating a global output root to the
corresponding singlet or triplet root window. The dictionary key, root, and
global_root fields are the mixed-file STATE N label. For unrestricted
open-shell calculations this global label is also the ORCA gradient IRoot,
even when states with different approximate multiplicity are interleaved. The
multiplicity-local orca_gradient_iroot field is instead the required
translation for spin-adapted multiplicity blocks. For example, global triplet
STATE 4 in a restricted singlet-plus-triplet file with two preceding singlets
has orca_gradient_iroot=2 and must be selected with IRoot 2 and IRootMult
triplet. The pysisyphus adapter chooses the convention from the reference type
before setting ORCA's root/IRoot value.
When all requested output roots have the same printed multiplicity, TDenTrack
uses it as a parser filter and verifies the binary/output multiplicities and
excitation energies agree.

The binary format has no explicit TDA flag. In the special case of two
same-energy records with the same stored root and multiplicity, the bytes alone
cannot distinguish one non-TDA X+Y/X-Y pair from two exactly degenerate TDA
roots. Validation reads ORCA's printed Tamm-Dancoff status to resolve that case;
direct parser users must pass an explicit tda=True or tda=False hint. The
parser refuses to guess when the case remains ambiguous.

For a restricted standard file, ORCA writes -1 sentinels instead of beta
active-orbital ranges and stores one spin-adapted spatial vector. TDenTrack
mirrors the alpha ranges for beta and reconstructs equal alpha/beta blocks,
each scaled by 1/sqrt(2) so their combined squared norm remains one. Printed
c= validation rescales a block back to ORCA's spatial coefficient. The
manifest records the layout, restriction flag, TDA flag, stored-vector count,
and available multiplicities. Truncated files, trailing data, non-finite
coefficients, inconsistent vector sizes, and multiplicity mismatches fail
closed. Spin-flip standard vectors are not currently supported.

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
7. Validate that the two off-diagonal blocks are mutual transposes within the same numerical tolerance.

An auxiliary result is accepted only when ORCA exits successfully, prints its
`ORCA TERMINATED NORMALLY` marker, produces the GBW, and `orca_2json` exits
successfully and produces the JSON export. A leftover GBW from an interrupted
calculation is therefore not sufficient for acceptance.

The extracted cross block is cached in:

```text
OUTDIR/json_cache/cross_pXXX_pYYY_<input-hash>.json
```

The hash covers the generated auxiliary input, including both geometries,
their charge, and the AO basis directives. Reusing a step label after changing
a geometry or basis cannot silently select the old matrix. Legacy unhashed
cache files are intentionally not reused.

Both source calculations must use identical basis specifications. TDenTrack
reads all ORCA `!` input lines and preserves a `%basis` block, including nested
`NewGTO`, `NewAuxGTO`, and `NewECP` sections. Unusual atom-index-specific,
external-file, ECP, or relativistic basis inputs still require inspection of
the generated `cross.inp` and validation against a real ORCA fixture. The
mandatory diagonal-block checks are the final guard against a basis ordering
or duplication error.

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

This Frobenius score is the legacy scan-reporting heuristic. Optimizer-facing
code can use the stricter, signed principal-angle analysis in
`excited_state_diabatizer.state_tracking`. Given the full root-to-root block
`O` between reference roots `P` and candidate roots `Q`, use:

```python
from excited_state_diabatizer import analyze_subspace_continuity

continuity = analyze_subspace_continuity(
    O,
    reference_roots=P,
    candidate_roots=Q,
    reference_norms=norms_P,
    candidate_norms=norms_Q,
    # Supply these for non-orthogonal transition-density descriptors:
    reference_gram=gram_P,
    candidate_gram=gram_Q,
)
```

The utility first normalizes the overlap block while retaining every sign. If
self-overlap Gram matrices are supplied, it symmetrically orthonormalizes both
sets and then performs an SVD. The singular values are cosines of the principal
angles between the manifolds. A root rotation, phase flip, or permutation can
therefore give singular values near one even when individual-root assignment
is ambiguous. The conservative continuity score is
`continuity.minimum_singular_value`; the corresponding diagnostic is
`continuity.maximum_principal_angle_deg`.

If Gram matrices are omitted, roots within each set are assumed orthogonal in
the chosen overlap metric. That assumption is not generally valid for simple
Frobenius overlaps of transition densities, so rigorous manifold decisions
should calculate the within-geometry root-root Gram blocks as well as the
cross-geometry block.

An optimization survey can attach this immutable result through
`StateSurvey.subspace_continuity`. Setting, for example,
`SelectionConfig(min_subspace_singular_value=0.80)` makes it an explicit gate:
a near-degenerate pair is reported as `MANIFOLD` only when every principal
direction passes. Missing, dimension-mismatched, or weak subspace information
returns `RETRY`. A `MANIFOLD` decision always has `selected_root=None` and
cannot be committed by `TrackingSession`; it is evidence of manifold
continuity, not permission to calculate or commit an arbitrary root gradient.

For production optimization, attach the complete root windows instead of
preselecting a manifold in the backend:

```python
from excited_state_diabatizer import RootOverlapBlock

root_overlaps = RootOverlapBlock(
    reference_roots=reference_roots,
    candidate_roots=candidate_roots,
    overlaps=signed_cross_block,
    reference_gram=reference_self_overlap_block,
    candidate_gram=candidate_self_overlap_block,
)
survey = session.survey(
    candidate_snapshot,
    signed_selected_root_overlaps,
    root_overlap_block=root_overlaps,
)
```

`RootOverlapBlock` validates and immutably owns the signed cross block and both
full self-overlap Gram matrices. Once scalar scores and energies reveal a
candidate manifold, `select_state` derives the reference manifold from roots
of the same multiplicity lying within `manifold_gap_ev` of the selected
reference root, extracts exactly that sub-block, and calculates its principal
angles. Thus a four-root calculation can analyze only the rotating two-state
manifold without the other roots diluting its score. `RootOverlapBlock.analyze`
and `.subset` expose the same operation for offline diagnostics.

With `require_equal_subspace_dimensions=True` (the default), unavailable
reference energy gaps or unequal reference/candidate manifold sizes produce
`RETRY`; these conditions can indicate an incomplete root window or a genuine
split/merge region. If the selected reference multiplicity is known, candidate
roots with missing multiplicity metadata are ineligible rather than being
silently assumed to match.

`SelectionDecision.signed_normalized_scores` retains the signed one-root
overlaps used before absolute-value ranking. This is required for later polar
or SVD transport even though ordinary root ranking remains phase-insensitive.

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

## 20. Experimental Excited-State Optimization

The implementation is divided at a transactional boundary:

- pysisyphus owns the geometry, RFO model, trust radius, convergence checks, and optimizer history;
- ORCA 6.1.1 supplies an all-root energy survey and the gradient for the root selected at an accepted endpoint;
- TDenTrack owns immutable electronic snapshots, root/manifold decisions, and commit authorization.

This differs from running a one-cycle native ORCA optimizer. It keeps every trial geometry visible to the same state selector and prevents a rejected trial from changing the tracked root, GBW reference, or quasi-Newton history. ORCA `ExtOpt` is not used: that facility replaces the energy/gradient provider while leaving ORCA in charge of geometry optimization, whereas this design needs pysisyphus to inspect several possible endpoints before one enters optimizer history.

### 20.1 One optimization transaction

For each gradient-evaluated geometry:

1. pysisyphus constructs its normal optimizer proposal.
2. The unscaled endpoint is surveyed first by an isolated, energy-only ORCA all-root job.
3. ORCA BSON wavefunctions at the committed and proposed geometries provide the analytic cross matrix `S_AB = <AO(A)|AO(B)>`. This is the same adjacent-geometry superposition quantity sought by a double-molecule calculation, but it is evaluated directly from the two retained shell sets with `Wavefunction.S_with`. Atom order, coordinates, shell definitions, ECP data, AO ordering, MO orthonormality, and forward/reverse transpose symmetry are checked before it is used.
4. The `.cis` amplitudes, `S_AB`, and the two same-geometry metrics produce a full signed root-overlap block plus both root self-overlap Gram matrices.
5. `TrackingSession` returns `ACCEPT`, `MANIFOLD`, `RETRY`, or `HALT`. Only `ACCEPT` carries a committable root.
6. pysisyphus stages the selected endpoint and requests an ORCA `EnGrad` calculation. A restricted spin-adapted triplet uses the root's multiplicity-local `IRoot` and explicitly sets `IRootMult triplet`; `Triplets true` alone does not make ORCA 6.1.1's `IRoot` select the triplet block. An unrestricted open-shell calculation instead uses the global printed `STATE N` ordinal as `IRoot N` and preserves CIS multiplicities separately for state-selection guards.
7. A finite, geometry-matched successful gradient atomically commits both the electronic snapshot and geometry step. Before commitment, the adapter verifies normal termination and agreement among ORCA's echoed `IRoot`/`IRootMult`, `DE(CIS)` root marker, and state-of-interest report. `FollowIRoot true` and `TGradList` are rejected. Failure restores the prior root and leaves the snapshot uncommitted.

ORCA's TDDFT state table combines the printed `E(SCF)` with excitation
energies, but its final EnGrad energy can include state-independent terms added
later, notably D3(BJ). The backend anchors the bootstrap's selected root—and
root zero in an energy-only survey—to `FINAL SINGLE POINT ENERGY`, then applies
that common correction to every state. This preserves excitation energies and
puts optimizer energies, descent tests, and fallback ranking on the same total-
energy scale. The correction and anchor are retained in the audit metadata.

Implicit-solvent runs must also set `CPCMEQ` explicitly in the TDDFT block.
ORCA's job-type defaults differ: an energy-only vertical calculation uses
non-equilibrium LR-CPCM, whereas requesting an analytic excited-state gradient
switches to equilibrium LR-CPCM. Because the transactional workflow alternates
all-root energy surveys and selected-root gradients, an omitted `CPCMEQ` would
silently compare different surfaces. The backend fails closed in that case.
Use `CPCMEQ true` for the usual relaxed excited-state optimization, or
explicitly choose `false` for a deliberately frozen-solvent calculation.

All-root surveys retain their input, output, CIS, BSON, GBW, and a JSON audit manifest in unique directories. Restart data serialize only the last committed snapshot; pending or merely staged trials must be surveyed again.

### 20.2 Mixed and near-degenerate regions

The default step controller surveys the optimizer's factor `1.0` proposal first. If that endpoint is uniquely identified and descending, it is used without launching fallback jobs. If it is mixed, weak, incomplete, or uphill, a bounded set of shorter and longer factors can be evaluated. This implements both ways of leaving a narrow mixing region: approach it more cautiously or bridge to a clean endpoint beyond it.

Fallback endpoints still have to pass root-window completeness, multiplicity, normalized-overlap, assignment-margin, energy, and maximum-step guards. Among acceptable fallbacks, the lowest state energy is preferred. A larger factor does not receive special permission merely because it leaves the mixed region; users who intentionally allow it beyond the optimizer's trust maximum must set an explicit overall step bound.

When close roots form a candidate manifold, the selector extracts the matching same-spin, energy-local reference and candidate subsets from the full `RootOverlapBlock` and evaluates their principal angles. This confirms continuity of the *manifold*, not a unique member of it. The decision therefore remains noncommittable and the controller tries another bounded endpoint. Missing energy gaps, unequal manifold dimensions, weak minimum singular values, or an exhausted factor set stop the optimization for inspection rather than forcing a root.

### 20.3 Scope and current limitations

- The integration is an opt-in Python API in the modified pysisyphus fork; it is not yet a stable YAML or TDenTrack CLI workflow.
- The built-in backend is deliberately version-gated to ORCA 6.1.1 and assumes a fixed atom order, basis/ECP definition, charge, multiplicity, and contiguous root window. Root ordinals are multiplicity-local for restricted spin-adapted blocks and global for unrestricted open-shell references.
- Point charges and other per-call Hamiltonian inputs must be supplied identically to surveys and gradients.
- Spin-flip CIS vectors are unsupported. Restricted ORCA 6.1.1 TDA is covered by a real binary fixture; non-TDA and unrestricted branches also have synthetic parser regressions but should be fixture-validated for the intended production method.
- The legacy pysisyphus `track: true` ORCA implementation uses its older mutable tracker and raw-GBW parser. It is not part of this transaction path and should not be substituted for `TDenTrackORCA` under ORCA 6.1.1.
- A clean endpoint reached by a longer step is a state-following heuristic, not a multistate treatment of a conical intersection or genuinely degenerate surface.

The concrete bootstrap and calculator construction example is in `docs/es_optimization.rst` in the modified pysisyphus fork.

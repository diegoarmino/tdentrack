# TDenTrack

TDenTrack is a numerical excited-state tracking and diabatization post-processor for ORCA TDDFT scans. It is designed for completed ORCA scan or optimization directories and does not rerun the TDDFT jobs it analyzes.

The trusted engine is transition-density overlap from ORCA `job.cis` amplitudes plus ORCA-generated cross-geometry AO overlaps. Self-derived NTOs are reconstructed from the same CIS/TDA amplitudes for diagnostics and visualization, avoiding the `.nto` JSON indexing ambiguity that motivated this tool.

## Quick Start

From an ORCA scan directory with `tddft/tddft_step_p000`, `tddft_step_p001`, etc.:

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

Optional self-derived NTO cube/PNG rendering:

```bash
/home/diegoa/miniconda3/bin/python orca_diabatize.py \
  --workdir . \
  --roots 1-15 \
  --engine tden-json \
  --orca /home/diegoa/orca_6_1_1/orca \
  --orca-2json /home/diegoa/orca_6_1_1/orca_2json \
  --reference-mode previous \
  --consensus-tracking \
  --render-self-nto-cubes \
  --self-nto-plot-steps 0-15 \
  --self-nto-plot-roots 4-10 \
  --self-nto-plot-weight-min 0.10 \
  --self-nto-plot-max-pairs 2 \
  --jmol /snap/bin/jmol \
  --outdir state_tracking_tden_cis_consensus_nto \
  --html
```

## What It Produces

The output directory contains:

- `state_tracking_report.html`
- `diabatic_assignments.csv`
- `tracked_state_energies.csv`
- `adjacent_similarity_matrices.csv`
- `adjacent_similarity_matrices.npz`
- `assignment_confidence.csv`
- `ambiguous_regions.csv`
- `manifold_episodes.csv`
- `track_segments.csv`
- `diagnostics/` with extraction, orthonormality, CIS-amplitude, self-NTO, purity, and visualization diagnostics
- `json_cache/`, `molden_cache/`, and `utility_logs/`

## Manual

See [docs/MANUAL.md](docs/MANUAL.md) for the current scientific model, command-line reference, output definitions, diagnostics, and known limitations.

## Scientific Caution

TDenTrack is deliberately conservative. Low overlap, close competitors, mixed NTO character, or forward/reverse disagreement should be treated as evidence for an ambiguous manifold or missing scan resolution, not as a uniquely followed diabatic state.

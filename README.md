# TDenTrack

TDenTrack provides numerical excited-state tracking and diabatization for ORCA TDDFT data. Its established command-line workflow is a post-processor for completed scan or optimization directories. It now also exposes an experimental transactional state-selection API for excited-state geometry optimization with the accompanying pysisyphus integration.

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

## Experimental ORCA 6.1.1 Optimization

The optimizer-facing API separates geometry proposals from electronic-state commitment:

1. pysisyphus proposes an RFO step.
2. An isolated ORCA 6.1.1 all-root calculation evaluates the proposed geometry.
3. Exact adjacent-geometry AO overlaps, signed transition-density overlaps, energies, and multiplicities are passed to `TrackingSession`.
4. A unique accepted root is used for an ORCA `EnGrad` calculation. Only a successful gradient commits the new electronic reference and optimizer step.
5. If the original endpoint is mixed, weak, or uphill, bounded shorter and longer proposals can be surveyed. A near-degenerate manifold is never committed as an arbitrary single root.

This is currently a Python API rather than a TDenTrack CLI command. The implementation and runnable setup contract are documented in `docs/es_optimization.rst` in the modified pysisyphus fork. The legacy pysisyphus `track: true` ORCA route is separate and is not used by this backend.

## Scientific Caution

TDenTrack is deliberately conservative. Low overlap, close competitors, mixed NTO character, or forward/reverse disagreement should be treated as evidence for an ambiguous manifold or missing scan resolution, not as a uniquely followed diabatic state.

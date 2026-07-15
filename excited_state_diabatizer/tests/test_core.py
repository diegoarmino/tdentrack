import tempfile
import struct
import unittest
from pathlib import Path

import numpy as np

from excited_state_diabatizer.basis_overlap import validate_orbital_orthonormality
from excited_state_diabatizer.cis_io import coefficient_from_amplitudes, parse_cis_amplitudes, validate_cis_against_output
from excited_state_diabatizer.json_io import parse_orca_json
from excited_state_diabatizer.models import AssignmentEdge, JobRecord, NTOOrbitalPair, NTOStateVector, SelfNTOWeightRecord, StateRecord, StepData
from excited_state_diabatizer.molden_io import parse_molden
from excited_state_diabatizer.nto_from_tden import derive_self_nto_state, derive_self_ntos_from_tden_steps
from excited_state_diabatizer.orca_io import parse_ci_coefficients, parse_nto_blocks, parse_tddft_output
from excited_state_diabatizer.overlap import cis_transition_density_overlap_matrix
from excited_state_diabatizer.report import write_outputs
from excited_state_diabatizer.segments import build_track_segments, make_reversed_matrix_provider
from excited_state_diabatizer.tracking import (
    TrackingConfig,
    classify_confidence,
    detect_subspaces_for_pair,
    hungarian_assignment,
    run_tracking,
)


class CoreTests(unittest.TestCase):
    def test_parse_orca_state_lines(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "job.out"
            out.write_text(
                """
FINAL SINGLE POINT ENERGY     -123.456789
STATE  1:  E=   0.034116 au      0.928 eV     7487.7 cm**-1 <S**2> =   2.025176 Mult 3
STATE  2:  E=   0.036441 au      0.992 eV     7997.5 cm**-1 <S**2> =   2.026000 Mult 3
ORCA TERMINATED NORMALLY
"""
            )
            ref, states, normal = parse_tddft_output(out, "p000", 0, 0.0)
        self.assertTrue(normal)
        self.assertEqual(ref, -123.456789)
        self.assertEqual(states[1].exc_ev, 0.928)
        self.assertEqual(states[1].multiplicity, 3)
        self.assertTrue(np.isclose(states[2].abs_energy_eh, -123.456789 + 0.036441))

    def test_parse_ci_coefficients(self):
        text = """
STATE  1:  E=   0.034116 au      0.928 eV     7487.7 cm**-1
   103a -> 104a  :     0.236391 (c= -0.48620072)
   101b -> 102b  :     0.462324 (c= -0.67994427)

STATE  2:  E=   0.036439 au      0.992 eV     7997.5 cm**-1
   100b -> 102b  :     0.883414 (c=  0.93990089)

NATURAL TRANSITION ORBITALS FOR STATE    1
   103a -> 104a  : n=  0.97877209
"""
        coeffs = parse_ci_coefficients(text)
        self.assertEqual(sorted(coeffs), [1, 2])
        self.assertEqual(coeffs[1][0].donor, 103)
        self.assertEqual(coeffs[1][1].donor_spin, "b")
        self.assertTrue(np.isclose(coeffs[2][0].coefficient, 0.93990089))

    def test_parse_nto_blocks(self):
        text = """
NATURAL TRANSITION ORBITALS FOR STATE    1
   103a -> 104a  : n=  0.97877209
   101b -> 102b  : n=  0.01196498
NATURAL TRANSITION ORBITALS FOR STATE    2
   101b -> 102b  : n=  0.97958384
"""
        blocks = parse_nto_blocks(text)
        self.assertEqual(sorted(blocks), [1, 2])
        self.assertEqual(blocks[1][0].donor, 103)
        self.assertEqual(blocks[1][1].donor_spin, "b")
        self.assertEqual(blocks[2][0].weight, 0.97958384)

    def test_parse_minimal_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "job.json"
            path.write_text(
                """
{
  "Molecule": {
    "Atoms": [{"ElementLabel": "H", "ElementNumber": 1, "Coords": [0, 0, 0]}],
    "S-Matrix": [[1.0, 0.0], [0.0, 1.0]]
  },
  "mo_coefficients": [[1, 0], [0, 1]],
  "nto_states": [{"root": 1, "pairs": [{"weight": 1.0, "donor": [1,0], "acceptor": [0,1]}]}]
}
"""
            )
            parsed = parse_orca_json(path)
        self.assertEqual(parsed.atoms[0]["label"], "H")
        self.assertEqual(parsed.overlap.shape, (2, 2))
        self.assertIn("alpha", parsed.mo_coefficients)
        self.assertIn(1, parsed.nto_states)

    def test_parse_orca_molecular_orbitals_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "job.json"
            path.write_text(
                """
{
  "Molecule": {
    "HFTyp": "UHF",
    "Atoms": [{"ElementLabel": "H", "ElementNumber": 1, "Coords": [0, 0, 0]}],
    "S-Matrix": [[1.0, 0.0], [0.0, 1.0]],
    "MolecularOrbitals": {
      "OrbitalLabels": ["H 1s", "H 2s", "H 1s", "H 2s"],
      "MOs": [
        {"MOCoefficients": [1.0, 0.0], "Occupancy": 1.0},
        {"MOCoefficients": [0.0, 1.0], "Occupancy": 0.0},
        {"MOCoefficients": [0.8, 0.6], "Occupancy": 1.0},
        {"MOCoefficients": [-0.6, 0.8], "Occupancy": 0.0}
      ]
    }
  }
}
"""
            )
            parsed = parse_orca_json(path)
        self.assertEqual(parsed.mo_coefficients["alpha"].shape, (2, 2))
        self.assertEqual(parsed.mo_coefficients["beta"].shape, (2, 2))
        self.assertTrue(np.allclose(parsed.mo_coefficients["alpha"], np.eye(2)))
        self.assertTrue(np.allclose(parsed.mo_coefficients["beta"], [[0.8, -0.6], [0.6, 0.8]]))

    def test_parse_minimal_molden(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mini.molden"
            path.write_text(
                """
[Molden Format]
[Atoms] Angs
H 1 1 0.0 0.0 0.0
[GTO]
  1 0
s 1 1.0
  1.0 1.0
[MO]
 Sym= 1a
 Ene= 0.0
 Spin= Alpha
 Occup= 1.0
  1 1.0
"""
            )
            molden = parse_molden(path)
        self.assertEqual(len(molden.atoms), 1)
        self.assertEqual(len(molden.aos), 1)
        self.assertEqual(len(molden.mos), 1)
        self.assertEqual(molden.mos[0].spin, "alpha")

    def test_hungarian_beats_row_greedy(self):
        mat = np.array([[0.90, 0.80], [0.89, 0.10]])
        pairs = set(hungarian_assignment(mat))
        self.assertEqual(pairs, {(0, 1), (1, 0)})
        self.assertGreater(sum(mat[r, c] for r, c in pairs), mat[0, 0] + mat[1, 1])

    def test_confidence_classification(self):
        self.assertEqual(classify_confidence(0.8, 0.6).confidence, "reliable")
        self.assertEqual(classify_confidence(0.6, 0.52).confidence, "low_confidence")
        self.assertEqual(classify_confidence(0.2, 0.1).confidence, "failed")

    def test_subspace_detection(self):
        step_a = _step("p000", 0, {1: 1.0, 2: 1.2})
        step_b = _step("p001", 1, {1: 1.00, 2: 1.05})
        mat = np.array([[0.70, 0.66], [0.20, 0.90]])
        edges = [
            AssignmentEdge("p000", "p001", 1, 1, 1, 0.70, 0.66, 0.04, 0.94, "low_confidence", ""),
            AssignmentEdge("p000", "p001", 2, 2, 2, 0.90, 0.20, 0.70, 0.22, "reliable", ""),
        ]
        regions = detect_subspaces_for_pair(
            mat,
            [1, 2],
            edges,
            step_a,
            step_b,
            TrackingConfig(subspace_detection=True, subspace_gap_ev=0.10),
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].manifold_roots, "1/2")

    def test_subspace_detection_excludes_assigned_column(self):
        step_a = _step("p000", 0, {1: 1.0, 2: 1.2})
        step_b = _step("p001", 1, {1: 1.00, 2: 1.05})
        mat = np.array([[0.90, 0.70], [0.10, 0.95]])
        edges = [
            AssignmentEdge("p000", "p001", 1, 1, 2, 0.70, 0.90, -0.20, 1.29, "low_confidence", ""),
            AssignmentEdge("p000", "p001", 2, 2, 1, 0.10, 0.95, -0.85, 9.50, "failed", ""),
        ]
        regions = detect_subspaces_for_pair(
            mat,
            [1, 2],
            edges,
            step_a,
            step_b,
            TrackingConfig(subspace_detection=True, subspace_gap_ev=0.10),
        )
        self.assertTrue(regions)
        self.assertEqual(regions[0].manifold_roots, "1/2")

    def test_report_writes_purity_and_manifold_episodes(self):
        roots = [1, 2]
        steps = [
            _step("p000", 0, {1: 1.0, 2: 1.04}),
            _step("p001", 1, {1: 1.02, 2: 1.05}),
            _step("p002", 2, {1: 1.03, 2: 1.06}),
        ]
        mats = {
            (0, 1): np.array([[0.70, 0.66], [0.20, 0.90]]),
            (1, 2): np.array([[0.72, 0.68], [0.20, 0.89]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        result = run_tracking(
            steps,
            roots,
            provider,
            TrackingConfig(subspace_detection=True, subspace_gap_ev=0.10),
            "synthetic",
        )
        self.assertGreaterEqual(len(result.ambiguous_regions), 2)
        weights = [
            SelfNTOWeightRecord("p001", 1, "alpha", Path("job.cis"), 1, 0.8, 0.64, 0.8, 0.8, 0.8, True),
            SelfNTOWeightRecord("p001", 1, "alpha", Path("job.cis"), 2, 0.4, 0.16, 0.2, 0.2, 1.0, True),
        ]
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            write_outputs(
                outdir,
                result,
                steps,
                extraction_statuses=[],
                orthonormality=[],
                self_nto_weights=weights,
                write_html=True,
                selected_roots=roots,
                engine="synthetic",
                nto_purity_threshold=0.80,
                nto_purity_guard=True,
            )
            episodes = (outdir / "manifold_episodes.csv").read_text()
            purity = (outdir / "diagnostics" / "nto_purity.csv").read_text()
            report = (outdir / "state_tracking_report.html").read_text()
        self.assertIn("E001", episodes)
        self.assertIn("p000,p002", episodes)
        self.assertIn("mixed", purity)
        self.assertIn("NTO Purity Diagnostics", report)
        self.assertIn("Manifold Episodes", report)

    def test_orthonormality_validation(self):
        s = np.eye(2)
        ok = validate_orbital_orthonormality(np.eye(2), s, "p000", Path("x"), "test")
        self.assertTrue(ok.passed)
        bad = validate_orbital_orthonormality(np.array([[1.0, 1.0], [0.0, 0.0]]), s, "p000", Path("x"), "test")
        self.assertFalse(bad.passed)

    def test_parse_synthetic_cis_file(self):
        with tempfile.TemporaryDirectory() as td:
            cis = Path(td) / "job.cis"
            out = Path(td) / "job.out"
            _write_synthetic_cis(cis)
            out.write_text(
                """
STATE  1:  E=   0.100000 au      2.721 eV     1.0 cm**-1
     2a -> 4a  :     0.160000 (c=  0.40000000)
     0b -> 2b  :     0.360000 (c=  0.60000000)

STATE  2:  E=   0.200000 au      5.442 eV     2.0 cm**-1
     1a -> 3a  :     0.010000 (c= -0.10000000)
"""
            )
            header, amps = parse_cis_amplitudes(cis, roots=[1, 2])
            checks_header, checked_amps, checks = validate_cis_against_output(cis, out, "p000", [1, 2])
        self.assertEqual(header.vector_length, 6)
        self.assertEqual(header.alpha_size, 4)
        self.assertEqual(checks_header.raw_ints, header.raw_ints)
        self.assertEqual(sorted(amps), [1, 2])
        self.assertEqual(amps[1]["orbital_index_base"], 0)
        self.assertTrue(np.isclose(coefficient_from_amplitudes(header, amps[1], 2, "a", 4, "a"), 0.4))
        self.assertTrue(np.isclose(coefficient_from_amplitudes(header, checked_amps[1], 0, "b", 2, "b"), 0.6))
        self.assertTrue(all(row.passed for row in checks))

    def test_self_nto_svd_reconstructs_transition_density(self):
        state = {
            "root": 1,
            "alpha": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "alpha_occ_range": (0, 1),
            "alpha_virt_range": (2, 3),
            "source_file": Path("job.cis"),
        }
        mo = {"alpha": np.eye(4)}
        vec, weights, recon = derive_self_nto_state("p000", 0, 1, state, mo, weight_min=0.0, cumulative=1.0)
        self.assertEqual(len(vec.pairs), 2)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(recon), 1)
        self.assertTrue(recon[0].passed)
        self.assertLess(recon[0].relative_error, 1.0e-12)
        self.assertTrue(all(weights[i].weight >= weights[i + 1].weight for i in range(len(weights) - 1)))

    def test_self_nto_uses_zero_based_cis_active_mo_ranges(self):
        state = {
            "root": 1,
            "alpha": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "alpha_occ_range": (1, 2),
            "alpha_virt_range": (3, 4),
            "orbital_index_base": 0,
            "source_file": Path("job.cis"),
        }
        mo = {"alpha": np.eye(5)}
        vec, weights, recon = derive_self_nto_state("p000", 0, 1, state, mo, weight_min=0.0, cumulative=1.0)
        self.assertTrue(recon[0].passed)
        self.assertEqual(len(vec.pairs), 2)
        self.assertTrue(np.allclose(np.abs(vec.pairs[0].donor_coeff), [0.0, 1.0, 0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(np.abs(vec.pairs[0].acceptor_coeff), [0.0, 0.0, 0.0, 1.0, 0.0]))
        self.assertTrue(weights[0].selected)

    def test_cis_transition_density_overlap_uses_zero_based_ranges(self):
        # A[0, 0] and B[1, 1] both describe MO 1 -> MO 3 only when the
        # inclusive active ranges are interpreted as zero-based.
        state_a = {
            "alpha": np.array([[1.0, 0.0], [0.0, 0.0]]),
            "alpha_occ_range": (1, 2),
            "alpha_virt_range": (3, 4),
            "orbital_index_base": 0,
        }
        state_b = {
            "alpha": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "alpha_occ_range": (0, 1),
            "alpha_virt_range": (2, 3),
            "orbital_index_base": 0,
        }
        mat = cis_transition_density_overlap_matrix(
            {1: state_a},
            {1: state_b},
            [1],
            {"alpha": np.eye(5)},
            {"alpha": np.eye(5)},
            np.eye(5),
        )
        self.assertEqual(mat.shape, (1, 1))
        self.assertTrue(np.isclose(mat[0, 0], 1.0))

    def test_self_nto_selection_is_global_across_spin_blocks(self):
        state = {
            "root": 1,
            "alpha": np.eye(2),
            "beta": 0.01 * np.eye(2),
            "alpha_occ_range": (0, 1),
            "alpha_virt_range": (2, 3),
            "beta_occ_range": (0, 1),
            "beta_virt_range": (2, 3),
            "source_file": Path("job.cis"),
        }
        mo = {"alpha": np.eye(4), "beta": np.eye(4)}
        vec, weights, recon = derive_self_nto_state("p000", 0, 1, state, mo, weight_min=0.01, cumulative=0.995)
        selected = [row for row in weights if row.selected]
        self.assertTrue(all(row.spin == "alpha" for row in selected))
        self.assertEqual(len(vec.pairs), 2)
        self.assertTrue(all(row.passed for row in recon))

    def test_self_nto_diagnostics_cache(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "job.cis"
            source.write_text("synthetic source")
            step = _step("p000", 0, {1: 1.0})
            step.tden_vectors[1] = {
                "root": 1,
                "alpha": np.eye(2),
                "alpha_occ_range": (0, 1),
                "alpha_virt_range": (2, 3),
                "source_file": source,
            }
            step.mo_coefficients = {"alpha": np.eye(4)}
            weights, recon, statuses = derive_self_ntos_from_tden_steps(
                [step],
                [1],
                weight_min=0.01,
                cumulative=0.995,
                cache_dir=Path(td) / "cache",
            )
            self.assertEqual(statuses[0].status, "ok")
            self.assertTrue(weights)
            self.assertTrue(recon)

            cached_step = _step("p000", 0, {1: 1.0})
            cached_step.tden_vectors[1] = dict(step.tden_vectors[1])
            cached_weights, cached_recon, cached_statuses = derive_self_ntos_from_tden_steps(
                [cached_step],
                [1],
                weight_min=0.01,
                cumulative=0.995,
                cache_dir=Path(td) / "cache",
            )
            self.assertEqual(cached_statuses[0].status, "cached")
            self.assertEqual(len(cached_weights), len(weights))
            self.assertEqual(len(cached_recon), len(recon))

    def test_synthetic_crossing_tracking(self):
        roots = [1, 2, 3]
        steps = [_step("p000", 0, {1: 1.0, 2: 2.0, 3: 2.1}), _step("p001", 1, {1: 1.1, 2: 2.0, 3: 1.9})]
        mat = np.array(
            [
                [0.95, 0.05, 0.05],
                [0.05, 0.10, 0.92],
                [0.05, 0.91, 0.12],
            ]
        )

        def provider(i, j):
            self.assertEqual((i, j), (0, 1))
            return mat

        result = run_tracking(steps, roots, provider, TrackingConfig(), "synthetic")
        points = {(p.track_id, p.step_label): p.adiabatic_root for p in result.track_points}
        self.assertEqual(points[(2, "p001")], 3)
        self.assertEqual(points[(3, "p001")], 2)

    def test_failed_track_remains_provisional(self):
        roots = [1, 2]
        steps = [
            _step("p000", 0, {1: 1.0, 2: 2.0}),
            _step("p001", 1, {1: 1.1, 2: 2.1}),
            _step("p002", 2, {1: 1.2, 2: 2.2}),
        ]
        mats = {
            (0, 1): np.array([[0.10, 0.09], [0.95, 0.99]]),
            (1, 2): np.array([[0.99, 0.01], [0.01, 0.99]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        result = run_tracking(steps, roots, provider, TrackingConfig(), "synthetic")
        edge = [e for e in result.assignment_edges if e.track_id == 1 and e.step_b == "p002"][0]
        self.assertEqual(edge.confidence, "ambiguous")
        self.assertIn("earlier provisional assignment", edge.reason)

    def test_adaptive_reference_holds_pre_ambiguity_anchor(self):
        roots = [1, 2]
        steps = [
            _step("p000", 0, {1: 1.0, 2: 2.0}),
            _step("p001", 1, {1: 1.1, 2: 2.1}),
            _step("p002", 2, {1: 1.2, 2: 2.2}),
        ]
        mats = {
            (0, 1): np.array([[0.60, 0.55], [0.05, 0.95]]),
            (1, 2): np.array([[0.05, 0.96], [0.10, 0.95]]),
            (0, 2): np.array([[0.92, 0.10], [0.05, 0.90]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        result = run_tracking(
            steps,
            roots,
            provider,
            TrackingConfig(reference_mode="adaptive"),
            "synthetic",
        )
        edge = [e for e in result.assignment_edges if e.track_id == 1 and e.step_b == "p002"][0]
        self.assertEqual(edge.root_b, 1)
        self.assertEqual(edge.confidence, "reliable")
        self.assertIn("reconnected to the last stable", edge.reason)

    def test_nto_purity_guard_skips_mixed_previous_reference(self):
        roots = [1, 2]
        steps = [
            _step("p000", 0, {1: 1.0, 2: 2.0}),
            _step("p001", 1, {1: 1.1, 2: 2.1}),
            _step("p002", 2, {1: 1.2, 2: 2.2}),
        ]
        _set_self_nto_purity(steps[0], 1, [0.95, 0.03])
        _set_self_nto_purity(steps[1], 1, [0.55, 0.43])
        _set_self_nto_purity(steps[2], 2, [0.94, 0.04])
        mats = {
            (0, 1): np.array([[0.90, 0.05], [0.05, 0.95]]),
            # If p001/root 1 were used as the reference, track 1 would stay on root 1.
            (1, 2): np.array([[0.88, 0.20], [0.95, 0.05]]),
            # The last unmixed anchor p000/root 1 clearly reconnects to p002/root 2.
            (0, 2): np.array([[0.10, 0.96], [0.95, 0.05]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        result = run_tracking(
            steps,
            roots,
            provider,
            TrackingConfig(
                reference_mode="previous",
                nto_purity_guard=True,
                nto_purity_threshold=0.80,
                ambiguity_rescue=True,
            ),
            "synthetic",
        )
        p001_edge = [e for e in result.assignment_edges if e.track_id == 1 and e.step_b == "p001"][0]
        p002_edge = [e for e in result.assignment_edges if e.track_id == 1 and e.step_b == "p002"][0]
        p002_point = [p for p in result.track_points if p.track_id == 1 and p.step_label == "p002"][0]
        self.assertEqual(p001_edge.confidence, "ambiguous")
        self.assertIn("NTO-mixed", p001_edge.reason)
        self.assertEqual(p002_edge.root_b, 2)
        self.assertTrue(np.isclose(p002_edge.similarity_to_previous, 0.20))
        self.assertGreater(p002_edge.similarity_to_anchor, 0.9)
        self.assertTrue(np.isclose(p002_point.similarity_to_previous, 0.20))
        self.assertGreater(p002_point.similarity_to_anchor, 0.9)

    def test_reversed_matrix_provider_transposes_forward_matrices(self):
        mats = {
            (0, 1): np.array([[1.0, 0.2], [0.3, 0.9]]),
            (1, 2): np.array([[0.8, 0.4], [0.1, 0.7]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        rev = make_reversed_matrix_provider(provider, 3)
        self.assertTrue(np.allclose(rev(0, 1), mats[(1, 2)].T))
        self.assertTrue(np.allclose(rev(1, 2), mats[(0, 1)].T))

    def test_reverse_tracking_reports_right_anchored_local_segment(self):
        roots = [1, 2]
        steps = [
            _step("p000", 0, {1: 1.0, 2: 2.0}),
            _step("p001", 1, {1: 1.1, 2: 2.1}),
            _step("p002", 2, {1: 1.2, 2: 2.2}),
            _step("p003", 3, {1: 1.3, 2: 2.3}),
        ]
        for step in steps:
            _set_self_nto_purity(step, 1, [0.95, 0.03])
            _set_self_nto_purity(step, 2, [0.95, 0.03])
        mats = {
            (0, 1): np.array([[0.10, 0.08], [0.04, 0.96]]),
            (1, 2): np.array([[0.96, 0.04], [0.05, 0.95]]),
            (2, 3): np.array([[0.97, 0.03], [0.04, 0.96]]),
            (0, 2): np.array([[0.10, 0.05], [0.02, 0.90]]),
            (0, 3): np.array([[0.09, 0.04], [0.02, 0.88]]),
            (1, 3): np.array([[0.95, 0.04], [0.04, 0.94]]),
        }

        def provider(i, j):
            return mats[(i, j)]

        config = TrackingConfig(reference_mode="previous", ambiguity_rescue=True)
        forward = run_tracking(steps, roots, provider, config, "synthetic")
        reverse = run_tracking(list(reversed(steps)), roots, make_reversed_matrix_provider(provider, len(steps)), config, "synthetic-reverse")
        segments = build_track_segments(
            forward,
            reverse,
            steps,
            min_purity=0.90,
            min_adjacent_similarity=0.85,
            min_length=2,
        )
        self.assertTrue(any(seg.direction == "reverse" and "p002:r1" in seg.root_sequence for seg in segments))
        self.assertTrue(any(seg.support == "two_sided" for seg in segments))


def _step(label, order, energies):
    states = {
        root: StateRecord(label, order, float(order), root, exc_ev=ev, abs_energy_eh=-100.0 + ev / 27.211386245988)
        for root, ev in energies.items()
    }
    job = JobRecord(
        order=order,
        label=label,
        scan_step=float(order),
        step_dir=Path("."),
        out_path=Path("job.out"),
        geom_path=Path("geom.xyz"),
        gbw_path=None,
        uno_path=None,
        nto_pattern="job.s{state}.nto",
    )
    return StepData(job=job, states=states)


def _set_self_nto_purity(step, root, weights):
    vec = NTOStateVector(step.job.label, step.job.order, root, Path("job.cis"), "self-nto-from-cis")
    for idx, weight in enumerate(weights, start=1):
        vec.pairs.append(
            NTOOrbitalPair(
                spin="alpha",
                donor_index=idx,
                acceptor_index=idx,
                weight=float(weight),
                donor_coeff=np.array([1.0]),
                acceptor_coeff=np.array([1.0]),
                source_file=Path("job.cis"),
            )
        )
    step.self_nto_vectors[root] = vec


def _write_synthetic_cis(path: Path) -> None:
    header = (2, 1, 2, 3, 4, 0, 0, 1, 2, 6, 0, 3, 0, 0, 0)
    root1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    root2 = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    doubles = [0.1, 0.0] + root1 + [0.0, 0.0, 0.0, 0.2, 0.0] + root2
    with Path(path).open("wb") as f:
        f.write(struct.pack("<15i", *header))
        f.write(np.asarray(doubles, dtype="<f8").tobytes())


if __name__ == "__main__":
    unittest.main()

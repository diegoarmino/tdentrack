import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

from excited_state_diabatizer.state_tracking import (
    ElectronicSnapshot,
    RootOverlapBlock,
    SelectionConfig,
    SelectionStatus,
    StateSelector,
    StateSurvey,
    SubspaceContinuity,
    TrackingSession,
    analyze_subspace_continuity,
    normalize_signed_overlap_block,
    select_state,
)
from excited_state_diabatizer.models import JobRecord, StateRecord, StepData
from excited_state_diabatizer.tracking import TrackingConfig


def _snapshot(
    label,
    roots=(1, 2),
    *,
    selected_root=None,
    requested_roots=(),
    energies=None,
    excitations=None,
    multiplicities=None,
    x=0.0,
):
    return ElectronicSnapshot(
        label=label,
        coordinates=np.array([[x, 0.0, 0.0]]),
        roots=roots,
        selected_root=selected_root,
        requested_roots=requested_roots,
        energies_eh={} if energies is None else energies,
        excitation_energies_ev={} if excitations is None else excitations,
        multiplicities={} if multiplicities is None else multiplicities,
    )


class TransactionalStateTrackingTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_owns_coordinate_copy(self):
        coordinates = np.array([[0.0, 0.0, 0.0]])
        snapshot = ElectronicSnapshot(
            "reference",
            coordinates,
            (1,),
            selected_root=1,
            energies_eh={1: -100.0},
        )
        coordinates[0, 0] = 9.0
        self.assertEqual(snapshot.coordinates[0, 0], 0.0)
        with self.assertRaises(ValueError):
            snapshot.coordinates[0, 0] = 1.0
        with self.assertRaises(ValueError):
            snapshot.coordinates.setflags(write=True)
        with self.assertRaises(TypeError):
            snapshot.energies_eh[1] = -99.0
        with self.assertRaises(FrozenInstanceError):
            snapshot.selected_root = 2

    def test_normalizes_raw_overlap_before_selecting(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot("r1", (1, 2), requested_roots=(1, 2))
        survey = StateSurvey(
            "s1",
            0,
            "r0",
            candidate,
            overlaps={1: -1.8, 2: 0.4},
            reference_norm=4.0,
            candidate_norms={1: 1.0, 2: 1.0},
        )
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.ACCEPT)
        self.assertEqual(decision.selected_root, 1)
        self.assertAlmostEqual(decision.best_score, 0.9)
        self.assertAlmostEqual(decision.normalized_scores[2], 0.2)
        self.assertAlmostEqual(decision.signed_normalized_scores[1], -0.9)
        self.assertAlmostEqual(decision.signed_normalized_scores[2], 0.2)

    def test_signed_overlap_block_is_normalized_without_erasing_phase(self):
        normalized = normalize_signed_overlap_block(
            np.array([[-1.8, 0.4], [0.6, -1.6]]),
            reference_norms=np.array([4.0, 1.0]),
            candidate_norms=np.array([1.0, 4.0]),
        )
        np.testing.assert_allclose(normalized, [[-0.9, 0.1], [0.6, -0.8]])
        with self.assertRaises(ValueError):
            normalized[0, 0] = 0.0
        with self.assertRaises(ValueError):
            normalized.setflags(write=True)

    def test_rotating_and_swapping_roots_preserve_the_manifold(self):
        angle = 0.73
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        rotated = analyze_subspace_continuity(rotation, (4, 5), (7, 8))
        self.assertIsInstance(rotated, SubspaceContinuity)
        np.testing.assert_allclose(rotated.normalized_signed_overlaps, rotation)
        np.testing.assert_allclose(rotated.singular_values, [1.0, 1.0], atol=1.0e-12)
        np.testing.assert_allclose(rotated.principal_angles_rad, [0.0, 0.0], atol=1.0e-8)
        self.assertAlmostEqual(rotated.minimum_singular_value, 1.0)
        self.assertAlmostEqual(rotated.maximum_principal_angle_deg, 0.0, places=6)

        signed_swap = np.array([[0.0, -1.0], [1.0, 0.0]])
        swapped = analyze_subspace_continuity(signed_swap, (4, 5), (8, 7))
        np.testing.assert_array_equal(swapped.normalized_signed_overlaps, signed_swap)
        np.testing.assert_allclose(swapped.singular_values, [1.0, 1.0])

    def test_nonorthogonal_root_descriptors_can_be_gram_whitened(self):
        gram = np.array([[1.0, 0.5], [0.5, 1.0]])
        continuity = analyze_subspace_continuity(
            gram,
            (1, 2),
            (3, 4),
            reference_gram=gram,
            candidate_gram=gram,
        )
        self.assertTrue(continuity.used_reference_gram)
        self.assertTrue(continuity.used_candidate_gram)
        np.testing.assert_allclose(continuity.singular_values, [1.0, 1.0], atol=1.0e-12)

    def test_full_root_overlap_block_analyzes_an_immutable_subset(self):
        angle = 0.61
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        overlaps = np.block(
            [
                [rotation, np.zeros((2, 2))],
                [np.zeros((2, 2)), np.diag([1.0, -1.0])],
            ]
        )
        block = RootOverlapBlock(
            (1, 2, 3, 4),
            (7, 8, 9, 10),
            overlaps,
            np.eye(4),
            np.eye(4),
        )
        continuity = block.analyze((1, 2), (7, 8))
        self.assertEqual(continuity.reference_roots, (1, 2))
        self.assertEqual(continuity.candidate_roots, (7, 8))
        np.testing.assert_allclose(continuity.singular_values, [1.0, 1.0])
        np.testing.assert_allclose(continuity.normalized_signed_overlaps, rotation)
        with self.assertRaises(ValueError):
            block.overlaps.setflags(write=True)

    def test_incomplete_root_window_requests_retry(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot("r1", (1, 2), requested_roots=(1, 2, 3))
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.95, 2: 0.1})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertEqual(decision.missing_roots, (3,))
        self.assertIn("incomplete", decision.reason)

    def test_missing_score_for_a_parsed_root_requests_retry(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot("r1", (1, 2))
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.95})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertEqual(decision.missing_roots, (2,))

    def test_close_overlap_and_energy_is_reported_as_manifold(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot(
            "r1",
            (1, 2, 3),
            requested_roots=(1, 2, 3),
            excitations={1: 2.000, 2: 2.035, 3: 3.0},
        )
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.82, 2: 0.78, 3: 0.1})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.MANIFOLD)
        self.assertIsNone(decision.selected_root)
        self.assertEqual(decision.manifold_roots, (1, 2))
        self.assertAlmostEqual(decision.competitor_gap_ev, 0.035)

    def test_rotating_manifold_passes_subspace_gate_without_selecting_a_root(self):
        reference = _snapshot("r0", (1, 2), selected_root=1)
        candidate = _snapshot(
            "r1",
            (7, 8),
            requested_roots=(7, 8),
            excitations={7: 2.000, 8: 2.015},
        )
        angle = np.pi / 4.0
        continuity = analyze_subspace_continuity(
            np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            ),
            (1, 2),
            (7, 8),
        )
        survey = StateSurvey(
            "s1",
            0,
            "r0",
            candidate,
            {7: np.cos(angle), 8: -np.sin(angle)},
            subspace_continuity=continuity,
        )
        decision = select_state(
            reference,
            survey,
            SelectionConfig(min_subspace_singular_value=0.95),
        )
        self.assertEqual(decision.status, SelectionStatus.MANIFOLD)
        self.assertIsNone(decision.selected_root)
        self.assertTrue(decision.subspace_continuous)
        self.assertIs(decision.subspace_continuity, continuity)
        self.assertAlmostEqual(decision.signed_normalized_scores[8], -np.sin(angle))

        session = TrackingSession(
            reference,
            StateSelector(SelectionConfig(min_subspace_singular_value=0.95)),
        )
        pending = session.survey(
            candidate,
            {7: np.cos(angle), 8: -np.sin(angle)},
            subspace_continuity=continuity,
        )
        with self.assertRaises(ValueError):
            session.commit(session.select(pending.survey_id))

    def test_weak_subspace_direction_requests_another_probe(self):
        reference = _snapshot("r0", (1, 2), selected_root=1)
        candidate = _snapshot(
            "r1",
            (7, 8),
            requested_roots=(7, 8),
            excitations={7: 2.000, 8: 2.015},
        )
        continuity = analyze_subspace_continuity(
            np.diag([0.95, 0.50]),
            (1, 2),
            (7, 8),
        )
        survey = StateSurvey(
            "s1",
            0,
            "r0",
            candidate,
            {7: 0.72, 8: 0.70},
            subspace_continuity=continuity,
        )
        decision = select_state(
            reference,
            survey,
            SelectionConfig(min_subspace_singular_value=0.80),
        )
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertFalse(decision.subspace_continuous)
        self.assertIn("weakest subspace singular value", decision.reason)

    def test_selector_extracts_rotating_manifold_from_full_four_root_block(self):
        reference = _snapshot(
            "r0",
            (1, 2, 3, 4),
            selected_root=1,
            excitations={1: 2.000, 2: 2.020, 3: 3.0, 4: 4.0},
            multiplicities={1: 3, 2: 3, 3: 3, 4: 3},
        )
        candidate = _snapshot(
            "r1",
            (7, 8, 9, 10),
            requested_roots=(7, 8, 9, 10),
            excitations={7: 2.010, 8: 2.025, 9: 3.1, 10: 4.1},
            multiplicities={7: 3, 8: 3, 9: 3, 10: 3},
        )
        angle = np.pi / 4.0
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        full_overlaps = np.block(
            [
                [rotation, np.zeros((2, 2))],
                [np.zeros((2, 2)), np.eye(2)],
            ]
        )
        full_block = RootOverlapBlock(
            (1, 2, 3, 4),
            (7, 8, 9, 10),
            full_overlaps,
            np.eye(4),
            np.eye(4),
        )
        survey = StateSurvey(
            "s1",
            0,
            "r0",
            candidate,
            {7: np.cos(angle), 8: -np.sin(angle), 9: 0.05, 10: 0.02},
            root_overlap_block=full_block,
        )
        decision = select_state(
            reference,
            survey,
            SelectionConfig(min_subspace_singular_value=0.95),
        )
        self.assertEqual(decision.status, SelectionStatus.MANIFOLD)
        self.assertEqual(decision.manifold_roots, (7, 8))
        self.assertIsNone(decision.selected_root)
        self.assertTrue(decision.subspace_continuous)
        self.assertEqual(decision.subspace_continuity.reference_roots, (1, 2))
        self.assertEqual(decision.subspace_continuity.candidate_roots, (7, 8))
        np.testing.assert_allclose(decision.subspace_continuity.singular_values, [1.0, 1.0])

    def test_full_overlap_block_rejects_split_and_merge_dimensions(self):
        candidate_roots = (7, 8, 9, 10)
        multiplicities_ref = {root: 3 for root in (1, 2, 3, 4)}
        multiplicities_cur = {root: 3 for root in candidate_roots}

        with self.subTest("reference manifold splits from three roots to two"):
            reference = _snapshot(
                "r0",
                (1, 2, 3, 4),
                selected_root=1,
                excitations={1: 2.000, 2: 2.020, 3: 2.040, 4: 4.0},
                multiplicities=multiplicities_ref,
            )
            candidate = _snapshot(
                "r1",
                candidate_roots,
                requested_roots=candidate_roots,
                excitations={7: 2.010, 8: 2.025, 9: 3.0, 10: 4.0},
                multiplicities=multiplicities_cur,
            )
            angle = np.pi / 4.0
            block = RootOverlapBlock(
                (1, 2, 3, 4),
                candidate_roots,
                np.block(
                    [
                        [
                            np.array(
                                [
                                    [np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)],
                                ]
                            ),
                            np.zeros((2, 2)),
                        ],
                        [np.zeros((2, 2)), np.eye(2)],
                    ]
                ),
                np.eye(4),
                np.eye(4),
            )
            survey = StateSurvey(
                "split",
                0,
                "r0",
                candidate,
                {7: np.cos(angle), 8: -np.sin(angle), 9: 0.05, 10: 0.01},
                root_overlap_block=block,
            )
            decision = select_state(reference, survey)
            self.assertEqual(decision.status, SelectionStatus.RETRY)
            self.assertIn("split/merge", decision.reason)

        with self.subTest("two reference roots merge into three candidate roots"):
            reference = _snapshot(
                "r0",
                (1, 2, 3, 4),
                selected_root=1,
                excitations={1: 2.000, 2: 2.020, 3: 3.0, 4: 4.0},
                multiplicities=multiplicities_ref,
            )
            candidate = _snapshot(
                "r1",
                candidate_roots,
                requested_roots=candidate_roots,
                excitations={7: 2.010, 8: 2.025, 9: 2.040, 10: 4.0},
                multiplicities=multiplicities_cur,
            )
            orthogonal = np.array(
                [
                    [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 0.0],
                    [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
                    [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            block = RootOverlapBlock(
                (1, 2, 3, 4),
                candidate_roots,
                orthogonal,
                np.eye(4),
                np.eye(4),
            )
            score = 1.0 / np.sqrt(3.0)
            survey = StateSurvey(
                "merge",
                0,
                "r0",
                candidate,
                {7: score, 8: score, 9: score, 10: 0.0},
                root_overlap_block=block,
            )
            decision = select_state(
                reference,
                survey,
                SelectionConfig(min_score=0.50),
            )
            self.assertEqual(decision.status, SelectionStatus.RETRY)
            self.assertIn("split/merge", decision.reason)

    def test_unknown_reference_manifold_energy_requests_retry(self):
        reference = _snapshot(
            "r0",
            (1, 2),
            selected_root=1,
            excitations={1: 2.0},
            multiplicities={1: 3, 2: 3},
        )
        candidate = _snapshot(
            "r1",
            (7, 8),
            requested_roots=(7, 8),
            excitations={7: 2.01, 8: 2.02},
            multiplicities={7: 3, 8: 3},
        )
        angle = np.pi / 4.0
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        survey = StateSurvey(
            "unknown-energy",
            0,
            "r0",
            candidate,
            {7: np.cos(angle), 8: -np.sin(angle)},
            root_overlap_block=RootOverlapBlock(
                (1, 2),
                (7, 8),
                rotation,
                np.eye(2),
                np.eye(2),
            ),
        )
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertIn("energies are unavailable", decision.reason)

    def test_ambiguous_but_energy_separated_roots_request_another_probe(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot(
            "r1",
            (1, 2),
            requested_roots=(1, 2),
            excitations={1: 2.0, 2: 2.5},
        )
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.82, 2: 0.78})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertIn("not a near-degenerate manifold", decision.reason)

    def test_energy_gate_is_applied_when_energies_are_available(self):
        reference = _snapshot("r0", (1,), selected_root=1, energies={1: -100.0})
        candidate = _snapshot(
            "r1",
            (1, 2),
            requested_roots=(1, 2),
            energies={1: -99.99, 2: -99.98},
        )
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.95, 2: 0.1})
        decision = select_state(
            reference,
            survey,
            SelectionConfig(max_energy_increase_eh=0.0),
        )
        self.assertEqual(decision.status, SelectionStatus.RETRY)
        self.assertAlmostEqual(decision.energy_change_eh, 0.01)

    def test_nonpositive_norm_is_a_hard_halt(self):
        reference = _snapshot("r0", (1,), selected_root=1)
        candidate = _snapshot("r1", (1,))
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.9}, candidate_norms={1: 0.0})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.HALT)
        self.assertIn("must be positive", decision.reason)

    def test_multiplicity_filter_does_not_select_an_incompatible_root(self):
        reference = _snapshot("r0", (1,), selected_root=1, multiplicities={1: 3})
        candidate = _snapshot(
            "r1",
            (1, 2),
            requested_roots=(1, 2),
            multiplicities={1: 1, 2: 3},
        )
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.99, 2: 0.85})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.ACCEPT)
        self.assertEqual(decision.selected_root, 2)

    def test_known_reference_multiplicity_rejects_missing_candidate_metadata(self):
        reference = _snapshot("r0", (1,), selected_root=1, multiplicities={1: 3})
        candidate = _snapshot(
            "r1",
            (1, 2),
            requested_roots=(1, 2),
            multiplicities={2: 3},
        )
        survey = StateSurvey("s1", 0, "r0", candidate, {1: 0.99, 2: 0.85})
        decision = select_state(reference, survey)
        self.assertEqual(decision.status, SelectionStatus.ACCEPT)
        self.assertEqual(decision.selected_root, 2)

    def test_session_surveys_are_noncommitting_and_lowest_energy_probe_wins(self):
        initial = _snapshot("r0", (1,), selected_root=1, energies={1: -100.0})
        session = TrackingSession(initial)
        short = _snapshot(
            "short",
            (1, 2),
            requested_roots=(1, 2),
            energies={1: -100.01, 2: -99.0},
            x=0.5,
        )
        long = _snapshot(
            "long",
            (1, 2),
            requested_roots=(1, 2),
            energies={1: -100.03, 2: -99.0},
            x=1.5,
        )
        short_survey = session.survey(short, {1: 0.98, 2: 0.1}, step_scale=0.5)
        long_survey = session.survey(long, {1: 0.90, 2: 0.1}, step_scale=1.5)

        self.assertIs(session.committed, initial)
        self.assertEqual(session.generation, 0)
        self.assertEqual(len(session.pending), 2)
        decision = session.select()
        self.assertEqual(decision.survey_id, long_survey.survey_id)

        committed = session.commit(decision)
        self.assertEqual(committed.label, "long")
        self.assertEqual(committed.selected_root, 1)
        self.assertIs(session.anchor, committed)
        self.assertEqual(session.generation, 1)
        self.assertEqual(len(session.pending), 0)
        self.assertEqual([item.label for item in session.history], ["r0", "long"])
        self.assertNotEqual(short_survey.survey_id, long_survey.survey_id)

    def test_discard_and_nonaccepting_commit_leave_reference_untouched(self):
        initial = _snapshot("r0", (1,), selected_root=1)
        session = TrackingSession(initial)
        candidate = _snapshot(
            "mixed",
            (1, 2),
            requested_roots=(1, 2),
            excitations={1: 2.0, 2: 2.01},
        )
        survey = session.survey(candidate, {1: 0.80, 2: 0.79})
        decision = session.select(survey.survey_id)
        self.assertEqual(decision.status, SelectionStatus.MANIFOLD)
        with self.assertRaises(ValueError):
            session.commit(decision)
        discarded = session.discard(survey.survey_id)
        self.assertIs(discarded, survey)
        self.assertIs(session.committed, initial)
        self.assertEqual(session.history, (initial,))

    def test_selector_configuration_can_be_shared_without_state(self):
        selector = StateSelector(SelectionConfig(min_score=0.8))
        reference = _snapshot("r0", (1,), selected_root=1)
        weak = _snapshot("weak", (1,))
        strong = _snapshot("strong", (1,))
        weak_survey = StateSurvey("weak", 0, "r0", weak, {1: 0.75})
        strong_survey = StateSurvey("strong", 0, "r0", strong, {1: 0.85})
        self.assertEqual(selector(reference, weak_survey).status, SelectionStatus.RETRY)
        self.assertEqual(selector(reference, strong_survey).status, SelectionStatus.ACCEPT)

    def test_existing_scan_models_have_additive_adapters(self):
        state = StateRecord(
            step_label="p000",
            step_order=0,
            scan_step=0.0,
            root=2,
            exc_ev=2.1,
            s2=2.02,
            multiplicity=3,
            abs_energy_eh=-99.9,
        )
        job = JobRecord(
            order=0,
            label="p000",
            scan_step=0.0,
            step_dir=Path("."),
            out_path=Path("job.out"),
            geom_path=Path("geom.xyz"),
            gbw_path=Path("job.gbw"),
            uno_path=None,
            nto_pattern="job.s{state}.nto",
        )
        snapshot = ElectronicSnapshot.from_step_data(
            StepData(job, {2: state}),
            np.zeros((1, 3)),
            selected_root=2,
            requested_roots=(1, 2),
        )
        self.assertEqual(snapshot.selected_root, 2)
        self.assertEqual(snapshot.missing_roots, (1,))
        self.assertEqual(snapshot.multiplicities[2], 3)
        self.assertEqual(snapshot.artifacts["gbw"], Path("job.gbw"))

        config = SelectionConfig.from_tracking_config(TrackingConfig())
        self.assertEqual(config.min_score, TrackingConfig().similarity_threshold_good)
        self.assertEqual(config.min_margin, TrackingConfig().assignment_margin_threshold)


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from excited_state_diabatizer.models import JobRecord, MissingDataError
from excited_state_diabatizer.orca_auxiliary import (
    _extract_basis_directives,
    _load_cross_block,
    orca_auxiliary_cross_overlap,
)


class OrcaAuxiliaryTests(unittest.TestCase):
    def test_cross_block_requires_diagonal_and_transpose_symmetry_validation(self):
        with tempfile.TemporaryDirectory() as td:
            json_file = Path(td) / "cross.json"
            statuses = []
            _write_overlap_json(json_file, [[1.0, 0.25], [0.25, 1.0]])
            cross = _load_cross_block(json_file, np.eye(1), np.eye(1), "a", "b", statuses)
            self.assertTrue(np.allclose(cross, [[0.25]]))
            self.assertEqual(statuses[-1].status, "ok")

            _write_overlap_json(json_file, [[1.0, 0.25], [0.35, 1.0]])
            statuses = []
            with self.assertRaisesRegex(MissingDataError, "transpose-symmetry"):
                _load_cross_block(json_file, np.eye(1), np.eye(1), "a", "b", statuses)
            self.assertEqual(statuses[-1].status, "failed")

            _write_overlap_json(json_file, [[0.9, 0.25], [0.25, 1.0]])
            statuses = []
            with self.assertRaisesRegex(MissingDataError, "A block error"):
                _load_cross_block(json_file, np.eye(1), np.eye(1), "a", "b", statuses)
            self.assertEqual(statuses[-1].status, "failed")

    def test_basis_extraction_reads_all_simple_lines_and_nested_percent_block(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "job.inp"
            inp.write_text(
                """! B3LYP TightSCF
! D4 STO-3G # basis occurs on a later simple-input line
%basis
  NewGTO H
    S 1
      1 1.0 1.0
  end
  AddGTO H
    P 1
      1 0.5 1.0
  end
end
* xyz 0 1
H 0 0 0
*
"""
            )
            tokens, block = _extract_basis_directives(inp)
            self.assertEqual(tokens, ["STO-3G"])
            self.assertIn("NewGTO H", block)
            self.assertIn("AddGTO H", block)
            self.assertEqual(block.casefold().count("end"), 3)
            self.assertTrue(block.rstrip().endswith("end"))
            self.assertNotIn("* xyz", block)

    def test_basis_can_be_defined_entirely_in_one_line_percent_block(self):
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "job.inp"
            inp.write_text('%basis Basis "def2-SVP" end\n* xyz 0 1\nH 0 0 0\n*\n')
            tokens, block = _extract_basis_directives(inp)
            self.assertEqual(tokens, [])
            self.assertEqual(block, '%basis Basis "def2-SVP" end')

    def test_cache_identity_changes_with_geometry_and_basis(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_a = _make_job(root, "a", "same/a", 0.0)
            job_b = _make_job(root, "b", "same/b", 0.7)
            outdir = root / "results"
            calls = []

            def fake_run(command, cwd, **kwargs):
                calls.append(tuple(command))
                cwd = Path(cwd)
                if command[0] == "orca":
                    (cwd / "cross.gbw").write_bytes(b"synthetic")
                    return subprocess.CompletedProcess(command, 0, "ORCA TERMINATED NORMALLY\n", "")
                _write_overlap_json(cwd / "cross.json", [[1.0, 0.25], [0.25, 1.0]])
                return subprocess.CompletedProcess(command, 0, "exported\n", "")

            with mock.patch("excited_state_diabatizer.orca_auxiliary.subprocess.run", side_effect=fake_run):
                first = orca_auxiliary_cross_overlap(
                    job_a, job_b, np.eye(1), np.eye(1), outdir, "orca", "orca_2json", []
                )
                cached = orca_auxiliary_cross_overlap(
                    job_a, job_b, np.eye(1), np.eye(1), outdir, "orca", "orca_2json", []
                )
                self.assertEqual(len(calls), 2)

                job_b.geom_path.write_text("1\nchanged geometry\nH 0.0 0.0 0.8\n")
                orca_auxiliary_cross_overlap(
                    job_a, job_b, np.eye(1), np.eye(1), outdir, "orca", "orca_2json", []
                )
                self.assertEqual(len(calls), 4)

                for job in (job_a, job_b):
                    (job.step_dir / "job.inp").write_text("! HF def2-TZVP\n* xyz 0 1\nH 0 0 0\n*\n")
                orca_auxiliary_cross_overlap(
                    job_a, job_b, np.eye(1), np.eye(1), outdir, "orca", "orca_2json", []
                )

            self.assertTrue(np.allclose(first, [[0.25]]))
            self.assertTrue(np.allclose(cached, first))
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(list((outdir / "json_cache").glob("cross_*.json"))), 3)

    def test_gbw_without_normal_orca_termination_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_a = _make_job(root, "a", "a", 0.0)
            job_b = _make_job(root, "b", "b", 0.7)
            statuses = []

            def fake_run(command, cwd, **kwargs):
                (Path(cwd) / "cross.gbw").write_bytes(b"stale-looking product")
                return subprocess.CompletedProcess(command, 0, "SCF finished\n", "")

            with mock.patch("excited_state_diabatizer.orca_auxiliary.subprocess.run", side_effect=fake_run) as run:
                with self.assertRaisesRegex(MissingDataError, "TERMINATED NORMALLY"):
                    orca_auxiliary_cross_overlap(
                        job_a, job_b, np.eye(1), np.eye(1), root / "results", "orca", "orca_2json", statuses
                    )
            self.assertEqual(run.call_count, 1)
            self.assertEqual(statuses[-1].status, "failed")

    def test_nonzero_orca_return_code_is_rejected_even_with_termination_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_a = _make_job(root, "a", "a", 0.0)
            job_b = _make_job(root, "b", "b", 0.7)

            def fake_run(command, cwd, **kwargs):
                (Path(cwd) / "cross.gbw").write_bytes(b"product")
                return subprocess.CompletedProcess(command, 7, "ORCA TERMINATED NORMALLY\n", "fatal error")

            with mock.patch("excited_state_diabatizer.orca_auxiliary.subprocess.run", side_effect=fake_run) as run:
                with self.assertRaisesRegex(MissingDataError, "return code 7"):
                    orca_auxiliary_cross_overlap(
                        job_a, job_b, np.eye(1), np.eye(1), root / "results", "orca", "orca_2json", []
                    )
            self.assertEqual(run.call_count, 1)

    def test_different_basis_specifications_are_rejected_before_orca(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job_a = _make_job(root, "a", "a", 0.0, basis="def2-SVP")
            job_b = _make_job(root, "b", "b", 0.7, basis="def2-TZVP")
            with mock.patch("excited_state_diabatizer.orca_auxiliary.subprocess.run") as run:
                with self.assertRaisesRegex(MissingDataError, "identical basis specifications"):
                    orca_auxiliary_cross_overlap(
                        job_a, job_b, np.eye(1), np.eye(1), root / "results", "orca", "orca_2json", []
                    )
            run.assert_not_called()


def _write_overlap_json(path: Path, overlap) -> None:
    path.write_text(json.dumps({"Molecule": {"Atoms": [], "S-Matrix": overlap}}))


def _make_job(root: Path, dirname: str, label: str, z: float, basis: str = "def2-SVP") -> JobRecord:
    step_dir = root / dirname
    step_dir.mkdir()
    geom = step_dir / "geom.xyz"
    geom.write_text(f"1\ngeometry {label}\nH 0.0 0.0 {z}\n")
    inp = step_dir / "job.inp"
    inp.write_text(f"! HF {basis}\n* xyz 0 1\nH 0 0 {z}\n*\n")
    out = step_dir / "job.out"
    out.write_text("")
    return JobRecord(
        order=0,
        label=label,
        scan_step=None,
        step_dir=step_dir,
        out_path=out,
        geom_path=geom,
        gbw_path=None,
        uno_path=None,
        nto_pattern="",
    )


if __name__ == "__main__":
    unittest.main()

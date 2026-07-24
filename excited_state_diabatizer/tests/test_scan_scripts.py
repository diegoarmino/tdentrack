import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("rerun_orca_tddft_sequential.py")
EXTENDER = load_script("orca_gs_scan_then_tddft_sequential.py")
RENAMER = load_script("rename_scan_steps_to_coordinate.py")


def write_xyz(path, distance):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "2\n"
        f"distance={distance}\n"
        "Ru 0.0 0.0 0.0\n"
        f"C  {distance} 0.0 0.0\n"
    )


class SequentialRunnerTests(unittest.TestCase):
    def test_points_are_sorted_by_measured_coordinate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            long = root / "long.xyz"
            short = root / "short.xyz"
            write_xyz(long, 2.0)
            write_xyz(short, 1.8)
            bonds = [RUNNER.Bond(1.0, 0, 1, "Ru-CO")]
            points = RUNNER.build_points(
                [("not-the-order", long), ("still-not-the-order", short)], bonds, 4
            )
            self.assertEqual([point.label for point in points], ["1.8000", "2.0000"])

    def test_sequential_run_uses_gs_seed_then_restarts_tddft(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for label, distance in (("2.0000", 2.0), ("1.8000", 1.8)):
                write_xyz(source / f"tddft_step_{label}" / "geom.xyz", distance)

            jobs = root / "jobs.csv"
            with jobs.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["label", "geom"])
                writer.writeheader()
                writer.writerow({"label": "2.0000", "geom": ""})
                writer.writerow({"label": "1.8000", "geom": ""})

            bonds = root / "bonds.dat"
            bonds.write_text("1.0 1 2 Ru-CO\n")
            gs_job = root / "gs" / "gs_scan_point_1.8000"
            write_xyz(gs_job / "optimized.xyz", 1.8)
            (gs_job / "job.gbw").write_text("ground-state gbw")
            (gs_job / "job.out").write_text("****ORCA TERMINATED NORMALLY****\n")
            (gs_job / "job.inp").write_text("* xyzfile 1 1 geom.xyz\n")
            template = root / "template.inp"
            template.write_text(
                "! RKS PBE def2-SVP\n"
                "%tddft\n"
                " nroots 2\n"
                " triplets true\n"
                " irootmult Triplet\n"
                "${tddft_restart}\n"
                "end\n"
                "${guess_block}\n"
                "* xyzfile ${charge} ${mult} ${xyzfile}\n"
            )
            fake_orca = root / "fake_orca"
            fake_orca.write_text(
                "#!/bin/sh\n"
                "printf 'fake gbw' > job.gbw\n"
                "printf '****ORCA TERMINATED NORMALLY****\\n'\n"
            )
            os.chmod(fake_orca, 0o755)
            output = root / "output"

            result = RUNNER.main(
                [
                    "--source-tddft-dir",
                    str(source),
                    "--source-gs-dir",
                    str(root / "gs"),
                    "--jobs-csv",
                    str(jobs),
                    "--bonds-file",
                    str(bonds),
                    "--template",
                    str(template),
                    "--charge",
                    "1",
                    "--orca",
                    str(fake_orca),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            first = output / "tddft_step_1.8000"
            second = output / "tddft_step_2.0000"
            self.assertIn("Guess MORead", (first / "job.inp").read_text())
            self.assertIn('MOInp "previous.gbw"', (first / "job.inp").read_text())
            self.assertNotIn('Restart "previous.gbw"', (first / "job.inp").read_text())
            self.assertEqual((first / "previous.gbw").read_text(), "ground-state gbw")
            self.assertIn("Guess MORead", (second / "job.inp").read_text())
            self.assertIn('MOInp "previous.gbw"', (second / "job.inp").read_text())
            self.assertIn('Restart "previous.gbw"', (second / "job.inp").read_text())
            self.assertEqual((second / "previous.gbw").read_text(), "fake gbw")

    def test_gs_seed_charge_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root / "geom.xyz"
            write_xyz(geometry, 1.8)
            point = RUNNER.ScanPoint("1.8000", 1.8, "1.8000", geometry)
            gs_job = root / "gs" / "gs_scan_point_1.8000"
            write_xyz(gs_job / "optimized.xyz", 1.8)
            (gs_job / "job.gbw").write_text("ground-state gbw")
            (gs_job / "job.out").write_text("****ORCA TERMINATED NORMALLY****\n")
            (gs_job / "job.inp").write_text("* xyzfile 2 1 geom.xyz\n")
            with self.assertRaisesRegex(ValueError, "charge/multiplicity 2/1"):
                RUNNER.find_gs_seed(
                    point,
                    root / "gs",
                    charge=1,
                    geometry_tolerance=1e-5,
                )


class SequentialExtensionTests(unittest.TestCase):
    def test_target_grid_preserves_spacing_and_includes_exact_stop(self):
        targets = EXTENDER.generate_targets(2.055247562115, 2.8, 0.01)
        self.assertEqual(len(targets), 75)
        self.assertAlmostEqual(targets[0], 2.065247562115, places=12)
        self.assertAlmostEqual(targets[-2], 2.795247562115, places=12)
        self.assertEqual(targets[-1], 2.8)

    def test_projection_reaches_requested_bond_distance(self):
        geometry = EXTENDER.GeometryData(
            ("Ru", "C"),
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        )
        bonds = [EXTENDER.Bond(1.0, 0, 1, "Ru-CO")]
        targets = EXTENDER.target_distances(geometry, bonds, 2.2)
        projected = EXTENDER.project_to_targets(geometry, bonds, targets)
        self.assertAlmostEqual(
            EXTENDER.collective_coordinate(projected, bonds), 2.2, places=8
        )

    def test_continuation_tddft_input_restarts_orbitals_and_amplitudes(self):
        template = (
            "! RKS PBE def2-SVP\n"
            "%tddft\n"
            " nroots 2\n"
            " triplets true\n"
            " irootmult Triplet\n"
            "${tddft_restart}\n"
            "end\n"
            "${guess_block}\n"
            "* xyzfile ${charge} ${mult} ${xyzfile}\n"
        )
        rendered = EXTENDER.render_tddft_input(
            template, Path("/tmp/geom.xyz"), 2, 2.1
        )
        self.assertIn('Restart "previous.gbw"', rendered)
        self.assertIn("Guess MORead", rendered)
        self.assertIn('MOInp "previous.gbw"', rendered)
        self.assertIn("* xyzfile 2 1", rendered)

    def test_extension_propagates_separate_gs_and_tddft_chains(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bonds = root / "bonds.dat"
            bonds.write_text("1.0 1 2 Ru-CO\n")

            gs_seed = root / "gs_seed"
            write_xyz(gs_seed / "optimized.xyz", 2.0)
            (gs_seed / "job.gbw").write_text("initial gs")
            (gs_seed / "job.out").write_text(
                "THE OPTIMIZATION HAS CONVERGED\n"
                "FINAL SINGLE POINT ENERGY -10.0\n"
                "****ORCA TERMINATED NORMALLY****\n"
            )
            (gs_seed / "job.inp").write_text("* xyzfile 2 1 geom.xyz\n")

            td_seed = root / "td_seed"
            write_xyz(td_seed / "geom.xyz", 2.0)
            (td_seed / "job.gbw").write_text("initial td")
            (td_seed / "job.cis").write_text("initial amplitudes")
            (td_seed / "job.out").write_text(
                "TD-DFT/TDA EXCITED STATES (TRIPLETS)\n"
                "STATE 1: E= 0.100000 au\n"
                "STATE 2: E= 0.200000 au\n"
                "ABSORPTION SPECTRUM\n"
                "****ORCA TERMINATED NORMALLY****\n"
            )
            (td_seed / "job.inp").write_text("* xyzfile 2 1 geom.xyz\n")

            gs_template = root / "gs_template.inp"
            gs_template.write_text(
                "! UKS PBE def2-SVP Opt\n"
                "${guess_block}\n${constraints_block}\n"
                "* xyzfile ${charge} ${mult} ${xyzfile}\n"
            )
            td_template = root / "td_template.inp"
            td_template.write_text(
                "! RKS PBE def2-SVP\n"
                "%tddft\n nroots 2\n triplets true\n"
                " irootmult Triplet\n${tddft_restart}\nend\n"
                "${guess_block}\n"
                "* xyzfile ${charge} ${mult} ${xyzfile}\n"
            )
            fake_orca = root / "fake_orca"
            fake_orca.write_text(
                "#!/bin/sh\n"
                "if grep -q ' Opt' job.inp; then\n"
                "  cp start_constrained.xyz job.xyz\n"
                "  printf 'new gs' > job.gbw\n"
                "  printf 'THE OPTIMIZATION HAS CONVERGED\\n'\n"
                "  printf 'FINAL SINGLE POINT ENERGY -10.1\\n'\n"
                "else\n"
                "  printf 'new td' > job.gbw\n"
                "  printf 'new amplitudes' > job.cis\n"
                "  printf 'TD-DFT/TDA EXCITED STATES (TRIPLETS)\\n'\n"
                "  printf 'STATE 1: E= 0.100000 au\\n'\n"
                "  printf 'STATE 2: E= 0.200000 au\\n'\n"
                "  printf 'ABSORPTION SPECTRUM\\n'\n"
                "fi\n"
                "printf '****ORCA TERMINATED NORMALLY****\\n'\n"
            )
            os.chmod(fake_orca, 0o755)
            output = root / "extension"

            result = EXTENDER.main(
                [
                    "--seed-gs-job",
                    str(gs_seed),
                    "--seed-tddft-job",
                    str(td_seed),
                    "--bonds-file",
                    str(bonds),
                    "--gs-template",
                    str(gs_template),
                    "--tddft-template",
                    str(td_template),
                    "--gs-charge",
                    "2",
                    "--tddft-charge",
                    "2",
                    "--stop-coordinate",
                    "2.02",
                    "--coordinate-step",
                    "0.01",
                    "--orca",
                    str(fake_orca),
                    "--workdir",
                    str(output),
                ]
            )
            self.assertEqual(result, 0)
            gs_first = output / "gs_scan" / "gs_scan_point_2.0100"
            td_first = (
                output
                / "tddft_rks_triplets_sequential"
                / "tddft_step_2.0100"
            )
            td_second = (
                output
                / "tddft_rks_triplets_sequential"
                / "tddft_step_2.0200"
            )
            self.assertEqual((gs_first / "previous.gbw").read_text(), "initial gs")
            self.assertEqual((td_first / "previous.gbw").read_text(), "initial td")
            self.assertEqual((td_second / "previous.gbw").read_text(), "new td")
            self.assertIn(
                'Restart "previous.gbw"', (td_first / "job.inp").read_text()
            )


class RenameTests(unittest.TestCase):
    def test_apply_renames_paired_directories_by_measured_coordinate(self):
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary) / "scan"
            gs = workdir / "gs_scan" / "gs_scan_point_p000"
            td = workdir / "tddft" / "tddft_step_p000"
            write_xyz(gs / "optimized.xyz", 1.91234)
            write_xyz(td / "geom.xyz", 1.91234)
            bonds = Path(temporary) / "bonds.dat"
            bonds.write_text("1.0 1 2 Ru-CO\n")

            plan = []
            definitions = RENAMER.read_bonds(bonds, zero_based=False)
            plan.extend(
                RENAMER.discover_kind(
                    kind="gs",
                    root=workdir / "gs_scan",
                    prefix="gs_scan_point_",
                    bonds=definitions,
                    decimals=4,
                )
            )
            plan.extend(
                RENAMER.discover_kind(
                    kind="tddft",
                    root=workdir / "tddft",
                    prefix="tddft_step_",
                    bonds=definitions,
                    decimals=4,
                )
            )
            RENAMER.validate_plan(plan, pair_tolerance=1e-8)
            RENAMER.apply_plan(plan)
            self.assertTrue((workdir / "gs_scan" / "gs_scan_point_1.9123").is_dir())
            self.assertTrue((workdir / "tddft" / "tddft_step_1.9123").is_dir())
            self.assertFalse(gs.exists())
            self.assertFalse(td.exists())

    def test_existing_destination_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tddft_step_p000"
            destination = root / "tddft_step_1.9000"
            write_xyz(source / "geom.xyz", 1.9)
            destination.mkdir()
            rename = RENAMER.Rename(
                "tddft",
                "p000",
                1.9,
                source,
                destination,
                source / "geom.xyz",
            )
            with self.assertRaises(FileExistsError):
                RENAMER.validate_plan([rename], pair_tolerance=1e-8)


if __name__ == "__main__":
    unittest.main()

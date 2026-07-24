import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYSISYPHUS = Path("/home/diegoa/dev/pysisyphus")
if PYSISYPHUS.is_dir():
    sys.path.insert(0, str(PYSISYPHUS))


def load_driver():
    path = ROOT / "scripts" / "run_pysis_root4_rks_triplet.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(PYSISYPHUS.is_dir(), "local pysisyphus checkout is unavailable")
class Root4RKSDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()

    def test_blocks_request_full_triplet_tddft_by_default(self):
        blocks = self.driver.build_blocks(15, tda=False).lower()
        self.assertIn("nroots 15", blocks)
        self.assertIn("triplets true", blocks)
        self.assertIn("irootmult triplet", blocks)
        self.assertIn("cpcmeq true", blocks)
        self.assertIn("guessmode cmatrix", blocks)
        self.assertIn("autostart false", blocks)
        self.assertNotIn("tda true", blocks)

    def test_tda_is_an_explicit_opt_in(self):
        self.assertIn(
            "tda true", self.driver.build_blocks(15, tda=True).lower()
        )

    def test_exact_engrad_returns_negative_gradient_as_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.engrad"
            path.write_text(
                "# atoms\n1\n# energy\n-10.25\n"
                "# gradient\n0.1\n-0.2\n0.3\n"
            )
            parsed = self.driver.exact_engrad(path)
            self.assertEqual(parsed["energy"], -10.25)
            self.assertEqual(parsed["forces"].tolist(), [-0.1, 0.2, -0.3])

    def test_complete_seed_artifact_set_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for suffix in ("cis", "bson", "gbw", "out", "engrad"):
                (directory / f"root4_seed_000.000.orca.{suffix}").touch()
            artifacts = self.driver.find_initial_artifacts(directory)
            self.assertEqual(set(artifacts), {"cis", "bson", "gbw", "out", "engrad"})


if __name__ == "__main__":
    unittest.main()

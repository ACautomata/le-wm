from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PackageLayoutTests(unittest.TestCase):
    def test_lightweight_package_modules_import_without_runtime_stack(self):
        import lewm
        import lewm.models
        from lewm.evaluation.pipeline import resolve_results_dir

        self.assertEqual(lewm.__version__, "0.1.0")
        self.assertEqual(lewm.models.__all__, [])
        self.assertTrue(callable(resolve_results_dir))

    def test_torch_backed_modules_import(self):
        from lewm.models.jepa import JEPA
        from lewm.training.forward import lejepa_forward

        self.assertTrue(callable(JEPA))
        self.assertTrue(callable(lejepa_forward))

    def test_random_policy_results_dir_uses_runtime_output_dir(self):
        from lewm.evaluation.pipeline import resolve_results_dir

        runtime_dir = Path("/tmp/hydra-run")
        cache_dir = Path("/tmp/stablewm")

        self.assertEqual(
            resolve_results_dir(policy="random", cache_dir=cache_dir, runtime_output_dir=runtime_dir),
            runtime_dir,
        )


if __name__ == "__main__":
    unittest.main()
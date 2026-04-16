from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TrainingPipelineImportTests(unittest.TestCase):
    def test_training_pipeline_imports(self):
        from lewm.training.pipeline import build_training_manager

        self.assertTrue(callable(build_training_manager))


if __name__ == "__main__":
    unittest.main()
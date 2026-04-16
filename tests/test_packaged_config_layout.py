import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class PackagedConfigLayoutTests(unittest.TestCase):
    def test_packaged_train_and_eval_configs_exist(self):
        import lewm

        package_root = Path(lewm.__file__).resolve().parent
        self.assertTrue((package_root / "config" / "train" / "lewm.yaml").exists())
        self.assertTrue((package_root / "config" / "eval" / "pusht.yaml").exists())
        self.assertTrue((package_root / "config" / "eval" / "solver" / "adam.yaml").exists())

    def test_docs_reference_new_package_layout(self):
        readme = (ROOT / "README.md").read_text()
        claude = (ROOT / "CLAUDE.md").read_text()

        self.assertIn("src/lewm/config/train/lewm.yaml", readme)
        self.assertIn("src/lewm/evaluation/pipeline.py", claude)
        self.assertIn("src/lewm/models/jepa.py", claude)
        self.assertIn("may not deserialize", readme)
        self.assertNotIn("`train.py`: training entrypoint", claude)
        self.assertNotIn("`eval.py`: evaluation/planning entrypoint", claude)
        self.assertNotIn("`module.py`", claude)
        self.assertNotIn("`utils.py`", claude)

    def test_hydra_config_modules_are_importable(self):
        train_module = importlib.import_module("lewm.config.train")
        eval_module = importlib.import_module("lewm.config.eval")
        train_data_module = importlib.import_module("lewm.config.train.data")
        eval_solver_module = importlib.import_module("lewm.config.eval.solver")
        train_launcher_module = importlib.import_module("lewm.config.train.launcher")
        eval_launcher_module = importlib.import_module("lewm.config.eval.launcher")

        self.assertTrue((Path(train_module.__file__).resolve().parent / "lewm.yaml").exists())
        self.assertTrue((Path(eval_module.__file__).resolve().parent / "pusht.yaml").exists())
        self.assertTrue((Path(train_data_module.__file__).resolve().parent / "pusht.yaml").exists())
        self.assertTrue((Path(eval_solver_module.__file__).resolve().parent / "adam.yaml").exists())
        self.assertTrue((Path(train_launcher_module.__file__).resolve().parent / "local.yaml").exists())
        self.assertTrue((Path(eval_launcher_module.__file__).resolve().parent / "local.yaml").exists())


if __name__ == "__main__":
    unittest.main()

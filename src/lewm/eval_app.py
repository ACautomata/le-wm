import os
import subprocess
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from lewm.evaluation.pipeline import evaluate

os.environ["MUJOCO_GL"] = "egl"


def _git_tag():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


OmegaConf.register_new_resolver("git_tag", _git_tag)


@hydra.main(version_base=None, config_path="config/eval", config_name="pusht")
def main(cfg: DictConfig):
    runtime_output_dir = Path(HydraConfig.get().runtime.output_dir)
    evaluate(cfg, runtime_output_dir=runtime_output_dir)


run = main


if __name__ == "__main__":
    main()

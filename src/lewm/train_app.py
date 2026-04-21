import subprocess

import hydra
from omegaconf import OmegaConf

from lewm.training.pipeline import build_training_manager


def _git_tag():
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


OmegaConf.register_new_resolver("git_tag", _git_tag)


@hydra.main(version_base=None, config_path="config/train", config_name="lewm")
def main(cfg):
    manager = build_training_manager(cfg)
    manager()


run = main


if __name__ == "__main__":
    main()

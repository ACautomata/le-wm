import hydra

from lewm.training.pipeline import build_training_manager


@hydra.main(version_base=None, config_path="config/train", config_name="lewm")
def main(cfg):
    manager = build_training_manager(cfg)
    manager()


run = main


if __name__ == "__main__":
    main()

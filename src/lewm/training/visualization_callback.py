"""Callback for visualizing [CLS] embeddings using decoder."""

import lightning as pl
import torch
from lightning.pytorch import Callback


class VisualizationCallback(Callback):
    """Decodes and logs reconstructed images from [CLS] embeddings."""

    def __init__(self, log_interval: int = 500, num_samples: int = 4):
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ):
        if batch_idx % self.log_interval != 0:
            return

        model = pl_module.model
        if model.decoder is None:
            return

        with torch.no_grad():
            # Encode observations
            pixels = batch["pixels"].float()[:self.num_samples]
            info = {"pixels": pixels}
            info = model.encode(info)

            # Decode embeddings to images
            emb = info["emb"]  # (B, T, D)
            reconstructed = model.decode(emb)  # (B, T, C, H, W)

            # Log to WandB
            if trainer.logger and hasattr(trainer.logger, "experiment"):
                try:
                    import wandb

                    # Take first timestep for visualization
                    images = reconstructed[:, 0]  # (B, C, H, W)

                    trainer.logger.experiment.log(
                        {
                            "train/reconstructed_images": [
                                wandb.Image(img, caption=f"Sample {i}")
                                for i, img in enumerate(images)
                            ],
                            "train/original_images": [
                                wandb.Image(img, caption=f"Original {i}")
                                for i, img in enumerate(pixels[:, 0])
                            ],
                        },
                        step=trainer.global_step,
                    )
                except ImportError:
                    pass
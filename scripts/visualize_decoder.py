"""Standalone script for visualizing [CLS] embeddings from checkpoints."""

import argparse
import torch
from pathlib import Path
import stable_worldmodel as swm
from hydra.utils import instantiate
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from lewm.models.decoder import Decoder
from lewm.training.transforms import get_img_preprocessor


def visualize_checkpoint(ckpt_path: str, output_dir: str, num_samples: int = 10):
    """Load checkpoint and visualize reconstructed images."""
    ckpt_path = Path(ckpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_state = checkpoint["state_dict"]

    # Find encoder params
    encoder_params = {
        k.replace("model.encoder.", ""): v
        for k, v in model_state.items()
        if k.startswith("model.encoder.")
    }

    # Find projector params to get embed_dim
    projector_params = {
        k.replace("model.projector.", ""): v
        for k, v in model_state.items()
        if k.startswith("model.projector.")
    }

    # Infer dimensions from state dict
    embed_dim = projector_params["net.3.weight"].shape[0]

    # Load config from checkpoint directory
    config_path = ckpt_path.parent / "config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        img_size = cfg.img_size
        patch_size = cfg.patch_size
        encoder_scale = cfg.encoder_scale
    else:
        # Default values
        img_size = 224
        patch_size = 14
        encoder_scale = "tiny"

    # Rebuild encoder
    import stable_pretraining as spt
    encoder = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=img_size,
        pretrained=False,
        use_mask_token=False,
    )

    # Load encoder weights
    encoder.load_state_dict(encoder_params)

    # Create decoder
    num_patches = (img_size // patch_size) ** 2
    decoder = Decoder(
        cls_dim=embed_dim,
        num_patches=num_patches,
        patch_size=patch_size,
        hidden_dim=256,
        depth=4,
        heads=8,
        dim_head=32,
        mlp_dim=512,
        dropout=0.0,
    )

    # Load dataset for samples
    dataset_cfg = cfg.data.dataset if config_path.exists() else {"name": "pusht"}
    dataset = swm.data.HDF5Dataset(**dataset_cfg, transform=None)

    # Add preprocessing
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    dataset.transform = transform

    # Sample and visualize
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

    for i in range(num_samples):
        sample = dataset[i]
        pixels = sample["pixels"].unsqueeze(0)  # Add batch dim

        # Encode
        with torch.no_grad():
            output = encoder(pixels.float(), interpolate_pos_encoding=True)
            cls_emb = output.last_hidden_state[:, 0]  # [CLS] token

            # Decode
            reconstructed = decoder(cls_emb)

        # Plot original
        original_img = pixels[0].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")

        # Plot reconstructed
        recon_img = reconstructed[0].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "visualization.png", dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_dir / 'visualization.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize [CLS] embeddings")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to model checkpoint (_weights.ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )
    args = parser.parse_args()

    visualize_checkpoint(args.ckpt_path, args.output_dir, args.num_samples)
# Decoder Usage Guide

The decoder is a diagnostic tool for visualizing what visual information is retained in the [CLS] token embeddings.

## Architecture

Following the paper design:
- Input: [CLS] embedding (192 dim) from encoder
- Projection to hidden dimension (256)
- 196 learnable query tokens (for 224x224 image with 16x16 patches)
- 4 cross-attention layers with residual MLP blocks
- Output: reconstructed 224x224 RGB image

## Enable During Training

Add to `src/lewm/config/train/lewm.yaml`:

```yaml
decoder:
  enabled: True
  hidden_dim: 256
  depth: 4
  heads: 8
  dim_head: 32
  mlp_dim: 512
  dropout: 0.0

callbacks:
  visualization:
    _target_: lewm.training.visualization_callback.VisualizationCallback
    log_interval: 500
    num_samples: 4
```

This will log reconstructed images to WandB every 500 training steps.

## Post-Training Visualization

Run standalone script:

```bash
python scripts/visualize_decoder.py \
    --ckpt_path ~/.stable-wm/<run_id>/lewm_weights.ckpt \
    --output_dir ./visualizations \
    --num_samples 10
```

This generates a side-by-side comparison of original and reconstructed images.

## Note

The decoder is **not trained** - it's only used to probe the encoder representation quality.
The reconstruction quality indicates how much visual information is preserved in [CLS] embeddings.
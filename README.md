
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## What this fork adds

This repository keeps the original LeWorldModel research code centered on the JEPA world model, but adds a more packaged and experiment-friendly layer around the upstream stack. Compared with the upstream codebase, the main changes are:

- **Packaged training and evaluation entrypoints:** `lewm-train` and `lewm-eval` wrap the Hydra applications under `src/lewm/`, so experiments can be launched without relying on ad hoc scripts.
- **Hydra-first model assembly:** model components, losses, optimizers, callbacks, datasets, and planner settings are expressed through packaged config files, making architecture and experiment changes possible through CLI overrides.
- **Clearer repository boundaries:** project-specific model, training, evaluation, transform, callback, and config code has been organized under `src/lewm/`, while environment management, planning, dataset access, and training infrastructure continue to come from `stable-worldmodel` and `stable-pretraining`.
- **Training observability:** the fork adds Lightning/W&B monitoring callbacks for prediction loss, SIGReg loss, representation quality, gradient norms, learning rate, embedding statistics, and post-training metric plots.
- **Diagnostic visualization:** an optional decoder path reconstructs visual diagnostics from `[CLS]` embeddings, disabled by default so it does not affect normal training or planning.
- **Checkpoint and package-layout fixes:** object-checkpoint export, packaged config discovery, and package-layout assumptions are documented and covered by tests for the current `src/lewm` layout.

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they let this repository focus on LeWM-specific model, training, evaluation, monitoring, and diagnostic code.

**Installation:**
```bash
conda activate jepa
pip install -e .
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `src/lewm/config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`src/lewm/models/jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files packaged under `src/lewm/config/train/`.

Before training, set your WandB `entity` and `project` in `src/lewm/config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
lewm-train data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Training Monitoring

LeWM now includes a comprehensive training monitoring system that automatically logs key metrics to WandB. The system tracks:

**Row 1: Training Health**
- `train/pred_loss` (log scale) — prediction loss trend
- `train/sigreg_loss` — SIGReg regularization loss
- `val/pred_loss` — validation loss (if enabled)

**Row 2: Representation Quality (Research Core)**
- `representation/rankme_per_dim` — effective rank / embedding dimension (0-1 scale, avoid collapse)
- `representation/embedding_dim` — current embedding dimension (from config)
- `representation/embedding_norm_std` — embedding L2 norm standard deviation

**Row 3: System State**
- `system/grad_norm` — global gradient norm (detect explosion/vanishing)
- `system/learning_rate` — learning rate schedule (cosine annealing)

**Row 4: Embedding Statistics (Optional)**
- `embedding/mean`, `embedding/std`, `embedding/max`, `embedding/min`
- `embedding/temporal_cosine_sim_mean` — temporal embedding similarity

The monitoring system is automatically enabled when you run `lewm-train`. All metrics are logged via Lightning callbacks without polluting the main training path.

**Key Monitoring Guidelines:**
- **RankMe**: Keep `rankme_per_dim` in [0.5, 1.0] to avoid representation collapse
- **Gradient Norm**: Monitor for explosion (> 1000) or vanishing (< 1e-6)
- **Loss Balance**: Adjust `loss.sigreg.weight` based on pred_loss vs sigreg_loss trade-off

**Customize Monitoring Frequency:**
```bash
lewm-train monitoring.representation_interval=50 monitoring.system_interval=25
```

See [docs/wandb_dashboard_guide.md](docs/wandb_dashboard_guide.md) for detailed WandB dashboard configuration and metric interpretation guidelines.

### Monitoring System Robustness

The monitoring system implements defensive programming to ensure training safety:

- **Graceful Error Handling**: Missing metrics → warnings, not crashes
- **Dependency Validation**: All callbacks verify required data before computing
- **Exception Safety**: Computational errors caught and reported, training continues
- **Shape Validation**: Invalid tensor shapes rejected with clear warnings

Example warning output:
```
RuntimeWarning: [RepresentationQualityCallback] 'emb' not found in outputs at batch 42.
Skipping representation quality monitoring for this batch.
```

This ensures monitoring failures never interrupt training.

### Automatic Training Visualization

After training completes, `TrainingMetricsPlotCallback` automatically generates metric evolution plots:

- **4-row plot layout**: Matches WandB dashboard structure
- **Output location**: Saved to `run_dir/plots/` (PNG by default)
- **Format options**: PNG, PDF, SVG (configurable via `callbacks.training_metrics_plot.plot_format`)
- **DPI control**: Default 300, customizable

Generated plots:
```
run_dir/plots/
├── Row1_TrainingHealth.png         (pred_loss, sigreg_loss curves)
├── Row2_RepresentationQuality.png  (rankme_per_dim, embedding norms)
├── Row3_SystemState.png            (grad_norm, learning_rate)
└── Row4_EmbeddingStatistics.png    (embedding statistics)
```

Customize plot settings:
```bash
lewm-train callbacks.training_metrics_plot.plot_format=pdf callbacks.training_metrics_plot.dpi=600
```

Plots are generated automatically at training end—no manual intervention required.

## Planning

Evaluation configs live under `src/lewm/config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
lewm-eval --config-name=pusht policy=pusht/lewm

# ✗ incorrect
lewm-eval --config-name=pusht policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what the packaged evaluation entrypoint and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Because object checkpoints are serialized with Python module paths, checkpoints created before the `src/lewm` package migration may not deserialize under the new package layout. Re-export them from the new package layout if you need guaranteed compatibility.

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`

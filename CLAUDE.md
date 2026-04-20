# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

### Environment setup
```bash
conda activate jepa
pip install -e .
```

### Training
```bash
lewm-train
lewm-train data=pusht
lewm-train data=tworoom
```

Hydra composes the training config from `src/lewm/config/train/lewm.yaml` plus one dataset config from `src/lewm/config/train/data/*.yaml` (`pusht`, `tworoom`, `dmc`, `ogb`).
Override parameters from the CLI when iterating on experiments, for example:

```bash
lewm-train data=pusht wm.embed_dim=256 trainer.max_epochs=10
```

### Evaluation / planning
```bash
lewm-eval --config-name=pusht policy=pusht/lewm
lewm-eval --config-name=tworoom policy=tworoom/lewm
```

Use Hydra overrides for planner or eval settings, for example:

```bash
lewm-eval --config-name=pusht policy=pusht/lewm solver=adam eval.num_eval=10
```

`policy` must be the checkpoint path relative to `$STABLEWM_HOME`, without the `_object.ckpt` suffix.

### Data location
Datasets and checkpoints are resolved from `$STABLEWM_HOME` (default `~/.stable-wm/`).
Dataset names in configs omit the `.h5` suffix.

### Output directory structure
Training output is organized as `<git_tag>/<hydra_job_id>` via the `${git_tag:}` OmegaConf resolver registered in `train_app.py`.
Checkpoints land in this subdirectory under the Hydra runtime cwd.

### Tests
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Hatch scripts (from `pyproject.toml`):
```bash
hatch run test             # same as unittest discover
hatch run package-layout   # verify package config inclusion
hatch run shape-tests      # JEPA shape verification
```

## Architecture overview

This repository is a thin research layer on top of two external libraries:

- `stable-worldmodel`: environment management, planning APIs, dataset access, checkpoint loading during evaluation
- `stable-pretraining`: transforms, data splitting, training module wrapper, manager abstraction

Most repository-specific logic now lives under `src/lewm/`:

- `src/lewm/train_app.py`: packaged Hydra training entrypoint
- `src/lewm/eval_app.py`: packaged Hydra evaluation/planning entrypoint
- `src/lewm/models/jepa.py`: the world model itself — observation encoding, autoregressive rollout, goal encoding, and planning cost computation
- `src/lewm/models/components.py`: project-specific components such as `Embedder` and `MLP`
- `src/lewm/models/transformer.py`: transformer building blocks and `ARPredictor`
- `src/lewm/models/regularizers.py`: `SIGReg`
- `src/lewm/training/transforms.py`: preprocessing helpers used during training
- `src/lewm/training/callbacks.py`: object checkpoint export callback. Also contains monitoring callbacks: `RepresentationQualityCallback`, `SystemMonitoringCallback`, `EmbeddingStatisticsCallback`, `PredictionQualityCallback`, `WandBSummaryCallback`, `TrainingMetricsPlotCallback`.
- `src/lewm/training/forward.py`: `lejepa_forward()` loss computation
- `src/lewm/training/pipeline.py`: training orchestration — all model modules instantiated from Hydra config via `_target_` entries, enabling full architecture swaps via CLI overrides
- `src/lewm/evaluation/pipeline.py`: evaluation orchestration and results-path handling

## Training flow

Start from `src/lewm/train_app.py` or `src/lewm/training/pipeline.py` when changing training behavior.

1. Hydra loads `src/lewm/config/train/lewm.yaml` and a dataset config from `src/lewm/config/train/data/*.yaml`.
2. `swm.data.HDF5Dataset` loads trajectories from `$STABLEWM_HOME`.
3. Image preprocessing and non-image column normalization are assembled in `src/lewm/training/transforms.py` and attached as dataset transforms.
4. The model is built entirely from Hydra config (`wm.encoder`, `wm.predictor`, etc.).
   Each module specifies `_target_` in config, instantiated via `hydra.utils.instantiate`.
   The pipeline infers `hidden_dim` from the encoder and injects it into dependent modules.
   Users can swap any module architecture via CLI: `lewm-train wm.encoder._target_=my.Encoder`
5. `src/lewm/training/forward.py` defines `lejepa_forward()`, which computes the two-term LeWM objective:
   - prediction loss between predicted and target embeddings
   - `SIGReg` regularization over embeddings
6. `stable_pretraining.Manager` and Lightning run training and save outputs into a run directory under the stable-worldmodel cache.

Important output behavior:
- `ModelObjectCallBack` saves `<name>_epoch_*_object.ckpt` files for evaluation-time loading
- `stable_pretraining.Manager` also writes a weights checkpoint named `<name>_weights.ckpt`

## Evaluation / planning flow

`src/lewm/eval_app.py` is not a generic inference script; it plugs the model into the `stable-worldmodel` planning stack.

1. Hydra loads one environment config from `src/lewm/config/eval/*.yaml` and one solver config from `src/lewm/config/eval/solver/*.yaml`.
2. `src/lewm/evaluation/pipeline.py` builds a `swm.World` and dataset-backed evaluation episodes.
3. If `policy != random`, `swm.policy.AutoCostModel(cfg.policy)` loads the serialized object checkpoint.
4. The selected solver (for example CEM or gradient-based Adam) wraps the model in `swm.policy.WorldModelPolicy`.
5. During planning, the repository-specific contract is implemented by `JEPA.get_cost()` in `src/lewm/models/jepa.py`, which:
   - encodes the goal observation
   - rolls out predicted latent trajectories from candidate action sequences
   - returns a terminal embedding-space MSE cost

If you are changing planning behavior, inspect both `src/lewm/evaluation/pipeline.py` and the inference-only methods in `src/lewm/models/jepa.py` (`rollout`, `criterion`, `get_cost`).

## Configuration layout

- `src/lewm/config/train/lewm.yaml`: main training defaults — all model modules under `wm.*` use `_target_` for Hydra instantiate, `optimizers` is a structured dict, `loss.sigreg` uses `_target_`, dynamic parameters (`hidden_dim`, `input_dim`, `num_patches`) injected at runtime
- `src/lewm/config/train/data/*.yaml`: per-dataset settings such as dataset name, frameskip, loaded columns, and cached columns
- `src/lewm/config/train/launcher/local.yaml`: optional local training launcher override
- `src/lewm/config/eval/*.yaml`: per-environment evaluation settings (`pusht`, `tworoom`, `cube`, `reacher`), dataset-backed initialization, and output filenames
- `src/lewm/config/eval/solver/*.yaml`: planner choice and solver hyperparameters
- `src/lewm/config/eval/launcher/local.yaml`: local eval launcher defaults

This repository relies heavily on Hydra config composition, so behavior changes are often expressed in config before touching Python entrypoints.

## Repository-specific conventions

- This repo intentionally keeps its own code small; many behaviors come from the upstream `stable-worldmodel` and `stable-pretraining` packages rather than local modules.
- The training model path and the evaluation policy path use different checkpoint formats:
  - object checkpoints are used by `AutoCostModel` and must be referenced without `_object.ckpt`
  - weights checkpoints are raw `state_dict` dumps for manual loading
- Object checkpoints created before the `src/lewm` package migration may no longer deserialize correctly, because `torch.save(model, path)` pickles Python module paths. Re-export old checkpoints from the new package layout if you need guaranteed compatibility.
- `SIGReg` in `src/lewm/models/regularizers.py` is documented as single-GPU oriented; be careful when changing training parallelism assumptions.
- `src/lewm/eval_app.py` forces `MUJOCO_GL=egl` at import time, so evaluation assumes headless GPU rendering.
- Solver device defaults in `src/lewm/config/eval/solver/*.yaml` are hardcoded to `cuda`; if evaluation must be device-agnostic, start there as well as the model-loading path in `src/lewm/evaluation/pipeline.py`.
- The decoder module (`wm.decoder`) is referenced in `lewm.yaml` but disabled by default (`enabled: false`). The `lewm.models.decoder` package does not exist yet — enable only after implementation.
- W&B logging is enabled by default (`wandb.enabled: True` in `lewm.yaml`). Set `wandb.enabled=false` via CLI to disable.

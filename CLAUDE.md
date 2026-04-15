# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

### Environment setup
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

### Training
```bash
python train.py
python train.py data=pusht
python train.py data=tworoom
```

Hydra composes the training config from `config/train/lewm.yaml` plus one dataset config from `config/train/data/*.yaml`.
Override parameters from the CLI when iterating on experiments, for example:

```bash
python train.py data=pusht wm.embed_dim=256 trainer.max_epochs=10
```

### Evaluation / planning
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

Use Hydra overrides for planner or eval settings, for example:

```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm solver=adam eval.num_eval=10
```

`policy` must be the checkpoint path relative to `$STABLEWM_HOME`, without the `_object.ckpt` suffix.

### Data location
Datasets and checkpoints are resolved from `$STABLEWM_HOME` (default `~/.stable-wm/`).
Dataset names in configs omit the `.h5` suffix.

### Tests and linting
There is currently no repo-local test suite, lint command, `Makefile`, or `justfile` in this repository. Do not invent `pytest`, single-test, or lint commands without first adding them to the repository.

## Architecture overview

This repository is a thin research layer on top of two external libraries:

- `stable-worldmodel`: environment management, planning APIs, dataset access, checkpoint loading during evaluation
- `stable-pretraining`: transforms, data splitting, training module wrapper, manager abstraction

Most repository-specific logic lives in five files:

- `train.py`: training entrypoint that wires dataset loading, preprocessing, model construction, Lightning trainer setup, WandB logging, and checkpoint export
- `eval.py`: evaluation/planning entrypoint that builds a `stable_worldmodel.World`, loads a policy checkpoint, and evaluates it from dataset-defined start/goal states
- `jepa.py`: the world model itself — observation encoding, autoregressive rollout, goal encoding, and planning cost computation
- `module.py`: repository-specific neural building blocks such as `SIGReg`, the action embedder, and the autoregressive predictor transformer
- `utils.py`: preprocessing helpers plus `ModelObjectCallBack`, which exports the serialized model object checkpoints expected by evaluation

## Training flow

`train.py` is the place to start when changing model behavior.

1. Hydra loads `config/train/lewm.yaml` and a dataset config from `config/train/data/*.yaml`.
2. `swm.data.HDF5Dataset` loads trajectories from `$STABLEWM_HOME`.
3. Image preprocessing and non-image column normalization are assembled in `utils.py` and attached as dataset transforms.
4. The model is built from:
   - a ViT encoder from `stable_pretraining`
   - `Embedder` for actions
   - `ARPredictor` for autoregressive next-embedding prediction
   - MLP projectors around encoder and predictor outputs
5. `train.py:17` defines `lejepa_forward()`, which computes the two-term LeWM objective:
   - prediction loss between predicted and target embeddings
   - `SIGReg` regularization over embeddings
6. `stable_pretraining.Manager` and Lightning run training and save outputs into a run directory under the stable-worldmodel cache.

Important output behavior:
- `ModelObjectCallBack` saves `<name>_epoch_*_object.ckpt` files for evaluation-time loading
- `stable_pretraining.Manager` also writes a weights checkpoint named `<name>_weights.ckpt`

## Evaluation / planning flow

`eval.py` is not a generic inference script; it plugs the model into the `stable-worldmodel` planning stack.

1. Hydra loads one environment config from `config/eval/*.yaml` and one solver config from `config/eval/solver/*.yaml`.
2. `eval.py` builds a `swm.World` and dataset-backed evaluation episodes.
3. If `policy != random`, `swm.policy.AutoCostModel(cfg.policy)` loads the serialized object checkpoint.
4. The selected solver (for example CEM or gradient-based Adam) wraps the model in `swm.policy.WorldModelPolicy`.
5. During planning, the repository-specific contract is implemented by `JEPA.get_cost()` in `jepa.py`, which:
   - encodes the goal observation
   - rolls out predicted latent trajectories from candidate action sequences
   - returns a terminal embedding-space MSE cost

If you are changing planning behavior, inspect both `eval.py` and the inference-only methods in `jepa.py` (`rollout`, `criterion`, `get_cost`).

## Configuration layout

- `config/train/lewm.yaml`: main training defaults for optimizer, trainer, model dimensions, WandB, and loss weight
- `config/train/data/*.yaml`: per-dataset settings such as dataset name, frameskip, loaded columns, and cached columns
- `config/train/launcher/local.yaml`: optional local training launcher override (not included by default from `config/train/lewm.yaml`)
- `config/eval/*.yaml`: per-environment evaluation settings, dataset-backed initialization, and output filenames
- `config/eval/solver/*.yaml`: planner choice and solver hyperparameters
- `config/eval/launcher/local.yaml`: local eval launcher defaults

This repository relies heavily on Hydra config composition, so behavior changes are often expressed in config before touching Python entrypoints.

## Repository-specific conventions

- This repo intentionally keeps its own code small; many behaviors come from the upstream `stable-worldmodel` and `stable-pretraining` packages rather than local modules.
- The training model path and the evaluation policy path use different checkpoint formats:
  - object checkpoints are used by `AutoCostModel` and must be referenced without `_object.ckpt`
  - weights checkpoints are raw `state_dict` dumps for manual loading
- `SIGReg` in `module.py` is documented as single-GPU oriented; be careful when changing training parallelism assumptions.
- `eval.py` forces `MUJOCO_GL=egl` at import time, so evaluation assumes headless GPU rendering.
- Solver device defaults in `config/eval/solver/*.yaml` are hardcoded to `cuda`; if evaluation must be device-agnostic, start there as well as the model-loading path in `eval.py`.

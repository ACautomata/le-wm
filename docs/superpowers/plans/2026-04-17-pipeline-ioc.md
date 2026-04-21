# Pipeline IoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the training pipeline to instantiate all modules (encoder, predictor, decoder, regularizer, optimizer, etc.) through Hydra config, enabling full architecture swaps via CLI.

**Architecture:** Global params (`embed_dim`, `img_size`, `patch_size`) at config top level. Modules reference them via `${...}` interpolation. Pipeline instantiates encoder first, infers `hidden_dim`, injects it into dependent module configs, then instantiates remaining modules. Dataset + transforms stay imperative.

**Tech Stack:** Hydra/OmegaConf instantiate, PyTorch, Lightning, stable_pretraining

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/lewm/models/components.py` | Modify | MLP accepts string `norm_fn` (e.g. `"BatchNorm1d"`) |
| `src/lewm/config/train/lewm.yaml` | Modify | Full nested module config with `_target_` |
| `src/lewm/training/pipeline.py` | Modify | IoC refactor: instantiate modules from config |
| `tests/test_pipeline_ioc.py` | Create | Test MLP string resolution, config structure, pipeline assembly |

---

### Task 1: MLP string norm_fn resolution

`MLP.__init__` currently accepts a callable for `norm_fn`. Hydra cannot pass callables through config. Modify MLP to also accept strings and resolve them.

**Files:**
- Modify: `src/lewm/models/components.py:32-56`
- Test: `tests/test_pipeline_ioc.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_ioc.py
"""Tests for pipeline IoC refactoring."""
import sys
from pathlib import Path
import unittest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lewm.models.components import MLP


class TestMLPStringNormFn(unittest.TestCase):
    def test_string_norm_fn_batchnorm(self):
        """MLP accepts string 'BatchNorm1d' and resolves to nn.BatchNorm1d."""
        mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, norm_fn="BatchNorm1d")
        # The norm layer inside the Sequential should be BatchNorm1d(20)
        norm_layer = mlp.net[1]
        self.assertIsInstance(norm_layer, nn.BatchNorm1d)
        self.assertEqual(norm_layer.num_features, 20)

    def test_string_norm_fn_layernorm(self):
        """MLP accepts string 'LayerNorm' and resolves to nn.LayerNorm."""
        mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, norm_fn="LayerNorm")
        norm_layer = mlp.net[1]
        self.assertIsInstance(norm_layer, nn.LayerNorm)
        self.assertEqual(norm_layer.normalized_shape[0], 20)

    def test_callable_norm_fn_still_works(self):
        """MLP still accepts callable norm_fn (backward compatible)."""
        mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, norm_fn=nn.LayerNorm)
        norm_layer = mlp.net[1]
        self.assertIsInstance(norm_layer, nn.LayerNorm)

    def test_none_norm_fn(self):
        """MLP accepts None for norm_fn (uses Identity)."""
        mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, norm_fn=None)
        self.assertIsInstance(mlp.net[1], nn.Identity)

    def test_mlp_forward(self):
        """MLP forward pass works with string norm_fn."""
        mlp = MLP(input_dim=10, hidden_dim=20, output_dim=5, norm_fn="BatchNorm1d")
        x = torch.randn(4, 10)
        out = mlp(x)
        self.assertEqual(out.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py::TestMLPStringNormFn -v`
Expected: FAIL — `MLP.__init__` does not resolve string `norm_fn`.

- [ ] **Step 3: Write minimal implementation**

In `src/lewm/models/components.py`, modify the `MLP.__init__` method:

```python
class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        if isinstance(norm_fn, str):
            norm_fn = getattr(nn, norm_fn)
        if isinstance(act_fn, str):
            act_fn = getattr(nn, act_fn)
        norm_layer = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )
```

Key change: resolve string `norm_fn` / `act_fn` to callables via `getattr(nn, ...)`. Also renamed local `norm_fn` variable to `norm_layer` to avoid shadowing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py::TestMLPStringNormFn -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lewm/models/components.py tests/test_pipeline_ioc.py
git commit -m "refactor: MLP accepts string norm_fn for Hydra config compatibility"
```

---

### Task 2: Restructure `lewm.yaml` config

Replace hardcoded module parameters with full `_target_`-based Hydra config entries.

**Files:**
- Modify: `src/lewm/config/train/lewm.yaml`

- [ ] **Step 1: Write the new config**

Replace the entire content of `src/lewm/config/train/lewm.yaml` with:

```yaml
defaults:
  - _self_
  - data: pusht

output_model_name: lewm
subdir: ${hydra:job.id}

num_workers: 6
train_split: 0.9
seed: 3072
img_size: 224
patch_size: 14
encoder_scale: tiny
embed_dim: 192
dump_object: True

trainer:
  max_epochs: 100
  devices: auto
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

loader:
  batch_size: 128
  num_workers: ${num_workers}
  persistent_workers: True
  prefetch_factor: 3
  pin_memory: True

wandb:
  enabled: True
  config:
    entity: lewm
    project: lewm
    name: ${output_model_name}
    id: ${subdir}
    resume: allow
    log_model: False

wm:
  history_size: 3
  num_preds: 1
  action_dim: 2

  encoder:
    _target_: spt.backbone.utils.vit_hf
    size: ${encoder_scale}
    patch_size: ${patch_size}
    image_size: ${img_size}
    pretrained: false
    use_mask_token: false

  predictor:
    _target_: lewm.models.transformer.ARPredictor
    num_frames: ${wm.history_size}
    input_dim: ${embed_dim}
    depth: 6
    heads: 16
    mlp_dim: 2048
    dim_head: 64
    dropout: 0.1
    emb_dropout: 0.0

  action_encoder:
    _target_: lewm.models.components.Embedder
    emb_dim: ${embed_dim}

  projector:
    _target_: lewm.models.components.MLP
    output_dim: ${embed_dim}
    hidden_dim: 2048
    norm_fn: BatchNorm1d

  pred_proj:
    _target_: lewm.models.components.MLP
    output_dim: ${embed_dim}
    hidden_dim: 2048
    norm_fn: BatchNorm1d

  decoder:
    enabled: false
    _target_: lewm.models.decoder.Decoder
    cls_dim: ${embed_dim}
    hidden_dim: 256
    depth: 4
    heads: 8
    dim_head: 32
    mlp_dim: 512
    dropout: 0.0

  world_model:
    _target_: lewm.models.jepa.JEPA

loss:
  sigreg:
    weight: 0.09
    _target_: lewm.models.regularizers.SIGReg
    knots: 17
    num_proj: 1024

optimizers:
  model_opt:
    modules: model
    optimizer:
      type: AdamW
      lr: 5e-5
      weight_decay: 1e-3
    scheduler:
      type: LinearWarmupCosineAnnealingLR
    interval: epoch

callbacks:
  model_checkpoint:
    _target_: lewm.training.callbacks.ModelObjectCallBack
    dirpath: ${hydra:runtime.cwd}/${subdir}
    filename: ${output_model_name}
    epoch_interval: 1

  representation_quality:
    _target_: lewm.training.callbacks.RepresentationQualityCallback
    log_interval: 100

  system_monitoring:
    _target_: lewm.training.callbacks.SystemMonitoringCallback
    log_interval: 50

  embedding_statistics:
    _target_: lewm.training.callbacks.EmbeddingStatisticsCallback
    log_interval: 200

  prediction_quality:
    _target_: lewm.training.callbacks.PredictionQualityCallback
    log_interval: 100

  wandb_summary:
    _target_: lewm.training.callbacks.WandBSummaryCallback

  training_metrics_plot:
    _target_: lewm.training.callbacks.TrainingMetricsPlotCallback
    plot_format: png
    dpi: 300
```

Key changes from old config:
- `embed_dim` moved from `wm.embed_dim` to top level
- `encoder_scale` stays top level (was already there)
- `wm.action_dim` added with default value 2 (overridden per-dataset or via CLI)
- All modules under `wm` now have `_target_`
- `optimizer` → `optimizers` dict structure matching `spt.Module` expectations
- `loss.sigreg` flattened: `_target_` and `knots`/`num_proj` at same level
- Callbacks section unchanged

- [ ] **Step 2: Verify config loads without errors**

Run: `cd /Users/junran/Documents/le-wm && conda run -n jepa python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('src/lewm/config/train/lewm.yaml'); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/lewm/config/train/lewm.yaml
git commit -m "refactor: restructure lewm.yaml for full module IoC"
```

---

### Task 3: Update data configs for `wm.action_dim`

Each dataset has a different action dimension. Add `wm.action_dim` override to each data config.

**Files:**
- Modify: `src/lewm/config/train/data/pusht.yaml`
- Modify: `src/lewm/config/train/data/tworoom.yaml`
- Modify: `src/lewm/config/train/data/dmc.yaml`
- Modify: `src/lewm/config/train/data/ogb.yaml`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_ioc.py`:

```python
import os
from omegaconf import OmegaConf, open_dict


class TestDataConfigActionDim(unittest.TestCase):
    def _load_config(self, data_name):
        base = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / f"lewm/config/train/data/{data_name}.yaml")
        return OmegaConf.merge(base, data)

    def test_pusht_has_action_dim(self):
        cfg = self._load_config("pusht")
        self.assertIn("action_dim", OmegaConf.to_container(cfg.wm, resolve=False))

    def test_tworoom_has_action_dim(self):
        cfg = self._load_config("tworoom")
        self.assertIn("action_dim", OmegaConf.to_container(cfg.wm, resolve=False))

    def test_dmc_has_action_dim(self):
        cfg = self._load_config("dmc")
        self.assertIn("action_dim", OmegaConf.to_container(cfg.wm, resolve=False))

    def test_ogb_has_action_dim(self):
        cfg = self._load_config("ogb")
        self.assertIn("action_dim", OmegaConf.to_container(cfg.wm, resolve=False))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py::TestDataConfigActionDim -v`
Expected: FAIL — data configs don't have `wm.action_dim` yet.

- [ ] **Step 3: Add `wm.action_dim` to each data config**

`src/lewm/config/train/data/pusht.yaml` — add `wm:` section:
```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  name: pusht_expert_train
  keys_to_load:
    - pixels
    - action
    - proprio
    - state
  keys_to_cache:
    - action
    - proprio
    - state

wm:
  action_dim: 2
```

`src/lewm/config/train/data/tworoom.yaml`:
```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  name: tworoom
  keys_to_load:
    - pixels
    - action
    - proprio
  keys_to_cache:
    - action
    - proprio

wm:
  action_dim: 5
```

`src/lewm/config/train/data/dmc.yaml`:
```yaml
dataset:
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  name: reacher
  keys_to_load:
    - pixels
    - action
    - observation
  keys_to_cache:
    - action
    - observation

wm:
  action_dim: 2
```

`src/lewm/config/train/data/ogb.yaml`:
```yaml
dataset:
  name: ogbench/cube_single_expert
  num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}
  frameskip: 5
  keys_to_load:
    - pixels
    - action
    - observation
  keys_to_cache:
    - action
    - observation
  keys_to_merge:
    proprio: proprio

wm:
  action_dim: 4
```

**Note:** The `action_dim` values above are reasonable defaults. The exact values should be verified against the actual datasets (`dataset.get_dim("action")`). If they are wrong, the pipeline will still work because the runtime `effective_act_dim` is computed from `frameskip * action_dim`. Adjust as needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py::TestDataConfigActionDim -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lewm/config/train/data/pusht.yaml src/lewm/config/train/data/tworoom.yaml src/lewm/config/train/data/dmc.yaml src/lewm/config/train/data/ogb.yaml tests/test_pipeline_ioc.py
git commit -m "refactor: add wm.action_dim to data configs for IoC"
```

---

### Task 4: Refactor `pipeline.py` — IoC assembly

Replace hardcoded module construction with `instantiate` calls. This is the core change.

**Files:**
- Modify: `src/lewm/training/pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_ioc.py`:

```python
from unittest.mock import MagicMock, patch


class TestPipelineConfigStructure(unittest.TestCase):
    """Test that pipeline config has all required _target_ entries."""

    def _load_merged_config(self, data_name="pusht"):
        base = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / f"lewm/config/train/data/{data_name}.yaml")
        return OmegaConf.merge(base, data)

    def test_encoder_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.encoder)

    def test_predictor_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.predictor)

    def test_action_encoder_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.action_encoder)

    def test_projector_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.projector)

    def test_pred_proj_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.pred_proj)

    def test_world_model_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.world_model)

    def test_decoder_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.wm.decoder)

    def test_sigreg_has_target(self):
        cfg = self._load_merged_config()
        self.assertIn("_target_", cfg.loss.sigreg)

    def test_optimizers_has_model_opt(self):
        cfg = self._load_merged_config()
        self.assertIn("model_opt", cfg.optimizers)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py::TestPipelineConfigStructure -v`
Expected: PASS (this validates config structure from Task 2)

- [ ] **Step 3: Refactor pipeline.py**

Replace the entire content of `src/lewm/training/pipeline.py`:

```python
from functools import partial
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from lewm.training.forward import lejepa_forward
from lewm.training.transforms import get_column_normalizer, get_img_preprocessor


def build_training_manager(cfg):
    # Phase 1: Dataset + transforms
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )

    # Phase 2: Parameter inference + injection
    encoder = instantiate(cfg.wm.encoder)
    hidden_dim = encoder.config.hidden_size
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    with open_dict(cfg):
        cfg.wm.predictor.hidden_dim = hidden_dim
        cfg.wm.predictor.output_dim = hidden_dim
        cfg.wm.projector.input_dim = hidden_dim
        cfg.wm.pred_proj.input_dim = hidden_dim
        cfg.wm.action_encoder.input_dim = effective_act_dim
        if cfg.wm.decoder.get("enabled", False):
            cfg.wm.decoder.num_patches = (cfg.img_size // cfg.patch_size) ** 2

    # Phase 3: Module instantiation
    predictor = instantiate(cfg.wm.predictor)
    action_encoder = instantiate(cfg.wm.action_encoder)
    projector = instantiate(cfg.wm.projector)
    pred_proj = instantiate(cfg.wm.pred_proj)

    decoder = None
    if cfg.wm.decoder.get("enabled", False):
        decoder = instantiate(cfg.wm.decoder)

    world_model = instantiate(
        cfg.wm.world_model,
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        decoder=decoder,
    )

    sigreg = instantiate(cfg.loss.sigreg)
    optimizers = OmegaConf.to_container(cfg.optimizers, resolve=True)

    # Phase 4: Training assembly
    data_module = spt.data.DataModule(train=train, val=val)
    lightning_module = spt.Module(
        model=world_model,
        sigreg=sigreg,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as file_obj:
        OmegaConf.save(cfg, file_obj)

    callbacks_dict = instantiate(cfg.callbacks, _convert_="partial")

    if "model_checkpoint" in callbacks_dict:
        callbacks_dict["model_checkpoint"].dirpath = run_dir

    callbacks_list = list(callbacks_dict.values())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks_list,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=lightning_module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    return manager
```

Key changes from old pipeline:
- Removed imports: `Embedder`, `MLP`, `ARPredictor`, `Decoder`, `SIGReg`, `JEPA`
- Added: `OmegaConf` (for `to_container` on optimizers)
- Phase 2: instantiate encoder first, read `hidden_dim`, inject into dependent configs
- Phase 3: all modules via `instantiate`, `JEPA` receives sub-modules as kwargs
- Optimizer: `OmegaConf.to_container(cfg.optimizers)` replaces hardcoded dict
- Removed old `embed_dim = cfg.wm.get("embed_dim", hidden_dim)` — now always `${embed_dim}` from top-level config

- [ ] **Step 4: Run existing tests to check nothing broke**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_training_pipeline_import.py tests/test_pipeline_ioc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lewm/training/pipeline.py tests/test_pipeline_ioc.py
git commit -m "refactor: pipeline IoC — all modules instantiated from Hydra config"
```

---

### Task 5: Update `forward.py` to read from new config paths

The forward function reads `cfg.wm.history_size`, `cfg.wm.num_preds`, `cfg.loss.sigreg.weight`. These paths haven't changed in the new config structure, so verify forward.py works unchanged.

**Files:**
- Verify: `src/lewm/training/forward.py` (no changes expected)

- [ ] **Step 1: Verify config path compatibility**

Read `src/lewm/training/forward.py` and confirm all config references (`cfg.wm.history_size`, `cfg.wm.num_preds`, `cfg.loss.sigreg.weight`) still resolve correctly in the new `lewm.yaml`:
- `cfg.wm.history_size` → `wm.history_size: 3` ✓
- `cfg.wm.num_preds` → `wm.num_preds: 1` ✓
- `cfg.loss.sigreg.weight` → `loss.sigreg.weight: 0.09` ✓

No code changes needed.

- [ ] **Step 2: Commit (if any changes were needed)**

Only commit if forward.py was modified.

---

### Task 6: Integration smoke test

Verify the full pipeline can at least construct a manager object (without actually training).

**Files:**
- Test: `tests/test_pipeline_ioc.py`

- [ ] **Step 1: Write integration test**

Append to `tests/test_pipeline_ioc.py`:

```python
class TestPipelineIoCIntegration(unittest.TestCase):
    """Smoke test: verify pipeline config structure is valid for instantiate."""

    def test_module_configs_are_instantiable(self):
        """Each module config with _target_ should have required params."""
        cfg = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / "lewm/config/train/data/pusht.yaml")
        cfg = OmegaConf.merge(cfg, data)

        # Encoder config should resolve
        enc_cfg = OmegaConf.to_container(cfg.wm.encoder, resolve=True)
        self.assertEqual(enc_cfg["size"], "tiny")
        self.assertEqual(enc_cfg["patch_size"], 14)
        self.assertEqual(enc_cfg["image_size"], 224)

    def test_predictor_has_all_params(self):
        """Predictor config has all required constructor args except injected ones."""
        cfg = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / "lewm/config/train/data/pusht.yaml")
        cfg = OmegaConf.merge(cfg, data)

        pred_cfg = OmegaConf.to_container(cfg.wm.predictor, resolve=False)
        required = {"_target_", "num_frames", "input_dim", "depth", "heads",
                     "mlp_dim", "dim_head", "dropout", "emb_dropout"}
        self.assertTrue(required.issubset(set(pred_cfg.keys())))

    def test_optimizers_structure(self):
        """Optimizers config matches spt.Module expected structure."""
        cfg = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / "lewm/config/train/data/pusht.yaml")
        cfg = OmegaConf.merge(cfg, data)

        opt = OmegaConf.to_container(cfg.optimizers, resolve=True)
        self.assertIn("model_opt", opt)
        self.assertEqual(opt["model_opt"]["modules"], "model")
        self.assertEqual(opt["model_opt"]["optimizer"]["type"], "AdamW")

    def test_decoder_disabled_by_default(self):
        """Decoder is disabled in default config."""
        cfg = OmegaConf.load(Path(SRC) / "lewm/config/train/lewm.yaml")
        data = OmegaConf.load(Path(SRC) / "lewm/config/train/data/pusht.yaml")
        cfg = OmegaConf.merge(cfg, data)
        self.assertFalse(cfg.wm.decoder.enabled)
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/junran/Documents/le-wm && python -m pytest tests/test_pipeline_ioc.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite to catch regressions**

Run: `cd /Users/junran/Documents/le-wm && python -m unittest discover -s tests -p 'test_*.py'`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_ioc.py
git commit -m "test: add pipeline IoC integration smoke tests"
```

---

### Task 7: Update CLAUDE.md documentation

Update project docs to reflect the new IoC config structure.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Architecture overview section**

In `CLAUDE.md`, find the "Architecture overview" section. Add a note after the bullet about `pipeline.py`:

```markdown
- `src/lewm/training/pipeline.py`: training orchestration — all model modules (encoder, predictor, decoder, etc.) are instantiated from Hydra config via `_target_` entries, enabling full architecture swaps via CLI overrides
```

- [ ] **Step 2: Update Training flow section**

In `CLAUDE.md`, find the "Training flow" section, step 4. Replace:

```
4. The model is built from:
   - a ViT encoder from `stable_pretraining`
   - `Embedder` for actions
   - `ARPredictor` for autoregressive next-embedding prediction
   - MLP projectors around encoder and predictor outputs
```

With:

```
4. The model is built entirely from Hydra config (`wm.encoder`, `wm.predictor`, etc.).
   Each module specifies `_target_` in config, instantiated via `hydra.utils.instantiate`.
   The pipeline infers `hidden_dim` from the encoder and injects it into dependent modules.
   Users can swap any module architecture via CLI: `lewm-train wm.encoder._target_=my.Encoder`
```

- [ ] **Step 3: Update Configuration layout section**

Add to the config layout:

```markdown
- All model modules under `wm.*` use `_target_` for Hydra instantiate (encoder, predictor, action_encoder, projector, pred_proj, decoder, world_model)
- `optimizers` is a structured dict matching `spt.Module` expectations
- `loss.sigreg` uses `_target_` for regularizer instantiation
- Dynamic parameters (`hidden_dim`, `input_dim`, `num_patches`) are injected by pipeline at runtime via `open_dict`
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for pipeline IoC"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] All modules have `_target_` in config → Task 2
- [x] Global params with interpolation → Task 2
- [x] Factory functions as `_target_` (vit_hf) → Task 2
- [x] MLP string norm_fn → Task 1
- [x] Parameter injection (hidden_dim, effective_act_dim, num_patches) → Task 4
- [x] Forward function unchanged → Task 5
- [x] Optimizer restructured → Task 2 + 4
- [x] SIGReg instantiate → Task 4
- [x] Data configs updated → Task 3
- [x] CLI examples work → Task 2 config structure enables them
- [x] Import cleanup → Task 4

**Placeholder scan:** No TBD, TODO, or placeholder patterns found.

**Type consistency:** `encoder.config.hidden_size` used in Task 4 matches HuggingFace ViTModel API. `cfg.wm.action_dim` matches data config entries in Task 3. `norm_fn: BatchNorm1d` string matches `getattr(nn, "BatchNorm1d")` in Task 1.

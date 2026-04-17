# Pipeline IoC Design: Full Hydra Inversion of Control

## Goal

Extend the Hydra instantiate pattern (already applied to callbacks) to all pipeline modules: encoder, predictor, action_encoder, projector, pred_proj, decoder, world_model, regularizer, and optimizer. Users can swap any module's architecture via CLI without code changes.

## Design Decisions

1. **Fully swappable** - all modules specify `_target_` in config
2. **Global params + interpolation** - `embed_dim`, `img_size`, `patch_size` at config top level, modules reference via `${...}`
3. **Factory functions as `_target_`** - e.g. `spt.backbone.utils.vit_hf` directly in config
4. **Forward stays as function** - `lejepa_forward` keeps `partial(..., cfg=cfg)`, reads params from config

## Config Structure (`lewm.yaml`)

```yaml
img_size: 224
patch_size: 14
embed_dim: 192
encoder_scale: tiny

wm:
  history_size: 3
  num_preds: 1

  encoder:
    _target_: spt.backbone.utils.vit_hf
    scale: ${encoder_scale}
    patch_size: ${patch_size}
    image_size: ${img_size}
    pretrained: false
    use_mask_token: false

  predictor:
    _target_: lewm.models.transformer.ARPredictor
    num_frames: ${wm.history_size}
    input_dim: ${embed_dim}
    # hidden_dim, output_dim: injected by pipeline from encoder

  action_encoder:
    _target_: lewm.models.components.Embedder
    # input_dim: injected by pipeline (frameskip * action_dim)
    emb_dim: ${embed_dim}

  projector:
    _target_: lewm.models.components.MLP
    # input_dim: injected by pipeline (= hidden_dim)
    output_dim: ${embed_dim}
    hidden_dim: 2048
    norm_fn: BatchNorm1d

  pred_proj:
    _target_: lewm.models.components.MLP
    # input_dim: injected by pipeline (= hidden_dim)
    output_dim: ${embed_dim}
    hidden_dim: 2048
    norm_fn: BatchNorm1d

  decoder:
    enabled: false
    _target_: lewm.models.decoder.Decoder
    cls_dim: ${embed_dim}
    # num_patches: injected by pipeline

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

# callbacks, wandb, trainer, loader: unchanged
```

## Pipeline Refactor (`pipeline.py`)

### Phase 1: Dataset + transforms (unchanged)

Dataset and transforms depend on runtime data (normalizer fitting, column dims), so this stays as imperative code.

### Phase 2: Parameter inference + injection

After instantiating encoder, read `hidden_dim` from it. Compute `effective_act_dim` from config. Inject into dependent module configs via `open_dict`:

- `predictor.hidden_dim`, `predictor.output_dim` = hidden_dim
- `projector.input_dim`, `pred_proj.input_dim` = hidden_dim
- `action_encoder.input_dim` = frameskip * action_dim
- `decoder.num_patches` = (img_size // patch_size) ** 2

### Phase 3: Module instantiation

```python
encoder = instantiate(cfg.wm.encoder)
predictor = instantiate(cfg.wm.predictor)
action_encoder = instantiate(cfg.wm.action_encoder)
projector = instantiate(cfg.wm.projector)
pred_proj = instantiate(cfg.wm.pred_proj)
decoder = instantiate(cfg.wm.decoder) if cfg.wm.decoder.get("enabled", False) else None
world_model = instantiate(cfg.wm.world_model,
    encoder=encoder, predictor=predictor,
    action_encoder=action_encoder, projector=projector,
    pred_proj=pred_proj, decoder=decoder)
sigreg = instantiate(cfg.loss.sigreg)
optimizers = OmegaConf.to_container(cfg.optimizers, resolve=True)
```

### Phase 4: Training assembly (unchanged)

spt.Module, logger, callbacks, trainer, manager assembly stays the same.

## Import Changes

**Remove:**
- `from lewm.models.components import Embedder, MLP`
- `from lewm.models.transformer import ARPredictor`
- `from lewm.models.decoder import Decoder`
- `from lewm.models.regularizers import SIGReg`
- `from lewm.models.jepa import JEPA`

**Keep:**
- `from hydra.utils import instantiate`
- `from lewm.training.forward import lejepa_forward`
- `from lewm.training.transforms import get_column_normalizer, get_img_preprocessor`

## MLP norm_fn Handling

`MLP.__init__` currently accepts a callable for `norm_fn`. Hydra cannot serialize callables. Options:

- Use a string key mapped to callable in MLP (e.g. `norm_fn: BatchNorm1d` → `nn.BatchNorm1d`)
- Or add a simple resolver: `OmegaConf.register_new_resolver("norm_fn", lambda name: getattr(nn, name))`

This requires a small change to `MLP.__init__` to accept string norm_fn names.

## CLI Usage Examples

```bash
# Swap encoder
lewm-train wm.encoder._target_=my_models.cnn_encoder.CNNEncoder wm.encoder.hidden_size=512

# Swap predictor
lewm-train wm.predictor._target_=my_models.lstm.LSTMPredictor wm.predictor.hidden_dim=512

# Toggle decoder
lewm-train wm.decoder.enabled=true wm.decoder.depth=6

# Swap optimizer
lewm-train optimizers.model_opt.optimizer.type=SGD optimizers.model_opt.optimizer.lr=1e-3

# Swap regularizer
lewm-train loss.sigreg._target_=my_module.VICReg loss.sigreg.weight=0.1
```

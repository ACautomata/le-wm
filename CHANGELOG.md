# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-04-16

### Added - Defensive Programming & Plot Callback - 2026-04-16 (v3)

Enhanced monitoring system robustness and added training metrics visualization.

#### 1. Defensive Programming for All Callbacks

**Improvement**: All callbacks now validate dependencies and catch exceptions, outputting warnings instead of crashing training.

**Changes**:
- Added dependency validation before computing metrics
- Wrapped all computations in try-except blocks
- Used `warnings.warn()` for error reporting (not `print()`)
- Added shape validation for tensors

**Affected Callbacks**:
- `RepresentationQualityCallback`: Validate 'emb' exists and has valid shape
- `SystemMonitoringCallback`: Catch exceptions in gradient/learning_rate computation
- `EmbeddingStatisticsCallback`: Validate 'emb' and handle invalid shapes
- `PredictionQualityCallback`: Validate 'pred_loss', 'pred_emb', 'tgt_emb'
- `WandBSummaryCallback`: Validate logger availability before updating summary

**Example Error Handling**:
```python
# Missing dependency
if "emb" not in outputs:
    warnings.warn(
        f"[RepresentationQualityCallback] 'emb' not found in outputs at batch {batch_idx}. "
        "Skipping representation quality monitoring for this batch.",
        RuntimeWarning
    )
    return

# Exception caught
try:
    rankme = self._compute_rankme(emb_flat)
    pl_module.log("representation/rankme_per_dim", rankme)
except Exception as exc:
    warnings.warn(
        f"[RepresentationQualityCallback] Error computing representation quality: {exc}. "
        "Training will continue without representation metrics for this batch.",
        RuntimeWarning
    )
```

**Benefits**:
- ✅ **Training safety**: Monitoring failures don't crash training
- ✅ **Graceful degradation**: Missing metrics → warnings, not errors
- ✅ **Debug visibility**: Clear warning messages indicate what failed
- ✅ **Robustness**: Handles edge cases (missing data, invalid shapes)

#### 2. Training Metrics Plot Callback

**New Callback**: `TrainingMetricsPlotCallback` automatically generates metric plots after training ends.

**Features**:
- Extracts metrics history from WandB logger
- Generates 4-row plot layout matching WandB dashboard
- Supports multiple formats (png, pdf, svg)
- Auto-detects output directory
- Log scale for pred_loss by default

**Configuration**:
```yaml
callbacks:
  training_metrics_plot:
    _target_: lewm.training.callbacks.TrainingMetricsPlotCallback
    plot_format: png
    dpi: 300
```

**Default Plot Rows**:
1. **Row1_TrainingHealth**: train/pred_loss, train/sigreg_loss
2. **Row2_RepresentationQuality**: representation/rankme_per_dim, representation/embedding_norm_std
3. **Row3_SystemState**: system/grad_norm, system/learning_rate
4. **Row4_EmbeddingStatistics**: embedding/mean, embedding/std

**Usage Example**:
```bash
# Default configuration
lewm-train

# Custom plot settings
lewm-train callbacks.training_metrics_plot.plot_format=pdf callbacks.training_metrics_plot.dpi=600

# Custom metrics (override in config)
callbacks:
  training_metrics_plot:
    _target_: lewm.training.callbacks.TrainingMetricsPlotCallback
    metrics_to_plot:
      CustomMetrics: ["train/pred_loss", "representation/rankme_per_dim"]
```

**Output**:
- Plots saved to `run_dir/plots/`
- Each row generates one image file
- Filename format: `RowN_CategoryName.{format}`

**Benefits**:
- ✅ **Automatic visualization**: No manual plotting after training
- ✅ **WandB dashboard correspondence**: 4-row layout matches online dashboard
- ✅ **Portable results**: PNG/PDF/SVG for reports and presentations
- ✅ **Offline access**: View metrics without WandB login

#### Modified Files

- `src/lewm/training/callbacks.py`:
  - Added defensive programming to 5 existing callbacks
  - Added `TrainingMetricsPlotCallback` class (~200 lines)
  - Total: ~150 lines added for error handling, ~200 lines for plot callback

- `src/lewm/config/train/lewm.yaml`:
  - Added `training_metrics_plot` callback configuration

- `tests/test_callbacks.py`:
  - Added `TestDefensiveProgramming`: 3 test cases
  - Added `TestTrainingMetricsPlotCallback`: 3 test cases
  - Total: 6 new test cases

- `tests/verify_defensive_and_plot.py`:
  - New comprehensive verification script

#### Testing

All tests passing (16/16):
```
tests/test_callbacks.py:
  TestDefensiveProgramming:
    - test_missing_emb_warning ✓
    - test_missing_pred_loss_warning ✓
    - test_invalid_embedding_shape_warning ✓

  TestTrainingMetricsPlotCallback:
    - test_callback_instantiation ✓
    - test_on_train_end_no_logger ✓
    - test_plot_generation_with_mock_data ✓
```

Comprehensive verification script demonstrates:
- Defensive programming handles all edge cases
- Plot callback generates 3 PNG files (Row1-Row3)
- Warnings issued for missing metrics

#### Error Handling Coverage

| Error Type | Callback Behavior | Training Impact |
|------------|-------------------|-----------------|
| Missing metric key | Warning + return early | None (skip batch) |
| Invalid tensor shape | Warning + return early | None (skip batch) |
| Exception in computation | Warning + catch | None (skip batch) |
| WandB logger unavailable | Warning + degrade | None (use fallback) |
| matplotlib unavailable | Warning + skip plots | None (training completes) |

#### Visualization Example

After training with `lewm-train`, the callback generates:

```
run_dir/plots/
├── Row1_TrainingHealth.png        (pred_loss decay curve, log scale)
├── Row2_RepresentationQuality.png (rankme evolution, embedding norms)
├── Row3_SystemState.png           (grad_norm, learning_rate schedule)
└── Row4_EmbeddingStatistics.png   (embedding statistics)
```

Each plot:
- Width: ~5 inches per subplot
- DPI: 300 (default, configurable)
- Format: PNG (default, configurable)

---

### Architecture Optimization - 2026-04-16 (v2)

Major architecture improvements for better flexibility and maintainability.

#### 1. Hydra Control Inversion for Callbacks

**Improvement**: Callbacks are now instantiated from Hydra configuration, enabling flexible customization without modifying code.

**Changes**:
- Removed hardcoded callback instantiation in `pipeline.py`
- Added `callbacks` section in `lewm.yaml` with `_target_` declarations
- Used `hydra.utils.instantiate()` to create callbacks from config
- Special handling for `model_checkpoint` callback to inject correct `run_dir`

**Benefits**:
- **Flexible addition**: Add new callbacks by editing config file only
- **Runtime customization**: Override parameters via CLI
  ```bash
  lewm-train callbacks.prediction_quality.log_interval=50
  ```
- **Easy disabling**: Remove callbacks via Hydra override
  ```bash
  lewm-train callbacks~=embedding_statistics
  ```
- **Version control**: Callback configuration is part of experiment config

**Modified Files**:
- `src/lewm/config/train/lewm.yaml`: Added `callbacks` section with all callback configs
- `src/lewm/training/pipeline.py`: Use `instantiate(cfg.callbacks)` instead of hardcoded list
- Removed explicit callback imports from `pipeline.py`

#### 2. Simplified Forward Output Dict

**Improvement**: Forward function now only exposes detached key variables; all derived statistics are computed in callbacks.

**Changes**:
- Removed derived statistics from `forward.py`: `pred_emb_norm`, `tgt_emb_norm`, `pred_error_per_dim`
- Only expose essential detached variables: `emb`, `pred_emb`, `tgt_emb`
- Updated `PredictionQualityCallback` to compute all derived stats from `pred_emb` and `tgt_emb`

**Benefits**:
- **Clean separation**: Forward only handles training logic; callbacks handle monitoring
- **No pollution**: Output dict contains only necessary training variables
- **Callback autonomy**: Each callback computes its own statistics
- **Easier extension**: New callbacks can compute custom stats from base variables

**Modified Files**:
- `src/lewm/training/forward.py`: Simplified output dict (removed 3 derived variables)
- `src/lewm/training/callbacks.py`: Updated `PredictionQualityCallback` to compute derived stats
- `tests/test_callbacks.py`: Updated tests to verify callback computes derived stats

#### Configuration Example

New callbacks configuration in `lewm.yaml`:

```yaml
callbacks:
  model_checkpoint:
    _target_: lewm.training.callbacks.ModelObjectCallBack
    dirpath: ${hydra:runtime.cwd}/${subdir}  # Will be overridden in pipeline
    filename: ${output_model_name}
    epoch_interval: 1

  representation_quality:
    _target_: lewm.training.callbacks.RepresentationQualityCallback
    log_interval: 100

  system_monitoring:
    _target_: lewm.training.callbacks.SystemMonitoringCallback
    log_interval: 50

  # ... more callbacks
```

#### Usage Examples

**Default configuration**:
```bash
lewm-train
```

**Customize callback frequency**:
```bash
lewm-train callbacks.prediction_quality.log_interval=25
```

**Add custom callback** (future):
```yaml
callbacks:
  custom_monitor:
    _target_: my_package.callbacks.CustomCallback
    custom_param: 42
```

**Disable specific callback**:
```bash
lewm-train 'callbacks~=embedding_statistics'
```

#### Testing

All tests updated and passing:
- Unit tests verify callback computes derived statistics
- Integration tests verify Hydra instantiate works correctly
- Forward output dict simplified without breaking functionality

---

## [Unreleased] - 2026-04-16 (v1)

### Added - Training Monitoring System

Comprehensive training monitoring system using Lightning callbacks. All monitoring is implemented via callbacks to avoid polluting the main training path.

#### New Callbacks (`src/lewm/training/callbacks.py`)

1. **RepresentationQualityCallback**
   - Computes `rankme_per_dim` (effective rank / embedding dimension)
   - Tracks embedding L2 norm standard deviation
   - Logs embedding dimension from config
   - Interval: configurable via `monitoring.representation_interval` (default: 100 steps)

2. **SystemMonitoringCallback**
   - Monitors global gradient norm (`system/grad_norm`)
   - Logs learning rate at each step and epoch
   - Interval: configurable via `monitoring.system_interval` (default: 50 steps)

3. **EmbeddingStatisticsCallback**
   - Detailed embedding statistics: mean, std, max, min
   - Temporal cosine similarity across time dimension
   - Interval: configurable via `monitoring.embedding_interval` (default: 200 steps)

4. **PredictionQualityCallback**
   - Prediction loss per sample
   - Prediction embedding statistics
   - Interval: configurable via `monitoring.prediction_interval` (default: 100 steps)

5. **WandBSummaryCallback**
   - Updates WandB run summary with final metrics
   - Records best/final performance at training end

#### Modified Files

- **`src/lewm/training/callbacks.py`**: Added 5 new monitoring callbacks
- **`src/lewm/training/forward.py`**: Exposed intermediate variables (`emb`, `pred_emb`, `tgt_emb`) for monitoring
- **`src/lewm/training/pipeline.py`**: Integrated all callbacks into training pipeline
- **`src/lewm/config/train/lewm.yaml`**: Added `monitoring` section for callback intervals

#### New Documentation

- **`docs/wandb_dashboard_guide.md`**: Comprehensive WandB dashboard configuration guide
  - Recommended 4-row dashboard layout
  - Metric interpretation guidelines (RankMe, gradient health, loss balance)
  - Guardrail rules for representation collapse detection
  - Integration with BraTS experimental insights

- **`examples/monitoring_example.py`**: Quick example for using monitoring system

- **`tests/test_callbacks.py`**: Unit tests for all callbacks (9 test cases, all passing)

#### Configuration

Monitoring intervals are configurable via Hydra:

```bash
lewm-train monitoring.representation_interval=50 monitoring.system_interval=25
```

Default configuration in `src/lewm/config/train/lewm.yaml`:

```yaml
monitoring:
  representation_interval: 100   # 表征质量计算频率（steps）
  system_interval: 50            # 梯度/学习率记录频率（steps）
  embedding_interval: 200        # embedding详细统计频率（steps）
  prediction_interval: 100       # 预测质量统计频率（steps）
```

#### Key Metrics

**Row 1: Training Health**
- `train/pred_loss` (log scale)
- `train/sigreg_loss`
- `val/pred_loss`

**Row 2: Representation Quality (Research Core)**
- `representation/rankme_per_dim` — critical for detecting representation collapse
- `representation/embedding_dim`
- `representation/embedding_norm_std`

**Row 3: System State**
- `system/grad_norm`
- `system/learning_rate`

**Row 4: Embedding Statistics (Optional)**
- `embedding/mean`, `embedding/std`, `embedding/max`, `embedding/min`
- `embedding/temporal_cosine_sim_mean`

#### Guardrails

The monitoring system includes guardrails based on BraTS experimental insights:

- **RankMe guardrail**: Reject `rankme_per_dim` gains caused only by smaller `embed_dim`
  - Reference: [BraTS RankMe guardrail](feedback_rankme_hacking.md) in project memory
  - Must monitor absolute rankme (rankme_per_dim × embed_dim) when changing embed_dim

#### Testing

All callbacks are tested with unit tests (`tests/test_callbacks.py`):
- 9 test cases covering all 5 callbacks
- Mock Lightning module for isolated testing
- RankMe computation validation
- Metric logging verification
- Interval enforcement checks

All tests passing (verified on 2026-04-16).

---

## Previous Releases

See git history for prior changes.
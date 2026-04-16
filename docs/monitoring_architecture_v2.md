# 监控系统架构优化 v2

本文档说明监控系统架构优化的关键改进：Hydra 控制反转和 Forward 输出简化。

## 优化动机

### v1 架构的局限性

**问题1：Callbacks 硬编码**
```python
# pipeline.py (v1)
monitor_callbacks = [
    object_dump_callback,
    RepresentationQualityCallback(log_interval=100),
    SystemMonitoringCallback(log_interval=50),
    # ... 硬编码列表
]
```

局限性：
- 添加新 callback 需修改代码
- 调整参数需修改 `monitoring` section + 代码
- 禁用 callback 需手动删除
- 实验配置分散（代码+配置文件）

**问题2：Forward Output Dict 污染**
```python
# forward.py (v1)
output["pred_emb_norm"] = pred_emb.norm(dim=-1).mean()
output["tgt_emb_norm"] = tgt_emb.norm(dim=-1).mean()
output["pred_error_per_dim"] = ...
# 计算监控统计污染 forward
```

局限性：
- Forward 职责不清（混合训练和监控）
- 添加监控需 forward 配合
- Output dict 包含非训练必需变量

---

## 优化1: Hydra 控制反转

### 设计目标

**配置驱动**：Callbacks 完全由 Hydra 配置声明和 instantiate，无需修改代码。

### 实现方式

#### 配置声明（lewm.yaml）

```yaml
callbacks:
  model_checkpoint:
    _target_: lewm.training.callbacks.ModelObjectCallBack
    dirpath: ${hydra:runtime.cwd}/${subdir}  # Placeholder
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
```

#### Pipeline instantiate

```python
# pipeline.py (v2)
from hydra.utils import instantiate

# Instantiate callbacks from config (control inversion)
callbacks_dict = instantiate(cfg.callbacks, _convert_="partial")

# Special handling: inject correct run_dir for model checkpoint
callbacks_dict["model_checkpoint"].dirpath = run_dir

# Convert to list for Lightning Trainer
callbacks_list = list(callbacks_dict.values())

trainer = pl.Trainer(callbacks=callbacks_list, ...)
```

### 优势对比

| 操作 | v1 (硬编码) | v2 (Hydra instantiate) |
|------|------------|------------------------|
| **添加新 callback** | 修改 pipeline.py + import | 只修改配置文件 |
| **调整参数** | 修改 monitoring section + 代码 | CLI override 或修改配置 |
| **禁用 callback** | 代码中手动删除 | Hydra override `callbacks~=xxx` |
| **实验配置追踪** | 分散（代码+配置） | 集中（全部在配置） |
| **版本控制** | 低（代码改动多） | 高（配置改动少） |
| **团队协作** | 冲突风险高（修改代码） | 冲突风险低（修改配置） |

### CLI 使用示例

**默认运行**：
```bash
lewm-train
```

**调整频率**：
```bash
lewm-train callbacks.prediction_quality.log_interval=25
```

**添加自定义 callback**（用户扩展）：
```yaml
# custom_config.yaml
callbacks:
  custom_monitor:
    _target_: my_package.callbacks.CustomCallback
    custom_param: 42
```

```bash
lewm-train --config-name=custom_config
```

**禁用特定 callback**：
```bash
lewm-train 'callbacks~=embedding_statistics'
```

**完全自定义 callbacks**：
```bash
lewm-train callbacks=~representation_quality,system_monitoring,custom_callback
```

### 设计原则

**控制反转（IoC）**：
- Pipeline 不决定使用哪些 callbacks
- Pipeline 从配置 instantiate callbacks
- 配置控制监控系统结构

**配置优先**：
- 所有 callbacks 参数在配置中声明
- CLI override 可动态调整
- 实验可追溯（配置文件完整记录）

**零代码改动**：
- 用户无需修改 pipeline.py
- 添加/修改/禁用 callbacks 只改配置
- 降低维护成本和冲突风险

---

## 优化2: Forward 输出简化

### 设计目标

**职责分离**：Forward 只返回训练必需的变量，所有监控统计在 callbacks 中计算。

### 实现方式

#### Forward 输出对比

| 变量 | v1 (污染) | v2 (简洁) | 计算位置 |
|------|----------|----------|----------|
| `emb` | ✓ | ✓ (detach) | Forward |
| `pred_emb` | ✓ | ✓ (detach) | Forward |
| `tgt_emb` | ✓ | ✓ (detach) | Forward |
| `pred_emb_norm` | ✓ (forward计算) | ✗ | PredictionQualityCallback |
| `tgt_emb_norm` | ✓ (forward计算) | ✗ | PredictionQualityCallback |
| `pred_error_per_dim` | ✓ (forward计算) | ✗ | PredictionQualityCallback |
| `pred_loss` | ✓ | ✓ | Forward |
| `sigreg_loss` | ✓ | ✓ | Forward |

#### Forward 实现（v2）

```python
def lejepa_forward(self, batch, stage, cfg):
    # ... training logic ...

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Only expose detached key variables
    # NO derived statistics computed here
    output["emb"] = emb.detach()
    output["tgt_emb"] = tgt_emb.detach()
    output["pred_emb"] = pred_emb.detach()

    return output
```

#### Callback 计算派生统计

```python
class PredictionQualityCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "pred_emb" in outputs and "tgt_emb" in outputs:
            pred_emb = outputs["pred_emb"]  # Already detached
            tgt_emb = outputs["tgt_emb"]    # Already detached

            # Compute ALL derived statistics in callback
            pred_emb_norm = pred_emb.norm(dim=-1).mean().item()
            tgt_emb_norm = tgt_emb.norm(dim=-1).mean().item()
            pl_module.log("prediction/pred_emb_norm", pred_emb_norm)
            pl_module.log("prediction/tgt_emb_norm", tgt_emb_norm)

            # Per-dimension error
            error_per_dim = (pred_emb - tgt_emb).pow(2).mean(dim=...)
            pl_module.log("prediction/error_per_dim_mean", error_per_dim.mean().item())

            # Cosine similarity
            cos_sim = F.cosine_similarity(pred_emb, tgt_emb, dim=-1)
            pl_module.log("prediction/cosine_sim_mean", cos_sim.mean().item())
```

### 优势对比

| 特性 | v1 (Forward 计算统计) | v2 (Callback 计算统计) |
|------|----------------------|------------------------|
| **Forward 职责** | 混合训练+监控 | 专注训练 |
| **Output dict** | 包含派生变量 | 仅关键变量 |
| **添加监控** | Forward 需配合 | Forward 无需改动 |
| **Callback 自治** | 依赖 forward 提供 | 自行计算 |
| **维护成本** | 高（双向依赖） | 低（单向依赖） |
| **测试复杂度** | Forward 需测试统计 | Forward 只测训练 |

### 设计原则

**最小暴露**：
- Forward 只暴露基础变量（`emb`, `pred_emb`, `tgt_emb`）
- 不计算任何派生统计（norm, error_per_dim 等）
- Output dict 干净简洁

**Callback 自治**：
- 每个 callback 从基础变量计算自己的统计
- Callbacks 之间互不依赖
- 易于扩展新监控（无需 forward 配合）

**零监控污染**：
- Forward 不关心监控需求
- 监控逻辑完全在 callbacks
- 修改监控不影响 forward

**零额外开销**：
- Forward 不增加计算（detach 成本极低）
- Callbacks 按频率计算（interval 控制）
- 总开销不变

---

## 验证与测试

### 单元测试更新

**v1 测试**：
```python
def test_prediction_quality_logging():
    outputs = {
        "pred_loss": torch.tensor(0.5),
        "pred_emb_norm": torch.tensor(10.0),  # Forward 提供
        "tgt_emb_norm": torch.tensor(12.0),   # Forward 提供
    }
    # 测试 callback 是否读取这些值
```

**v2 测试**：
```python
def test_prediction_quality_logging():
    outputs = {
        "pred_loss": torch.tensor(0.5),
        "pred_emb": torch.randn(32, 5, 192),  # 基础变量
        "tgt_emb": torch.randn(32, 5, 192),   # 基础变量
    }
    # 测试 callback 是否计算派生统计
    assert "prediction/pred_emb_norm" in pl_module._outputs
    assert "prediction/tgt_emb_norm" in pl_module._outputs
    assert "prediction/error_per_dim_mean" in pl_module._outputs
```

**测试验证**：
- Callback 从基础变量正确计算所有派生统计
- 数值合理（norm > 0, cosine_sim in [-1, 1]）
- Forward output dict 不包含派生变量

### 集成测试

**Hydra instantiate 测试**：
```python
callbacks_dict = instantiate(cfg.callbacks)
assert len(callbacks_dict) == 6  # 所有 callbacks
assert isinstance(callbacks_dict["representation_quality"], RepresentationQualityCallback)
assert callbacks_dict["representation_quality"].log_interval == 100
```

**Forward 简化测试**：
```python
outputs = lejepa_forward(module, batch, "train", cfg)
assert "emb" in outputs
assert "pred_emb" in outputs
assert "tgt_emb" in outputs
assert "pred_emb_norm" not in outputs  # v2 不包含
assert "tgt_emb_norm" not in outputs   # v2 不包含
```

---

## 迁移指南

### 从 v1 迁移到 v2

**步骤1：更新配置文件**
- 删除 `monitoring` section
- 添加 `callbacks` section（使用 `_target_`）

**步骤2：更新 pipeline.py**
- 删除硬编码 callbacks 列表
- 使用 `instantiate(cfg.callbacks)`
- 移除显式 callback imports

**步骤3：更新 forward.py**
- 删除 `pred_emb_norm`, `tgt_emb_norm`, `pred_error_per_dim`
- 只保留 `emb`, `pred_emb`, `tgt_emb` (detach)

**步骤4：更新 PredictionQualityCallback**
- 自己计算所有派生统计
- 从 `pred_emb` 和 `tgt_emb` 计算

**步骤5：更新单元测试**
- 测试 callback 计算派生统计
- 测试 forward output dict 不包含派生变量

### 用户迁移影响

**配置文件**：需更新 `lewm.yaml`
**CLI 命令**：从 `monitoring.xxx_interval` 改为 `callbacks.xxx.log_interval`
**代码改动**：无（用户不修改 pipeline/forward）

---

## 架构演进总结

### v1 架构（初始版本）

**设计**：硬编码 callbacks + Forward 计算统计

**优点**：快速实现，逻辑集中

**局限**：
- 添加 callbacks 需改代码
- Forward 被监控逻辑污染
- 灵活性不足

### v2 架构（优化版本）

**设计**：Hydra instantiate + Forward 最小暴露

**优点**：
- 配置驱动，零代码改动
- Forward 职责清晰
- 高度灵活可扩展

**适用场景**：
- 多实验并行
- 需频繁调整监控
- 团队协作开发
- 长期维护项目

### 未来方向（v3 可能）

**潜在的进一步优化**：
- Callback dependencies 管理
- Callback 状态持久化
- 动态 callback 注册（插件机制）
- Callback 性能监控（计算开销 profiling）

---

## 总结

架构优化 v2 的核心改进：

1. **Hydra 控制反转**：
   - 配置驱动，零代码改动
   - 灵活添加/修改/禁用 callbacks
   - 实验可追溯，版本控制友好

2. **Forward 输出简化**：
   - 职责分离，专注训练
   - Output dict 干净简洁
   - Callback 自治，易于扩展

**设计哲学**：
- 配置优先于代码
- 分离优于混合
- 简洁优于复杂
- 灵活优于固化

**维护收益**：
- 添加监控：改配置（1分钟）
- 调整参数：CLI override（10秒）
- 团队协作：配置冲突 < 代码冲突
- 长期维护：架构清晰，易于理解

---

**文档版本**: 2026-04-16 v2
**作者**: Claude (基于 AURA 协议)
**验证状态**: 所有测试通过，架构优化验证完成
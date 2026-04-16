# WandB Dashboard 配置指南

本文档说明如何配置 WandB dashboard 以可视化 LeWM 训练监控系统。

## 推荐的 Dashboard 布局

训练监控系统自动记录以下指标，建议按以下方式组织 WandB panels：

### Row 1: 基础训练健康监控

**目的：快速判断训练是否正常进行**

1. **train/pred_loss (log scale)**
   - Panel类型: Line Plot
   - Y轴: Log scale
   - 显示训练预测损失趋势
   - 关键观察：是否平滑下降，有无异常震荡

2. **train/sigreg_loss**
   - Panel类型: Line Plot
   - 显示 SIGReg 正则化损失
   - 关键观察：是否稳定，避免过大（表征坍塌）或过小（失去约束）

3. **val/pred_loss** (如果启用validation)
   - Panel类型: Line Plot
   - 显示验证集预测损失
   - 关键观察：训练集与验证集gap，判断过拟合

### Row 2: 表征质量监控（研究核心）

**目的：验证表征学习质量，避免坍塌和退化**

1. **representation/rankme_per_dim vs epoch**
   - Panel类型: Line Plot
   - X轴: Epoch 或 Step
   - Y轴: rankme_per_dim (范围0-1)
   - **关键指标**：
     - 接近 1.0: 表征充分利用所有维度（好）
     - 接近 0.0: 表征坍塌到低维子空间（坏）
     - **Guardrail**: 如果 embed_dim 减小导致 rankme_per_dim 增加，这不应被视为改进
     - 参考 BraTS 实验中的 guardrail 规则

2. **representation/embedding_dim**
   - Panel类型: Scalar
   - 显示当前 embedding 维度配置
   - 用于解释 rankme_per_dim 的变化

3. **representation/embedding_norm_std**
   - Panel类型: Line Plot
   - 显示 embedding L2 范数的标准差
   - 关键观察：范数分布的均匀性

4. **embedding/temporal_cosine_sim_mean** (可选)
   - Panel类型: Line Plot
   - 显示时间维度上的 embedding 余弦相似度
   - 关键观察：embedding 是否随时间有合理变化

### Row 3: 系统状态监控

**目的：诊断训练稳定性问题**

1. **system/grad_norm**
   - Panel类型: Line Plot
   - 显示全局梯度范数
   - 关键观察：
     - 是否爆炸（> 1000）
     - 是否消失（< 1e-6）
     - 是否在合理范围内震荡

2. **system/learning_rate**
   - Panel类型: Line Plot
   - 显示学习率调度曲线（cosine annealing）
   - 用于理解训练动态

### Row 4: Embedding 详细统计（可选）

**目的：深入理解表征分布**

1. **embedding/mean, embedding/std**
   - Panel类型: Line Plot (多条曲线)
   - 显示 embedding 的均值和标准差

2. **embedding/max, embedding/min**
   - Panel类型: Line Plot
   - 显示 embedding 的范围

### Row 5: 预测质量（可选）

**目的：理解预测误差分布**

1. **prediction/loss_per_sample**
   - Panel类型: Line Plot
   - 显示每个样本的预测损失

2. **prediction/pred_emb_norm, prediction/tgt_emb_norm**
   - Panel类型: Line Plot
   - 对比预测 embedding 和目标 embedding 的范数

## WandB 配置步骤

### 1. 创建自定义 Dashboard

在 WandB 项目页面：

1. 点击 "Add Panel" → "Custom Panel"
2. 选择 "Line Plot" 或其他类型
3. 选择要可视化的指标（上述列表）
4. 按推荐 Rows 组织布局

### 2. 保存 Dashboard 配置

创建好布局后：

1. 点击 "Save View" 按钮
2. 命名 dashboard（例如 "Training Monitor - LeWM"）
3. 设置为默认视图或团队共享视图

### 3. 使用 WandB Panel Templates

可以使用 WandB 的 panel templates 功能：

1. 创建 panel template 文件（JSON格式）
2. 在 WandB 设置中导入
3. 所有实验自动应用该 dashboard 布局

## 关键指标解读指南

### RankMe Interpretation

| rankme_per_dim 值 | 表征状态 | 可能原因 | 建议行动 |
|-------------------|----------|----------|----------|
| > 0.9 | 优秀 | 表征充分利用维度 | 继续训练 |
| 0.7-0.9 | 良好 | 表征有合理的有效秩 | 监控是否稳定 |
| 0.5-0.7 | 中等 | 表征部分退化 | 检查 SIGReg weight |
| < 0.5 | 坍塌风险 | 表征坍塌到低维 | 增加 SIGReg weight 或 embed_dim |

**重要警告**: 如果 embed_dim 从 192 减小到 64，rankme_per_dim 可能从 0.6 上升到 0.9。
这**不是真正的改进**，而是维度减少导致的有效秩占比增加。
**必须同时查看 absolute rankme (rankme_per_dim * embed_dim)**。

参考 memory 中的 guardrail 规则：
- [BraTS RankMe guardrail](feedback_rankme_hacking.md)

### Gradient Health

| grad_norm 范围 | 训练状态 | 建议 |
|----------------|----------|------|
| < 1e-6 | 梯度消失 | 检查模型初始化，增加学习率 |
| 1e-3 - 10 | 正常 | 继续训练 |
| 10 - 100 | 较大但可接受 | 监控是否收敛 |
| > 1000 | 梯度爆炸 | 减小学习率，检查梯度裁剪 |

### Prediction vs SIGReg Balance

观察 train/pred_loss 和 train/sigreg_loss 的比例：

- **理想**: pred_loss 下降，sigreg_loss 稳定在合理范围
- **问题1**: sigreg_loss 过大 → 表征被过度约束，可能降低表达能力
- **问题2**: sigreg_loss 过小 → 表征缺乏约束，可能坍塌

建议调整 `loss.sigreg.weight` 参数。

## 实验间对比

WandB 自动支持多实验对比：

1. 在 Runs selector 中选择多个 runs
2. 对比关键指标（rankme_per_dim, pred_loss）
3. 查看不同配置（embed_dim, SIGReg weight）的影响

## 实验建议

基于监控系统的最佳实践：

1. **每 100 steps**: 检查 rankme_per_dim，确保表征不坍塌
2. **每 epoch**: 检查 pred_loss 收敛情况
3. **每 50 steps**: 监控 grad_norm，诊断稳定性问题
4. **训练结束**: 查看 WandB summary 中的 final metrics
5. **超参调整**: 基于 rankme 和 pred_loss 的权衡调整 SIGReg weight

## 下游验证集成（未来工作）

当前监控系统专注于训练过程。
未来可添加：

- **Periodic checkpoint evaluation**: 每 N epochs 在评估集上测试 planning success rate
- **Planning cost convergence curve**: 记录规划过程中 cost 的收敛轨迹

这些需要在 `src/lewm/evaluation/pipeline.py` 中集成 callback，
并通过 Hydra config 控制评估频率。

## 配置文件

监控参数在 `src/lewm/config/train/lewm.yaml` 中：

```yaml
monitoring:
  representation_interval: 100   # 表征质量计算频率（steps）
  system_interval: 50            # 梯度/学习率记录频率（steps）
  embedding_interval: 200        # embedding详细统计频率（steps）
  prediction_interval: 100       # 预测质量统计频率（steps）
```

可以通过 Hydra override 调整：

```bash
lewm-train monitoring.representation_interval=50 monitoring.system_interval=25
```

## 相关文件

- Callbacks 实现: `src/lewm/training/callbacks.py`
- Forward 函数: `src/lewm/training/forward.py`
- Pipeline 集成: `src/lewm/training/pipeline.py`
- 配置文件: `src/lewm/config/train/lewm.yaml`
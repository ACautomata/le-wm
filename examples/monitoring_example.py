"""
快速示例：如何在训练中使用监控系统

这个脚本展示如何启动带有完整监控的训练。
"""

# 方式1: 使用默认监控配置（推荐）
# 默认配置已经在 src/lewm/config/train/lewm.yaml 中设置
# 直接运行即可启用所有监控 callbacks

# 命令:
# lewm-train

# 方式2: 自定义监控频率
# 通过 Hydra override 调整监控间隔

# 命令示例:
# lewm-train monitoring.representation_interval=50 monitoring.system_interval=25

# 方式3: 在自定义训练脚本中使用
from lewm.training.pipeline import build_training_manager
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load("src/lewm/config/train/lewm.yaml")

# 自定义监控参数
with OmegaConf.open_dict(cfg):
    cfg.monitoring.representation_interval = 50  # 更频繁的表征监控
    cfg.monitoring.system_interval = 25          # 更频繁的系统监控

# 构建训练 manager（自动包含所有 callbacks）
manager = build_training_manager(cfg)

# 开始训练
manager.train()

# 训练完成后，在 WandB 中查看监控 dashboard:
# 1. 打开项目页面: wandb.ai/lewm/lewm
# 2. 查看以下关键指标:
#    - train/pred_loss (log scale)
#    - representation/rankme_per_dim
#    - system/grad_norm
#    - system/learning_rate

# 关键监控点:
# - RankMe: 应该保持在 0.5-1.0 范围，避免表征坍塌
# - Gradient norm: 应该在合理范围，避免爆炸或消失
# - Pred loss: 应该平滑下降

# 详细配置指南见: docs/wandb_dashboard_guide.md
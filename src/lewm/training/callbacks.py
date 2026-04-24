from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
import numpy as np
import warnings


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as exc:
            warnings.warn(f"Error saving model object: {exc}. Training will continue.", RuntimeWarning)


class RepresentationQualityCallback(Callback):
    """监控表征质量：rankme_per_dim, embedding norms, embedding_dim"""

    def __init__(self, log_interval: int = 100, embed_dim_key: str = "wm.embed_dim"):
        super().__init__()
        self.log_interval = log_interval
        self.embed_dim_key = embed_dim_key

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练batch结束后计算表征质量指标"""
        if not trainer.is_global_zero:
            return
        if batch_idx % self.log_interval != 0:
            return

        # 防御性编程：验证依赖指标存在
        if "emb" not in outputs:
            warnings.warn(
                f"[RepresentationQualityCallback] 'emb' not found in outputs at batch {batch_idx}. "
                "Skipping representation quality monitoring for this batch.",
                RuntimeWarning
            )
            return

        try:
            # 获取 embeddings
            emb = outputs["emb"]  # shape: [batch, time, dim]

            # 验证 emb 形状合理
            if emb.dim() < 2 or emb.size(-1) == 0:
                warnings.warn(
                    f"[RepresentationQualityCallback] Invalid embedding shape: {emb.shape}. "
                    "Skipping this batch.",
                    RuntimeWarning
                )
                return

            emb_flat = emb.detach().view(-1, emb.size(-1))  # [batch*time, dim]

            # 计算 rankme_per_dim
            rankme = self._compute_rankme(emb_flat)
            pl_module.log("representation/rankme_per_dim", rankme, on_step=True)

            # 计算 embedding L2 norm std
            norms = emb_flat.norm(dim=-1)  # [batch*time]
            norm_std = norms.std()
            pl_module.log("representation/embedding_norm_std", norm_std, on_step=True)

            # 记录 embedding_dim (from config)
            embed_dim = emb.size(-1)
            pl_module.log("representation/embedding_dim", embed_dim, on_step=False, on_epoch=True)

        except Exception as exc:
            warnings.warn(
                f"[RepresentationQualityCallback] Error computing representation quality: {exc}. "
                "Training will continue without representation metrics for this batch.",
                RuntimeWarning
            )

    def _compute_rankme(self, embeddings):
        """
        计算 RankMe 指标 (有效秩 / embedding维度)
        参考: https://arxiv.org/abs/2103.01714
        """
        # 计算 embedding matrix 的奇异值
        with torch.no_grad():
            # SVD 不支持 BFloat16，转换为 float32
            embeddings_float32 = embeddings.float()

            # SVD decomposition
            U, S, V = torch.linalg.svd(embeddings_float32, full_matrices=False)

            # 归一化奇异值
            S_normalized = S / S.sum()

            # 计算有效秩: rank_eff = exp(H) where H = -sum(p_i * log(p_i))
            # 避免数值不稳定
            S_normalized = torch.clamp(S_normalized, min=1e-10)
            entropy = -(S_normalized * torch.log(S_normalized)).sum()
            rank_eff = torch.exp(entropy)

            # rankme = rank_eff / embed_dim
            embed_dim = embeddings.size(-1)
            rankme = rank_eff / embed_dim

            return rankme.item()


class SystemMonitoringCallback(Callback):
    """监控系统状态：gradient norm, learning rate"""

    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """监控梯度和学习率"""
        if not trainer.is_global_zero:
            return
        if batch_idx % self.log_interval != 0:
            return

        try:
            # 计算梯度范数
            total_norm = 0.0
            for p in pl_module.parameters(with_callbacks=False):
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            pl_module.log("system/grad_norm", total_norm, on_step=True)

            # 记录当前学习率
            optimizers = trainer.optimizers
            if optimizers:
                lr = optimizers[0].param_groups[0]['lr']
                pl_module.log("system/learning_rate", lr, on_step=True)

        except Exception as exc:
            warnings.warn(
                f"[SystemMonitoringCallback] Error monitoring system metrics: {exc}. "
                "Training will continue without system metrics for this batch.",
                RuntimeWarning
            )

    def on_train_epoch_start(self, trainer, pl_module):
        """Epoch开始时也记录学习率"""
        if not trainer.is_global_zero:
            return
        try:
            optimizers = trainer.optimizers
            if optimizers:
                lr = optimizers[0].param_groups[0]['lr']
                pl_module.log("system/learning_rate_epoch", lr, on_epoch=True)
        except Exception as exc:
            warnings.warn(
                f"[SystemMonitoringCallback] Error logging epoch learning rate: {exc}.",
                RuntimeWarning
            )


class EmbeddingStatisticsCallback(Callback):
    """详细的embedding统计：均值、方差、余弦相似度等"""

    def __init__(self, log_interval: int = 200):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """详细的embedding统计信息"""
        if not trainer.is_global_zero:
            return
        if batch_idx % self.log_interval != 0:
            return

        # 防御性编程：验证依赖指标存在
        if "emb" not in outputs:
            warnings.warn(
                f"[EmbeddingStatisticsCallback] 'emb' not found in outputs at batch {batch_idx}. "
                "Skipping embedding statistics for this batch.",
                RuntimeWarning
            )
            return

        try:
            emb = outputs["emb"].detach()  # [batch, time, dim]

            # 验证 emb 形状合理
            if emb.dim() < 2:
                warnings.warn(
                    f"[EmbeddingStatisticsCallback] Invalid embedding shape: {emb.shape}. "
                    "Skipping this batch.",
                    RuntimeWarning
                )
                return

            # Embedding 维度统计
            emb_flat = emb.view(-1, emb.size(-1))
            pl_module.log("embedding/mean", emb_flat.mean().item(), on_step=True)
            pl_module.log("embedding/std", emb_flat.std().item(), on_step=True)
            pl_module.log("embedding/max", emb_flat.max().item(), on_step=True)
            pl_module.log("embedding/min", emb_flat.min().item(), on_step=True)

            # 时间维度上的embedding变化
            if emb.size(1) > 1:
                emb_first = emb[:, 0, :]  # 第一帧
                emb_last = emb[:, -1, :]  # 最后一帧

                # 计算时间跨度上的余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(emb_first, emb_last, dim=-1)
                pl_module.log("embedding/temporal_cosine_sim_mean", cos_sim.mean().item(), on_step=True)
                pl_module.log("embedding/temporal_cosine_sim_std", cos_sim.std().item(), on_step=True)

        except Exception as exc:
            warnings.warn(
                f"[EmbeddingStatisticsCallback] Error computing embedding statistics: {exc}. "
                "Training will continue without embedding statistics for this batch.",
                RuntimeWarning
            )


class PredictionQualityCallback(Callback):
    """监控预测质量：预测误差分布、预测embedding统计"""

    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """预测质量监控"""
        if not trainer.is_global_zero:
            return
        if batch_idx % self.log_interval != 0:
            return

        # 防御性编程：验证依赖指标存在
        if "pred_loss" not in outputs:
            warnings.warn(
                f"[PredictionQualityCallback] 'pred_loss' not found in outputs at batch {batch_idx}. "
                "Skipping prediction quality monitoring for this batch.",
                RuntimeWarning
            )
            return

        try:
            # 记录预测误差
            pred_loss = outputs["pred_loss"].detach()
            pl_module.log("prediction/loss_per_sample", pred_loss.item(), on_step=True)

            # 从 pred_emb 和 tgt_emb 计算派生统计（不要依赖 forward 中暴露）
            if "pred_emb" in outputs and "tgt_emb" in outputs:
                pred_emb = outputs["pred_emb"]  # 已经 detached
                tgt_emb = outputs["tgt_emb"]  # 已经 detached

                # 验证形状合理
                if pred_emb.dim() < 2 or tgt_emb.dim() < 2:
                    warnings.warn(
                        f"[PredictionQualityCallback] Invalid embedding shapes: pred_emb={pred_emb.shape}, tgt_emb={tgt_emb.shape}. "
                        "Skipping derived statistics.",
                        RuntimeWarning
                    )
                    return

                # 计算预测和目标 embedding 范数对比
                pred_emb_norm = pred_emb.norm(dim=-1).mean().item()
                tgt_emb_norm = tgt_emb.norm(dim=-1).mean().item()
                pl_module.log("prediction/pred_emb_norm", pred_emb_norm, on_step=True)
                pl_module.log("prediction/tgt_emb_norm", tgt_emb_norm, on_step=True)

                # 计算每个维度的预测误差（检测是否有维度坍塌）
                pred_error_per_dim = (pred_emb - tgt_emb).pow(2)
                # 按维度平均（batch 和 time 维度）
                error_mean_dims = list(range(pred_error_per_dim.ndim - 1))
                error_per_dim = pred_error_per_dim.mean(dim=error_mean_dims)
                pl_module.log("prediction/error_per_dim_mean", error_per_dim.mean().item(), on_step=True)
                pl_module.log("prediction/error_per_dim_max", error_per_dim.max().item(), on_step=True)
                pl_module.log("prediction/error_per_dim_min", error_per_dim.min().item(), on_step=True)

                # 预测与目标的余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(pred_emb, tgt_emb, dim=-1)
                pl_module.log("prediction/cosine_sim_mean", cos_sim.mean().item(), on_step=True)

        except Exception as exc:
            warnings.warn(
                f"[PredictionQualityCallback] Error computing prediction quality: {exc}. "
                "Training will continue without prediction quality metrics for this batch.",
                RuntimeWarning
            )


class WandBSummaryCallback(Callback):
    """WandB特定的summary记录：将关键指标写入WandB summary"""

    def on_train_epoch_end(self, trainer, pl_module):
        """每个epoch结束时更新WandB summary"""
        if not trainer.is_global_zero:
            return

        # 防御性编程：验证 logger 存在
        if trainer.logger is None or not hasattr(trainer.logger, 'experiment'):
            warnings.warn(
                "[WandBSummaryCallback] WandB logger not available. Skipping summary update.",
                RuntimeWarning
            )
            return

        try:
            # 获取wandb实验对象
            wandb_run = trainer.logger.experiment

            # 记录当前epoch的最佳指标到summary
            metrics = trainer.callback_metrics

            # 记录最终模型性能到summary
            if trainer.current_epoch == trainer.max_epochs - 1:
                if "train/pred_loss" in metrics:
                    wandb_run.summary["final_train_pred_loss"] = metrics["train/pred_loss"].item()
                if "train/sigreg_loss" in metrics:
                    wandb_run.summary["final_train_sigreg_loss"] = metrics["train/sigreg_loss"].item()

            # 记录最佳性能
            if "representation/rankme_per_dim" in metrics:
                current_rankme = metrics["representation/rankme_per_dim"].item()
                wandb_run.summary["current_rankme"] = current_rankme

        except Exception as exc:
            warnings.warn(
                f"[WandBSummaryCallback] Error updating WandB summary: {exc}. "
                "Training will continue.",
                RuntimeWarning
            )


class TrainingMetricsPlotCallback(Callback):
    """
    训练结束后绘制监控指标的折线图。

    从 WandB 或 Lightning logger 中提取关键指标的历史数据并绘制。
    """

    def __init__(
        self,
        output_dir: str = None,
        metrics_to_plot: dict = None,
        plot_format: str = "png",
        dpi: int = 300,
    ):
        """
        Args:
            output_dir: 图片保存目录（默认为 run_dir/plots）
            metrics_to_plot: 要绘制的指标配置
                格式：{"row_name": ["metric1", "metric2", ...]}
                默认绘制4行：训练健康、表征质量、系统状态、embedding统计
            plot_format: 图片格式（png, pdf, svg）
            dpi: 图片分辨率
        """
        super().__init__()
        self.output_dir = Path(output_dir) if output_dir else None
        self.plot_format = plot_format
        self.dpi = dpi

        # 默认绘图配置：对应 WandB dashboard 的4行布局
        self.metrics_to_plot = metrics_to_plot or {
            "Row1_TrainingHealth": [
                "train/pred_loss",
                "train/sigreg_loss",
            ],
            "Row2_RepresentationQuality": [
                "representation/rankme_per_dim",
                "representation/embedding_norm_std",
            ],
            "Row3_SystemState": [
                "system/grad_norm",
                "system/learning_rate",
            ],
            "Row4_EmbeddingStatistics": [
                "embedding/mean",
                "embedding/std",
            ],
        }

    def on_train_end(self, trainer, pl_module):
        """训练结束后绘制metrics折线图"""
        if not trainer.is_global_zero:
            return

        # 设置输出目录
        if self.output_dir is None:
            # 默认保存到 run_dir/plots
            run_dir = Path(trainer.logger.experiment.dir if trainer.logger else ".")
            self.output_dir = run_dir / "plots"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 尝试从不同数据源获取metrics历史
        metrics_history = self._get_metrics_history(trainer)

        if not metrics_history:
            warnings.warn(
                "[TrainingMetricsPlotCallback] No metrics history found. "
                "Cannot generate plots.",
                RuntimeWarning
            )
            return

        # 绘制各行的折线图
        try:
            import matplotlib.pyplot as plt

            for row_name, metrics_list in self.metrics_to_plot.items():
                self._plot_metrics_row(
                    row_name, metrics_list, metrics_history, plt
                )

            print(f"\n[TrainingMetricsPlotCallback] Plots saved to: {self.output_dir}")

        except ImportError:
            warnings.warn(
                "[TrainingMetricsPlotCallback] matplotlib not available. "
                "Cannot generate plots. Install with: pip install matplotlib",
                RuntimeWarning
            )
        except Exception as exc:
            warnings.warn(
                f"[TrainingMetricsPlotCallback] Error generating plots: {exc}. "
                "Training completed but plots not saved.",
                RuntimeWarning
            )

    def _get_metrics_history(self, trainer):
        """
        从 WandB 或 Lightning logger 中获取 metrics 历史。

        Returns:
            dict: {metric_name: [(step, value), ...]}
        """
        metrics_history = {}

        # 尝试从 WandB 获取历史数据
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            try:
                wandb_run = trainer.logger.experiment

                # WandB API: run.history() 返回 pandas DataFrame
                if hasattr(wandb_run, 'history'):
                    history_df = wandb_run.history()

                    # 提取每个 metric 的历史
                    for row_name, metrics_list in self.metrics_to_plot.items():
                        for metric_name in metrics_list:
                            if metric_name in history_df.columns:
                                # 获取 (step, value) 数据
                                steps = history_df['_step'].values
                                values = history_df[metric_name].values

                                # 过滤 NaN 值
                                valid_mask = ~np.isnan(values)
                                steps_valid = steps[valid_mask]
                                values_valid = values[valid_mask]

                                metrics_history[metric_name] = list(zip(steps_valid, values_valid))

                    return metrics_history

            except Exception as exc:
                warnings.warn(
                    f"[TrainingMetricsPlotCallback] Error fetching WandB history: {exc}. "
                    "Will attempt to use Lightning callback_metrics.",
                    RuntimeWarning
                )

        # 备选方案：从 Lightning trainer.callback_metrics 获取（仅最新值）
        # 注意：这只能获取最终值，无法绘制完整历史曲线
        warnings.warn(
            "[TrainingMetricsPlotCallback] WandB history not available. "
            "Lightning callback_metrics only provides final values, not full history. "
            "Full plotting requires WandB logger.",
            RuntimeWarning
        )

        return metrics_history

    def _plot_metrics_row(self, row_name, metrics_list, metrics_history, plt):
        """绘制一行指标的折线图"""

        # 检查是否有该行的任何指标数据
        available_metrics = [m for m in metrics_list if m in metrics_history]

        if not available_metrics:
            warnings.warn(
                f"[TrainingMetricsPlotCallback] No data available for {row_name}. "
                f"Requested metrics: {metrics_list}",
                RuntimeWarning
            )
            return

        # 创建图形
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))

        if len(available_metrics) == 1:
            axes = [axes]  # 单个 subplot 时保持一致性

        for idx, metric_name in enumerate(available_metrics):
            ax = axes[idx]

            # 提取 steps 和 values
            data = metrics_history[metric_name]
            steps = [d[0] for d in data]
            values = [d[1] for d in data]

            # 绘制折线
            ax.plot(steps, values, linewidth=1.5, alpha=0.8)

            # 设置标题和标签
            # 移除前缀（如 train/, representation/）用于简洁标题
            display_name = metric_name.split('/')[-1] if '/' in metric_name else metric_name
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)

            # 网格线
            ax.grid(True, alpha=0.3)

            # 特殊处理：log scale for pred_loss
            if 'pred_loss' in metric_name and min(values) > 0:
                ax.set_yscale('log')

        # 整体标题
        fig.suptitle(row_name.replace('_', ' '), fontsize=14, fontweight='bold', y=1.02)

        # 保存图片
        filename = f"{row_name}.{self.plot_format}"
        filepath = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filepath}")

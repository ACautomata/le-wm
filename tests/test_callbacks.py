"""Tests for training monitoring callbacks."""

import pytest
import torch
import lightning as pl
import warnings
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

from lewm.training.callbacks import (
    RepresentationQualityCallback,
    SystemMonitoringCallback,
    EmbeddingStatisticsCallback,
    PredictionQualityCallback,
    WandBSummaryCallback,
    TrainingMetricsPlotCallback,
)


class MockLightningModule(pl.LightningModule):
    """Mock Lightning module for testing callbacks."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 10)
        self.sigreg = Mock()
        self._outputs = {}

    def forward(self, batch):
        return self.model(batch)

    def log(self, name, value, **kwargs):
        """Mock log method to capture logged metrics."""
        self._outputs[name] = value

    def log_dict(self, dict, **kwargs):
        """Mock log_dict method."""
        for k, v in dict.items():
            self.log(k, v)

    def parameters(self, with_callbacks=False):
        """Return mock parameters with gradients."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10) * 0.1  # Mock gradient
        return [param]


class TestRepresentationQualityCallback:
    """Test RepresentationQualityCallback."""

    def test_rankme_computation(self):
        """Test RankMe computation is correct."""
        callback = RepresentationQualityCallback(log_interval=1)

        # Create test embeddings with different rank properties
        # Full rank embeddings
        full_rank_emb = torch.randn(100, 64)

        rankme_full = callback._compute_rankme(full_rank_emb)
        assert rankme_full > 0.5  # Full rank should have high rankme

        # Low rank embeddings (simulating collapsed representation)
        low_rank_emb = torch.randn(100, 8).repeat(1, 8)  # Only 8 effective dimensions
        rankme_low = callback._compute_rankme(low_rank_emb)
        assert rankme_low < rankme_full  # Low rank should have lower rankme

    def test_callback_logs_metrics(self):
        """Test that callback logs expected metrics."""
        callback = RepresentationQualityCallback(log_interval=1)

        trainer = Mock()
        trainer.is_global_zero = True

        pl_module = MockLightningModule()

        outputs = {
            "emb": torch.randn(32, 5, 192)  # batch=32, time=5, dim=192
        }

        batch = Mock()
        batch_idx = 0

        callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        # Verify logged metrics
        assert "representation/rankme_per_dim" in pl_module._outputs
        assert "representation/embedding_norm_std" in pl_module._outputs
        assert 0 <= pl_module._outputs["representation/rankme_per_dim"] <= 1

    def test_interval_respected(self):
        """Test that log_interval is respected."""
        callback = RepresentationQualityCallback(log_interval=10)

        trainer = Mock()
        pl_module = MockLightningModule()
        outputs = {"emb": torch.randn(32, 5, 192)}

        # Should not log at batch_idx=5
        callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 5)
        assert "representation/rankme_per_dim" not in pl_module._outputs

        # Should log at batch_idx=10
        pl_module._outputs = {}
        callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 10)
        assert "representation/rankme_per_dim" in pl_module._outputs


class TestSystemMonitoringCallback:
    """Test SystemMonitoringCallback."""

    def test_gradient_norm_computation(self):
        """Test gradient norm is computed correctly."""
        callback = SystemMonitoringCallback(log_interval=1)

        trainer = Mock()
        trainer.optimizers = [Mock(param_groups=[{"lr": 5e-5}])]

        pl_module = MockLightningModule()

        callback.on_train_batch_end(trainer, pl_module, {}, Mock(), 0)

        # Verify logged metrics
        assert "system/grad_norm" in pl_module._outputs
        assert pl_module._outputs["system/grad_norm"] > 0  # Should have positive norm

    def test_learning_rate_logging(self):
        """Test learning rate is logged."""
        callback = SystemMonitoringCallback(log_interval=1)

        trainer = Mock()
        trainer.optimizers = [Mock(param_groups=[{"lr": 1e-4}])]

        pl_module = MockLightningModule()

        callback.on_train_batch_end(trainer, pl_module, {}, Mock(), 0)

        assert "system/learning_rate" in pl_module._outputs
        assert abs(pl_module._outputs["system/learning_rate"] - 1e-4) < 1e-8


class TestEmbeddingStatisticsCallback:
    """Test EmbeddingStatisticsCallback."""

    def test_embedding_stats_computation(self):
        """Test embedding statistics are computed."""
        callback = EmbeddingStatisticsCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()

        # Create embeddings with known statistics
        emb = torch.randn(32, 5, 192) * 2.0 + 1.0  # mean≈1, std≈2
        outputs = {"emb": emb}

        callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)

        # Verify logged metrics
        assert "embedding/mean" in pl_module._outputs
        assert "embedding/std" in pl_module._outputs
        assert "embedding/max" in pl_module._outputs
        assert "embedding/min" in pl_module._outputs

        # Check statistics are reasonable
        assert abs(pl_module._outputs["embedding/mean"] - 1.0) < 0.5
        assert abs(pl_module._outputs["embedding/std"] - 2.0) < 0.5

    def test_temporal_cosine_similarity(self):
        """Test temporal cosine similarity computation."""
        callback = EmbeddingStatisticsCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()

        # Create embeddings with different temporal similarity
        emb = torch.randn(32, 10, 192)
        outputs = {"emb": emb}

        callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)

        assert "embedding/temporal_cosine_sim_mean" in pl_module._outputs
        assert "embedding/temporal_cosine_sim_std" in pl_module._outputs

        # Cosine similarity should be in [-1, 1]
        assert -1 <= pl_module._outputs["embedding/temporal_cosine_sim_mean"] <= 1


class TestPredictionQualityCallback:
    """Test PredictionQualityCallback."""

    def test_prediction_quality_logging(self):
        """Test prediction quality metrics are logged."""
        callback = PredictionQualityCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()

        # 新架构：output 只包含关键变量，派生统计在 callback 中计算
        outputs = {
            "pred_loss": torch.tensor(0.5),
            "pred_emb": torch.randn(32, 5, 192),
            "tgt_emb": torch.randn(32, 5, 192),
        }

        callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)

        # 验证 callback 计算的派生统计
        assert "prediction/loss_per_sample" in pl_module._outputs
        assert "prediction/pred_emb_norm" in pl_module._outputs
        assert "prediction/tgt_emb_norm" in pl_module._outputs
        assert "prediction/error_per_dim_mean" in pl_module._outputs
        assert "prediction/error_per_dim_max" in pl_module._outputs
        assert "prediction/error_per_dim_min" in pl_module._outputs
        assert "prediction/cosine_sim_mean" in pl_module._outputs

        # 验证数值合理
        assert abs(pl_module._outputs["prediction/loss_per_sample"] - 0.5) < 1e-6
        assert pl_module._outputs["prediction/pred_emb_norm"] > 0  # 范数应为正
        assert -1 <= pl_module._outputs["prediction/cosine_sim_mean"] <= 1  # 余弦相似度在 [-1,1]


class TestWandBSummaryCallback:
    """Test WandBSummaryCallback."""

    def test_wandb_summary_update(self):
        """Test WandB summary is updated."""
        callback = WandBSummaryCallback()

        trainer = Mock()
        trainer.is_global_zero = True
        trainer.current_epoch = 99  # Last epoch
        trainer.max_epochs = 100

        # Mock WandB logger
        wandb_run = Mock()
        wandb_run.summary = {}

        logger = Mock()
        logger.experiment = wandb_run
        trainer.logger = logger

        # Mock callback metrics
        trainer.callback_metrics = {
            "train/pred_loss": torch.tensor(0.3),
            "train/sigreg_loss": torch.tensor(0.1),
            "representation/rankme_per_dim": torch.tensor(0.85),
        }

        pl_module = MockLightningModule()

        callback.on_train_epoch_end(trainer, pl_module)

        # Verify summary was updated
        assert "final_train_pred_loss" in wandb_run.summary
        assert "final_train_sigreg_loss" in wandb_run.summary
        assert "current_rankme" in wandb_run.summary

    def test_defensive_logging_no_logger(self):
        """Test callback handles missing logger gracefully."""
        callback = WandBSummaryCallback()

        trainer = Mock()
        trainer.is_global_zero = True
        trainer.logger = None  # No logger available

        pl_module = MockLightningModule()

        # Should not crash, should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.on_train_epoch_end(trainer, pl_module)
            # Warning should be issued
            assert len(w) >= 1
            assert "WandB logger not available" in str(w[0].message)


class TestDefensiveProgramming:
    """Test defensive programming for all callbacks."""

    def test_missing_emb_warning(self):
        """Test callback handles missing 'emb' gracefully."""
        callback = RepresentationQualityCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()
        outputs = {}  # No 'emb' in outputs

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)
            # Should issue warning, not crash
            assert len(w) >= 1
            assert "'emb' not found" in str(w[0].message)

    def test_missing_pred_loss_warning(self):
        """Test callback handles missing 'pred_loss' gracefully."""
        callback = PredictionQualityCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()
        outputs = {}  # No 'pred_loss' in outputs

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)
            # Should issue warning, not crash
            assert len(w) >= 1
            assert "'pred_loss' not found" in str(w[0].message)

    def test_invalid_embedding_shape_warning(self):
        """Test callback handles invalid embedding shape gracefully."""
        callback = RepresentationQualityCallback(log_interval=1)

        trainer = Mock()
        pl_module = MockLightningModule()
        outputs = {
            "emb": torch.tensor(1.0)  # Invalid shape (scalar)
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.on_train_batch_end(trainer, pl_module, outputs, Mock(), 0)
            # Should issue warning about invalid shape
            assert len(w) >= 1
            assert "Invalid embedding shape" in str(w[0].message)


class TestTrainingMetricsPlotCallback:
    """Test TrainingMetricsPlotCallback."""

    def test_callback_instantiation(self):
        """Test callback can be instantiated."""
        callback = TrainingMetricsPlotCallback(
            output_dir="/tmp/test_plots",
            plot_format="png",
            dpi=150
        )

        assert callback.plot_format == "png"
        assert callback.dpi == 150
        assert callback.output_dir == Path("/tmp/test_plots")

        # Verify default metrics_to_plot
        assert "Row1_TrainingHealth" in callback.metrics_to_plot
        assert "train/pred_loss" in callback.metrics_to_plot["Row1_TrainingHealth"]

    def test_on_train_end_no_logger(self):
        """Test callback handles missing logger gracefully."""
        callback = TrainingMetricsPlotCallback()

        trainer = Mock()
        trainer.is_global_zero = True
        trainer.logger = None

        pl_module = MockLightningModule()

        # Should not crash
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            callback.on_train_end(trainer, pl_module)
            # Should issue warning about no metrics history
            assert len(w) >= 1
            assert "No metrics history found" in str(w[-1].message)

    def test_plot_generation_with_mock_data(self):
        """Test plot generation with mocked WandB history."""
        import pandas as pd
        import numpy as np

        temp_dir = Path(tempfile.mkdtemp())

        callback = TrainingMetricsPlotCallback(
            output_dir=str(temp_dir / "plots"),
            plot_format="png"
        )

        trainer = Mock()
        trainer.is_global_zero = True

        # Mock WandB run with history DataFrame
        wandb_run = Mock()
        wandb_run.dir = str(temp_dir)

        # Create mock history DataFrame
        steps = np.arange(0, 100, 5)
        pred_loss = np.exp(-steps / 50)  # Exponential decay
        rankme = 0.7 + 0.2 * np.sin(steps / 20)  # Oscillating rankme

        history_df = pd.DataFrame({
            '_step': steps,
            'train/pred_loss': pred_loss,
            'representation/rankme_per_dim': rankme,
        })

        wandb_run.history = Mock(return_value=history_df)

        logger = Mock()
        logger.experiment = wandb_run
        trainer.logger = logger

        pl_module = MockLightningModule()

        # Run on_train_end
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for testing

            callback.on_train_end(trainer, pl_module)

            # Verify plots were saved
            output_dir = temp_dir / "plots"
            assert output_dir.exists()

            # Check if at least one plot file was created
            plot_files = list(output_dir.glob("*.png"))
            assert len(plot_files) > 0

            # Cleanup temp dir
            import shutil
            shutil.rmtree(temp_dir)

        except ImportError:
            # matplotlib not available, should issue warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                callback.on_train_end(trainer, pl_module)
                assert any("matplotlib not available" in str(w_item.message) for w_item in w)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
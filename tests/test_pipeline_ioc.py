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


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size=7):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, interpolate_pos_encoding=True):
        batch = x.shape[0]
        hidden = torch.arange(batch * self.hidden_size, dtype=x.dtype, device=x.device)
        hidden = hidden.reshape(batch, 1, self.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyPredictor(nn.Module):
    def forward(self, emb, act_emb):
        return emb + act_emb[..., : emb.shape[-1]]


class DummyActionEncoder(nn.Module):
    def forward(self, action):
        return action.float()


class JEPAShapeTests(unittest.TestCase):
    def test_encode_and_predict_preserve_expected_shapes(self):
        from lewm.models.jepa import JEPA

        model = JEPA(
            encoder=DummyEncoder(hidden_size=7),
            predictor=DummyPredictor(),
            action_encoder=DummyActionEncoder(),
            projector=nn.Identity(),
            pred_proj=nn.Identity(),
        )

        batch = {
            "pixels": torch.zeros(2, 3, 3, 4, 4),
            "action": torch.ones(2, 3, 7),
        }

        encoded = model.encode(dict(batch))
        predicted = model.predict(encoded["emb"], encoded["act_emb"])

        self.assertEqual(encoded["emb"].shape, (2, 3, 7))
        self.assertEqual(encoded["act_emb"].shape, (2, 3, 7))
        self.assertEqual(predicted.shape, (2, 3, 7))

    def test_decode_raises_error_without_decoder(self):
        """Test that decode() raises error when decoder is None."""
        from lewm.models.jepa import JEPA

        model = JEPA(
            encoder=DummyEncoder(hidden_size=7),
            predictor=DummyPredictor(),
            action_encoder=DummyActionEncoder(),
            decoder=None,
        )

        emb = torch.randn(2, 3, 7)

        with self.assertRaises(RuntimeError):
            model.decode(emb)


if __name__ == "__main__":
    unittest.main()
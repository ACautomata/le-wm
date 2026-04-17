"""Integration tests for Decoder with JEPA model."""

import unittest
import torch
from torch import nn
from types import SimpleNamespace

from lewm.models.jepa import JEPA
from lewm.models.decoder import Decoder
from lewm.models.components import Embedder


class DummyEncoder(nn.Module):
    """Dummy encoder for testing."""

    def __init__(self, hidden_size=192):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, x, interpolate_pos_encoding=True):
        batch = x.shape[0]
        # Simulate ViT output: [CLS] token + patch tokens
        hidden = torch.randn(batch, 197, self.hidden_size, device=x.device, dtype=x.dtype)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyPredictor(nn.Module):
    """Dummy predictor for testing."""

    def forward(self, emb, act_emb):
        return emb + act_emb[..., :emb.shape[-1]]


class TestJEPADecoderIntegration(unittest.TestCase):
    """Test JEPA with decoder integration."""

    def test_jepa_with_decoder_initialization(self):
        """Test that JEPA can be initialized with decoder."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)

        decoder = Decoder(
            cls_dim=192,
            num_patches=196,
            patch_size=16,
        )

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            decoder=decoder,
        )

        self.assertIsNotNone(model.decoder)
        self.assertIsInstance(model.decoder, Decoder)

    def test_jepa_without_decoder(self):
        """Test that JEPA works without decoder."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            decoder=None,
        )

        self.assertIsNone(model.decoder)

    def test_decode_method_raises_error_without_decoder(self):
        """Test that decode() raises error when decoder is None."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            decoder=None,
        )

        emb = torch.randn(2, 3, 192)

        with self.assertRaises(RuntimeError) as context:
            model.decode(emb)

        self.assertIn("Decoder not initialized", str(context.exception))

    def test_decode_method_with_decoder(self):
        """Test that decode() works when decoder is present."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)
        projector = nn.Identity()

        decoder = Decoder(cls_dim=192, num_patches=196, patch_size=16)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            decoder=decoder,
        )

        # Create embeddings
        emb = torch.randn(2, 3, 192)

        # Decode embeddings to images
        images = model.decode(emb)

        # Should output (B, T, C, H, W)
        self.assertEqual(images.shape, (2, 3, 3, 224, 224))

    def test_encode_then_decode_workflow(self):
        """Test full workflow: encode -> decode."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)
        projector = nn.Identity()

        decoder = Decoder(cls_dim=192, num_patches=196, patch_size=16)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            decoder=decoder,
        )

        # Create observations
        batch = {
            "pixels": torch.zeros(4, 5, 3, 224, 224),
            "action": torch.ones(4, 5, 10),
        }

        # Encode
        info = model.encode(batch)
        self.assertIn("emb", info)
        self.assertEqual(info["emb"].shape, (4, 5, 192))

        # Decode
        images = model.decode(info["emb"])
        self.assertEqual(images.shape, (4, 5, 3, 224, 224))

    def test_decode_single_timestep(self):
        """Test decoding single timestep."""
        encoder = DummyEncoder(hidden_size=192)
        predictor = DummyPredictor()
        action_encoder = Embedder(input_dim=10, emb_dim=192)

        decoder = Decoder(cls_dim=192, num_patches=196, patch_size=16)

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            decoder=decoder,
        )

        # Single timestep embedding
        emb = torch.randn(8, 1, 192)
        images = model.decode(emb)

        self.assertEqual(images.shape, (8, 1, 3, 224, 224))


class TestDecoderDifferentDimensions(unittest.TestCase):
    """Test decoder with different embedding dimensions."""

    def test_decoder_with_different_cls_dim(self):
        """Test decoder adapts to different [CLS] dimensions."""
        # Test with smaller dimension
        encoder_small = DummyEncoder(hidden_size=128)
        decoder_small = Decoder(cls_dim=128, num_patches=196)

        model_small = JEPA(
            encoder=encoder_small,
            predictor=DummyPredictor(),
            action_encoder=Embedder(input_dim=10, emb_dim=128),
            decoder=decoder_small,
        )

        emb_small = torch.randn(2, 3, 128)
        images_small = model_small.decode(emb_small)
        self.assertEqual(images_small.shape, (2, 3, 3, 224, 224))

        # Test with larger dimension
        encoder_large = DummyEncoder(hidden_size=384)
        decoder_large = Decoder(cls_dim=384, num_patches=196)

        model_large = JEPA(
            encoder=encoder_large,
            predictor=DummyPredictor(),
            action_encoder=Embedder(input_dim=10, emb_dim=384),
            decoder=decoder_large,
        )

        emb_large = torch.randn(2, 3, 384)
        images_large = model_large.decode(emb_large)
        self.assertEqual(images_large.shape, (2, 3, 3, 224, 224))

    def test_decoder_matches_projector_output_dim(self):
        """Test that decoder cls_dim should match projector output."""
        hidden_size = 768
        embed_dim = 192

        encoder = DummyEncoder(hidden_size=hidden_size)
        projector = nn.Linear(hidden_size, embed_dim)
        decoder = Decoder(cls_dim=embed_dim, num_patches=196)

        model = JEPA(
            encoder=encoder,
            predictor=DummyPredictor(),
            action_encoder=Embedder(input_dim=10, emb_dim=embed_dim),
            projector=projector,
            decoder=decoder,
        )

        batch = {
            "pixels": torch.zeros(2, 3, 3, 224, 224),
            "action": torch.ones(2, 3, 10),
        }

        info = model.encode(batch)
        images = model.decode(info["emb"])

        self.assertEqual(images.shape, (2, 3, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
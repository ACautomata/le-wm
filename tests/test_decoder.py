"""Unit tests for Decoder module."""

import unittest
import torch
from torch import nn

from lewm.models.decoder import Decoder, CrossAttention, DecoderBlock


class TestCrossAttention(unittest.TestCase):
    """Test CrossAttention layer."""

    def test_output_shape_matches_query_shape(self):
        """Test that output shape matches query tokens shape."""
        cross_attn = CrossAttention(dim=64, heads=8, dim_head=8)

        # Query tokens: (B, N_q, D)
        queries = torch.randn(2, 10, 64)
        # Context: (B, N_kv, D)
        context = torch.randn(2, 5, 64)

        output = cross_attn(queries, context)

        self.assertEqual(output.shape, (2, 10, 64))

    def test_different_context_lengths(self):
        """Test that context can have different sequence length."""
        cross_attn = CrossAttention(dim=128, heads=4, dim_head=32)

        queries = torch.randn(4, 20, 128)
        context_single = torch.randn(4, 1, 128)  # Single context token

        output = cross_attn(queries, context_single)

        self.assertEqual(output.shape, (4, 20, 128))


class TestDecoderBlock(unittest.TestCase):
    """Test DecoderBlock."""

    def test_block_preserves_shape(self):
        """Test that decoder block preserves query token shape."""
        block = DecoderBlock(dim=64, heads=4, dim_head=16, mlp_dim=256)

        queries = torch.randn(2, 10, 64)
        context = torch.randn(2, 1, 64)

        output = block(queries, context)

        self.assertEqual(output.shape, (2, 10, 64))

    def test_residual_connections(self):
        """Test that residual connections are applied."""
        block = DecoderBlock(dim=32, heads=2, dim_head=16, mlp_dim=128)

        queries = torch.randn(1, 5, 32)
        context = torch.randn(1, 1, 32)

        # Set all weights to small values to check residual effect
        with torch.no_grad():
            for param in block.parameters():
                param.fill_(0.01)

        output = block(queries, context)

        # Output should be close to input plus small perturbation
        # (due to residual connections)
        self.assertTrue(torch.allclose(output, queries, atol=0.5))


class TestDecoder(unittest.TestCase):
    """Test Decoder module."""

    def test_output_image_shape(self):
        """Test that decoder outputs correct image shape."""
        decoder = Decoder(
            cls_dim=192,
            hidden_dim=256,
            num_patches=196,  # 224x224 / 16x16 = 196 patches
            patch_size=16,
            depth=4,
            heads=8,
            dim_head=32,
            mlp_dim=512,
        )

        # [CLS] embedding: (B, D)
        cls_emb = torch.randn(2, 192)

        # Reconstructed image
        image = decoder(cls_emb)

        # Should output (B, C, H, W) = (2, 3, 224, 224)
        self.assertEqual(image.shape, (2, 3, 224, 224))

    def test_different_patch_sizes(self):
        """Test decoder with different patch sizes."""
        # Test with patch_size=14 (common in ViT)
        decoder = Decoder(
            cls_dim=192,
            num_patches=256,  # 224x224 / 14x14 = 256 patches
            patch_size=14,
        )

        cls_emb = torch.randn(1, 192)
        image = decoder(cls_emb)

        self.assertEqual(image.shape, (1, 3, 224, 224))

    def test_learnable_query_tokens(self):
        """Test that query tokens are learnable parameters."""
        decoder = Decoder(cls_dim=192, num_patches=196)

        # Check that query_tokens is a Parameter
        self.assertIsInstance(decoder.query_tokens, nn.Parameter)

        # Check shape: (1, num_patches, hidden_dim)
        self.assertEqual(decoder.query_tokens.shape, (1, 196, 256))

    def test_cls_projection(self):
        """Test that [CLS] embedding is projected correctly."""
        decoder = Decoder(cls_dim=192, hidden_dim=128)

        cls_emb = torch.randn(4, 192)

        # After projection: (B, hidden_dim)
        projected = decoder.cls_proj(cls_emb)

        self.assertEqual(projected.shape, (4, 128))

    def test_to_pixels_projection(self):
        """Test final projection to patch pixels."""
        decoder = Decoder(cls_dim=192, patch_size=16)

        # Patch embeddings after decoder blocks
        patch_emb = torch.randn(2, 196, 256)

        # Project to pixels: (B, num_patches, patch_size^2 * 3)
        pixels = decoder.to_pixels(patch_emb)

        self.assertEqual(pixels.shape, (2, 196, 16 * 16 * 3))

    def test_batch_independence(self):
        """Test that decoder handles different batch sizes."""
        decoder = Decoder(cls_dim=192, num_patches=196)

        for batch_size in [1, 4, 8]:
            cls_emb = torch.randn(batch_size, 192)
            image = decoder(cls_emb)
            self.assertEqual(image.shape[0], batch_size)

    def test_gradient_flow(self):
        """Test that gradients flow through decoder."""
        decoder = Decoder(cls_dim=192, num_patches=196)

        cls_emb = torch.randn(2, 192, requires_grad=True)
        image = decoder(cls_emb)

        # Compute loss and backprop
        loss = image.sum()
        loss.backward()

        # Check that input has gradient
        self.assertIsNotNone(cls_emb.grad)

        # Check that decoder parameters have gradients
        for param in decoder.parameters():
            self.assertIsNotNone(param.grad)


class TestDecoderArchitecture(unittest.TestCase):
    """Test decoder architecture matches paper specification."""

    def test_cross_attention_architecture(self):
        """Test that cross-attention matches paper description."""
        # Paper: cross-attention layers with residual MLP blocks
        decoder = Decoder(
            cls_dim=192,
            hidden_dim=256,
            depth=4,
            heads=8,
            dim_head=32,
            mlp_dim=512,
        )

        # Verify number of decoder blocks
        self.assertEqual(len(decoder.blocks), 4)

        # Verify each block has cross-attention and MLP
        for block in decoder.blocks:
            self.assertIsInstance(block, DecoderBlock)
            self.assertIsInstance(block.cross_attn, CrossAttention)
            self.assertIsInstance(block.mlp, nn.Sequential)

    def test_num_query_tokens_matches_num_patches(self):
        """Test that query token count matches patch count."""
        # For 224x224 image with 16x16 patches: P = (224/16)^2 = 196
        decoder_16 = Decoder(cls_dim=192, num_patches=196, patch_size=16)
        self.assertEqual(decoder_16.query_tokens.shape[1], 196)

        # For 224x224 image with 14x14 patches: P = (224/14)^2 = 256
        decoder_14 = Decoder(cls_dim=192, num_patches=256, patch_size=14)
        self.assertEqual(decoder_14.query_tokens.shape[1], 256)


if __name__ == "__main__":
    unittest.main()
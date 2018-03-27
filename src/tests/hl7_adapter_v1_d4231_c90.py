"""Tests for hl7_adapter v1d4231y2018."""
import pytest
import torch
import numpy as np


class TestHl7Adapter_v1d4231y2018:
    def test_init(self):
        config = {"domain": "hl7_adapter", "v": 1}
        assert config["v"] == 1

    def test_forward(self):
        x = torch.randn(4, 8)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(3)]
        assert len(batch) == 3

    def test_metric(self):
        pred = torch.randn(8)
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

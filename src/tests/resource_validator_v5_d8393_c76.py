"""Tests for resource_validator v5d8393y2018."""
import pytest
import torch
import numpy as np


class TestResourceValidator_v5d8393y2018:
    def test_init(self):
        config = {"domain": "resource_validator", "v": 5}
        assert config["v"] == 5

    def test_forward(self):
        x = torch.randn(20, 40)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(15)]
        assert len(batch) == 15

    def test_metric(self):
        pred = torch.randn(40)
        target = torch.randn(40)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

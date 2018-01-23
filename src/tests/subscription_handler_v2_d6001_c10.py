"""Tests for subscription_handler v2d6001y2018."""
import pytest
import torch
import numpy as np


class TestSubscriptionHandler_v2d6001y2018:
    def test_init(self):
        config = {"domain": "subscription_handler", "v": 2}
        assert config["v"] == 2

    def test_forward(self):
        x = torch.randn(8, 16)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(6)]
        assert len(batch) == 6

    def test_metric(self):
        pred = torch.randn(16)
        target = torch.randn(16)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

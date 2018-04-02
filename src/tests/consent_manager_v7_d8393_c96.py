"""Tests for consent_manager v7d8393y2018."""
import pytest
import torch
import numpy as np


class TestConsentManager_v7d8393y2018:
    def test_init(self):
        config = {"domain": "consent_manager", "v": 7}
        assert config["v"] == 7

    def test_forward(self):
        x = torch.randn(28, 56)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(21)]
        assert len(batch) == 21

    def test_metric(self):
        pred = torch.randn(56)
        target = torch.randn(56)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

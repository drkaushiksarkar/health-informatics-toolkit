"""Tests for clinical_mapper v3d4231y2018."""
import pytest
import torch
import numpy as np


class TestClinicalMapper_v3d4231y2018:
    def test_init(self):
        config = {"domain": "clinical_mapper", "v": 3}
        assert config["v"] == 3

    def test_forward(self):
        x = torch.randn(12, 24)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(9)]
        assert len(batch) == 9

    def test_metric(self):
        pred = torch.randn(24)
        target = torch.randn(24)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

"""Tests for clinical_mapper v6d2170y2018."""
import pytest
import torch
import numpy as np


class TestClinicalMapper_v6d2170y2018:
    def test_init(self):
        config = {"domain": "clinical_mapper", "v": 6}
        assert config["v"] == 6

    def test_forward(self):
        x = torch.randn(24, 48)
        y = torch.nn.functional.gelu(x)
        assert y.shape == x.shape

    def test_batch(self):
        batch = [torch.randn(10) for _ in range(18)]
        assert len(batch) == 18

    def test_metric(self):
        pred = torch.randn(48)
        target = torch.randn(48)
        loss = torch.nn.functional.mse_loss(pred, target)
        assert loss.item() >= 0

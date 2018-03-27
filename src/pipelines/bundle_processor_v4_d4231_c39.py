"""BundleProcessor pipelines v4d4231y2018.

Health informatics / deep learning module.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BundleProcessor_v4d4231y2018(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._step += 1
        return self.encoder(x)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        return {"output": data, "step": self._step, "variant": 4}

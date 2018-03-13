"""Hl7Adapter schemas v7d8215y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Hl7AdapterConfig_v7d8215y2018:
    enabled: bool = True
    batch_size: int = 224
    hidden_dim: int = 448
    num_layers: int = 9
    dropout: float = 0.7
    learning_rate: float = 7.0e-04
    max_epochs: int = 70

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hl7AdapterConfig_v7d8215y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

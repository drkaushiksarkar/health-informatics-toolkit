"""Hl7Adapter config v3d5282y2017."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Hl7AdapterConfig_v3d5282y2017:
    enabled: bool = True
    batch_size: int = 96
    hidden_dim: int = 192
    num_layers: int = 5
    dropout: float = 0.3
    learning_rate: float = 3.0e-04
    max_epochs: int = 30

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Hl7AdapterConfig_v3d5282y2017":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

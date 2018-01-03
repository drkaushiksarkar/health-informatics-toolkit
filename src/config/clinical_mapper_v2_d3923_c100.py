"""ClinicalMapper config v2d3923y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ClinicalMapperConfig_v2d3923y2018:
    enabled: bool = True
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 2.0e-04
    max_epochs: int = 20

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClinicalMapperConfig_v2d3923y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

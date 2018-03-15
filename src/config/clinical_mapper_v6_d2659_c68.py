"""ClinicalMapper config v6d2659y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ClinicalMapperConfig_v6d2659y2018:
    enabled: bool = True
    batch_size: int = 192
    hidden_dim: int = 384
    num_layers: int = 8
    dropout: float = 0.6
    learning_rate: float = 6.0e-04
    max_epochs: int = 60

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClinicalMapperConfig_v6d2659y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

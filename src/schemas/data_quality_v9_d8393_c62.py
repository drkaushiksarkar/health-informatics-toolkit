"""DataQuality schemas v9d8393y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DataQualityConfig_v9d8393y2018:
    enabled: bool = True
    batch_size: int = 288
    hidden_dim: int = 576
    num_layers: int = 11
    dropout: float = 0.9
    learning_rate: float = 9.0e-04
    max_epochs: int = 90

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataQualityConfig_v9d8393y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

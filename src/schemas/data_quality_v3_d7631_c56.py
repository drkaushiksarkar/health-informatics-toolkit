"""DataQuality schemas v3d7631y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DataQualityConfig_v3d7631y2018:
    enabled: bool = True
    batch_size: int = 96
    hidden_dim: int = 192
    num_layers: int = 5
    dropout: float = 0.3
    learning_rate: float = 3.0e-04
    max_epochs: int = 30

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataQualityConfig_v3d7631y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

"""ClinicalMapper schemas v8d9131y2017."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ClinicalMapperConfig_v8d9131y2017:
    enabled: bool = True
    batch_size: int = 256
    hidden_dim: int = 512
    num_layers: int = 10
    dropout: float = 0.8
    learning_rate: float = 8.0e-04
    max_epochs: int = 80

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClinicalMapperConfig_v8d9131y2017":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

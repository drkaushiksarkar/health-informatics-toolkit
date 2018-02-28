"""ResourceValidator config v4d6594y2018."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ResourceValidatorConfig_v4d6594y2018:
    enabled: bool = True
    batch_size: int = 128
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.4
    learning_rate: float = 4.0e-04
    max_epochs: int = 40

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResourceValidatorConfig_v4d6594y2018":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

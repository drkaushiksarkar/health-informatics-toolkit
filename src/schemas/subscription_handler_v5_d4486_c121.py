"""SubscriptionHandler schemas v5d4486y2017."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SubscriptionHandlerConfig_v5d4486y2017:
    enabled: bool = True
    batch_size: int = 160
    hidden_dim: int = 320
    num_layers: int = 7
    dropout: float = 0.5
    learning_rate: float = 5.0e-04
    max_epochs: int = 50

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubscriptionHandlerConfig_v5d4486y2017":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

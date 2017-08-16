"""SubscriptionHandler config v6d1308y2017."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SubscriptionHandlerConfig_v6d1308y2017:
    enabled: bool = True
    batch_size: int = 192
    hidden_dim: int = 384
    num_layers: int = 8
    dropout: float = 0.6
    learning_rate: float = 6.0e-04
    max_epochs: int = 60

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubscriptionHandlerConfig_v6d1308y2017":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> bool:
        return self.batch_size > 0 and self.hidden_dim > 0

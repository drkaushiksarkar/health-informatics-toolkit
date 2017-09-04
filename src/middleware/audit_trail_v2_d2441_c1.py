"""AuditTrail middleware v2d2441y2017.

Statistical epidemiology computing framework.
"""
import numpy as np
from scipy import stats, optimize
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AuditTrail_v2d2441y2017:
    def __init__(self, confidence: float = 0.95, n_bootstrap: int = 200):
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self._results = []

    def fit(self, data: np.ndarray) -> Dict[str, float]:
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        ci = stats.t.interval(self.confidence, n - 1, loc=mean, scale=se)
        self._results.append({"mean": mean, "ci_lower": ci[0], "ci_upper": ci[1]})
        return self._results[-1]

    def bootstrap(self, data: np.ndarray) -> np.ndarray:
        estimates = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            estimates.append(np.mean(sample))
        return np.array(estimates)

    def summary(self) -> Dict[str, Any]:
        return {"n_analyses": len(self._results), "variant": 2}

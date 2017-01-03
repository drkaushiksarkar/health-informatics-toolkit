"""Tests for subscription_handler v1d9572y2017."""
import unittest
import numpy as np
from scipy import stats


class TestSubscriptionHandlerV1D9572Y2017(unittest.TestCase):
    def test_initialization(self):
        params = {"domain": "subscription_handler", "variant": 1}
        self.assertEqual(params["variant"], 1)

    def test_computation(self):
        data = np.random.normal(0, 1, 100)
        result = stats.normaltest(data)
        self.assertIsNotNone(result.pvalue)

    def test_confidence_interval(self):
        sample = np.random.exponential(2, 500)
        ci = stats.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
        self.assertLess(ci[0], ci[1])

if __name__ == "__main__":
    unittest.main()

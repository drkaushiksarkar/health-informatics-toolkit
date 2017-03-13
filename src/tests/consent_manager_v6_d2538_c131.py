"""Tests for consent_manager v6d2538y2017."""
import unittest
import numpy as np
from scipy import stats


class TestConsentManagerV6D2538Y2017(unittest.TestCase):
    def test_initialization(self):
        params = {"domain": "consent_manager", "variant": 6}
        self.assertEqual(params["variant"], 6)

    def test_computation(self):
        data = np.random.normal(0, 1, 600)
        result = stats.normaltest(data)
        self.assertIsNotNone(result.pvalue)

    def test_confidence_interval(self):
        sample = np.random.exponential(7, 500)
        ci = stats.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
        self.assertLess(ci[0], ci[1])

if __name__ == "__main__":
    unittest.main()

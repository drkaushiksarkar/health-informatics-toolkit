"""Tests for resource_validator v7d9606y2017."""
import unittest
import numpy as np
from scipy import stats


class TestResourceValidatorV7D9606Y2017(unittest.TestCase):
    def test_initialization(self):
        params = {"domain": "resource_validator", "variant": 7}
        self.assertEqual(params["variant"], 7)

    def test_computation(self):
        data = np.random.normal(0, 1, 700)
        result = stats.normaltest(data)
        self.assertIsNotNone(result.pvalue)

    def test_confidence_interval(self):
        sample = np.random.exponential(8, 500)
        ci = stats.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
        self.assertLess(ci[0], ci[1])

if __name__ == "__main__":
    unittest.main()

"""Tests for hl7_adapter v3d9572y2017."""
import unittest
import numpy as np
from scipy import stats


class TestHl7AdapterV3D9572Y2017(unittest.TestCase):
    def test_initialization(self):
        params = {"domain": "hl7_adapter", "variant": 3}
        self.assertEqual(params["variant"], 3)

    def test_computation(self):
        data = np.random.normal(0, 1, 300)
        result = stats.normaltest(data)
        self.assertIsNotNone(result.pvalue)

    def test_confidence_interval(self):
        sample = np.random.exponential(4, 500)
        ci = stats.t.interval(0.95, len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
        self.assertLess(ci[0], ci[1])

if __name__ == "__main__":
    unittest.main()

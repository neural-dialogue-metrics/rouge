import unittest
from rouge.metrics import RougeScore


class TestRougeScore(unittest.TestCase):

    def test_basic(self):
        """
        Basic properties of RougeScore.
        :return:
        """
        score = RougeScore(recall=0.5, precision=0.5, f1_measure=0.5)
        print(score)
        self.assertAlmostEqual(score.recall, 0.5)
        self.assertAlmostEqual(score.precision, 0.5)
        self.assertAlmostEqual(score.f1_measure, 0.5)

        self.assertEqual(list(score), [0.5] * 3)

import unittest
from rouge.rouge import rouge_n
from rouge.tests import make_rouge


class TestRougeN(unittest.TestCase):
    summary = 'gunman kill the police'
    references = 'the police killed the gunman'

    def test_rouge_2(self):
        r, p, f = rouge_n(self.summary.split(), [self.references.split()])
        rouge = make_rouge(summary=[[self.summary]], reference=[[[self.references]]], n_gram=2)
        score = rouge.calc_score()
        self.assertAlmostEqual(r, score['ROUGE-2-R'], places=5)
        self.assertAlmostEqual(p, score['ROUGE-2-P'], places=5)
        self.assertAlmostEqual(f, score['ROUGE-2-F'], places=5)

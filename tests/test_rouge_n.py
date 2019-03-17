import unittest

from rouge.rouge import rouge_n_sentence_level
from rouge.tests.wrapper import rouge_l_sentence_level as _rouge_n


class TestRougeN(unittest.TestCase):
    summary = 'gunman kill the police'.split()
    references = 'the police killed the gunman'.split()

    def test_rouge_2(self):
        r, p, f = rouge_n_sentence_level(self.summary, [self.references], n=2)
        r_, p_, f_ = _rouge_n(self.summary, [self.references], n=2, alpha=0.5)
        self.assertAlmostEqual(r, r_, places=5)
        self.assertAlmostEqual(p, p_, places=5)
        self.assertAlmostEqual(f, f_, places=5)

    def test_rouge_1(self):
        r, p, f = rouge_n_sentence_level(self.summary, [self.references], n=1)
        r_, p_, f_ = _rouge_n(self.summary, [self.references], n=1, alpha=0.5)
        self.assertAlmostEqual(r, r_, places=5)
        self.assertAlmostEqual(p, p_, places=5)
        self.assertAlmostEqual(f, f_, places=5)

import unittest

from rouge.tests import load_sentence_pairs
from rouge.rouge import rouge_n_sentence_level
from rouge.tests.wrapper import rouge_n_sentence_level as _rouge_n


class TestRougeN(unittest.TestCase):
    summary = 'gunman kill the police'.split()
    references = 'the police killed the gunman'.split()
    N_TO_TEST = [i + 1 for i in range(2)]  # test 1-4 gram

    def test_rouge_2(self):
        r, p, f = rouge_n_sentence_level(self.summary, self.references, n=2)
        r_, p_, f_ = _rouge_n(self.summary, self.references, n=2, alpha=0.5)
        self.assertAlmostEqual(r, r_, places=5)
        self.assertAlmostEqual(p, p_, places=5)
        self.assertAlmostEqual(f, f_, places=5)

    def test_rouge_1(self):
        r, p, f = rouge_n_sentence_level(self.summary, self.references, n=1)
        r_, p_, f_ = _rouge_n(self.summary, self.references, n=1, alpha=0.5)
        self.assertAlmostEqual(r, r_, places=5)
        self.assertAlmostEqual(p, p_, places=5)
        self.assertAlmostEqual(f, f_, places=5)

    def test_rouge_n_sentence_level(self):
        for n in self.N_TO_TEST:
            for ours_data, theirs_data in load_sentence_pairs():
                ours_score = rouge_n_sentence_level(*ours_data, n=n)
                theirs_score = _rouge_n(*theirs_data, n=n)

                for ours, theirs in zip(ours_score, theirs_score):
                    self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                      ours_data = %r
                                      theirs_data = %r
                                      
                                      ours_score = %r
                                      theirs_score = %r
                                      n = %r
                                      """ % (ours_data, theirs_data, ours_score, theirs_score, n))

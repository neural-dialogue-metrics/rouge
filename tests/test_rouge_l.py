import unittest

from rouge.rouge import rouge_l_summary_level, rouge_l_sentence_level
from rouge.tests.wrapper import rouge_l_sentence_level as _rouge_l_sentence_level
from rouge.tests import summary, reference
from rouge.tests import load_sentence_pairs


class TestRougeL(unittest.TestCase):

    def test_sentence_level(self):
        r_, p_, f_ = _rouge_l_sentence_level(summary, reference, 0.5)
        r, p, f = rouge_l_sentence_level(summary, reference)
        self.assertAlmostEqual(r_, r, places=5)
        self.assertAlmostEqual(p_, p, places=5)
        self.assertAlmostEqual(f_, f, places=5)

    def test_summary_level(self):
        for n in self.N_TO_TEST:
            for ours_data, theirs_data in load_summary_pairs():
                ours_score = rouge_l_summary_level(*ours_data, n=n)
                theirs_score = _rouge_l_summary_level(*theirs_data, n=n)

                for ours, theirs in zip(ours_score, theirs_score):
                    self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                                     ours_data = %r
                                                     theirs_data = %r

                                                     ours_score = %r
                                                     theirs_score = %r
                                                     n = %r
                                                     """ % (ours_data, theirs_data, ours_score, theirs_score, n))

    def test_sentence_level_(self):
        for ours_data, theirs_data in load_sentence_pairs():
            score = rouge_l_sentence_level(*ours_data)
            score_ = _rouge_l_sentence_level(*theirs_data)

            for ours, theirs in zip(score, score_):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                        ours_data = %r
                        theirs_tokens = %r
                        """ % (ours_data, theirs_data))

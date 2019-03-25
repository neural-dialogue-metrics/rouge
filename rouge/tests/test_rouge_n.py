"""Testing ROUGE-N."""
import unittest

from rouge.tests import load_sentence_pairs
from rouge.tests import load_summary_pairs
from rouge.tests import summary
from rouge.tests import reference

from rouge.tests.wrapper import rouge_n_sentence_level as _rouge_n_sentence_level
from rouge.tests.wrapper import rouge_n_summary_level as _rouge_n_summary_level

from rouge.metrics import rouge_n_sentence_level
from rouge.metrics import rouge_n_summary_level


class TestRougeN(unittest.TestCase):
    N_TO_TEST = [i + 1 for i in range(4)]  # test 1-4 gram

    def test_example(self):
        for n in self.N_TO_TEST:
            ours_score = rouge_n_sentence_level(summary, reference, n=n)
            theirs_score = _rouge_n_sentence_level(summary, reference, n=n)

            for ours, theirs in zip(ours_score, theirs_score):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                                 ours_score = %r
                                                 theirs_score = %r
                                                 n = %r
                                                 """ % (ours_score, theirs_score, n))

    def test_sentence_level(self):
        for n in self.N_TO_TEST:
            for ours_data, theirs_data in load_sentence_pairs():
                ours_score = rouge_n_sentence_level(*ours_data, n=n)
                theirs_score = _rouge_n_sentence_level(*theirs_data, n=n)

                for ours, theirs in zip(ours_score, theirs_score):
                    self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                      ours_data = %r
                                      theirs_data = %r
                                      
                                      ours_score = %r
                                      theirs_score = %r
                                      n = %r
                                      """ % (ours_data, theirs_data, ours_score, theirs_score, n))

    def test_summary_level(self):
        for n in self.N_TO_TEST:
            for ours_data, theirs_data in load_summary_pairs():
                ours_score = rouge_n_summary_level(*ours_data, n=n)
                theirs_score = _rouge_n_summary_level(*theirs_data, n=n)

                for ours, theirs in zip(ours_score, theirs_score):
                    self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                             ours_data = %r
                                             theirs_data = %r
    
                                             ours_score = %r
                                             theirs_score = %r
                                             n = %r
                                             """ % (ours_data, theirs_data, ours_score, theirs_score, n))

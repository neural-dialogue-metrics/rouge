"""Testing ROUGE-L."""
import unittest

from rouge.tests.wrapper import rouge_l_sentence_level as _rouge_l_sentence_level
from rouge.tests.wrapper import rouge_l_summary_level as _rouge_l_summary_level

from rouge.tests import summary, reference
from rouge.tests import load_sentence_pairs
from rouge.tests import load_summary_pairs

from rouge.metrics import rouge_l_sentence_level
from rouge.metrics import rouge_l_summary_level


class TestRougeL(unittest.TestCase):

    def test_example(self):
        ours_score = rouge_l_sentence_level(summary, reference)
        theirs_score = _rouge_l_sentence_level(summary, reference)

        for ours, theirs in zip(ours_score, theirs_score):
            self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                                ours_score = %r
                                theirs_score = %r
                                """ % (ours_score, theirs_score))

    def test_sentence_level(self):
        for ours_data, theirs_data in load_sentence_pairs():
            ours_score = rouge_l_sentence_level(*ours_data)
            theirs_score = _rouge_l_sentence_level(*theirs_data)

            for ours, theirs in zip(ours_score, theirs_score):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                        ours_data = %r
                        theirs_data = %r
                        ours_score = %r
                        theirs_score = %r
                        """ % (ours_data, theirs_data, ours_score, theirs_score))

    def test_summary_level(self):
        for ours_data, theirs_data in load_summary_pairs():
            ours_score = rouge_l_summary_level(*ours_data)
            theirs_score = _rouge_l_summary_level(*theirs_data)

            for ours, theirs in zip(ours_score, theirs_score):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                         ours_data = %r
                         theirs_data = %r
    
                         ours_score = %r
                         theirs_score = %r
                         """ % (ours_data, theirs_data, ours_score, theirs_score))

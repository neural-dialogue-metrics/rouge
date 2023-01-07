# MIT License
# 
# Copyright (c) 2019 Cong Feng
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Testing ROUGE-W."""
import unittest

from rouge.tests.wrapper import rouge_w_sentence_level as _rouge_w_sentence_level
from rouge.tests.wrapper import rouge_w_summary_level as _rouge_w_summary_level

from rouge.tests import summary, reference
from rouge.tests import load_sentence_pairs
from rouge.tests import load_summary_pairs

from rouge.metrics import rouge_w_sentence_level
from rouge.metrics import rouge_w_summary_level


class TestRougeW(unittest.TestCase):

    def test_example(self):
        ours_score = rouge_w_sentence_level(summary, reference)
        theirs_score = _rouge_w_sentence_level(summary, reference)

        for ours, theirs in zip(ours_score, theirs_score):
            self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                        ours_score = %r
                        theirs_score = %r
                        """ % (ours_score, theirs_score))

    def test_sentence_level(self):
        for ours_data, theirs_data in load_sentence_pairs():
            ours_score = rouge_w_sentence_level(*ours_data)
            theirs_score = _rouge_w_sentence_level(*theirs_data)

            for ours, theirs in zip(ours_score, theirs_score):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                        ours_data = %r
                        theirs_data = %r
                        ours_score = %r
                        theirs_score = %r
                        """ % (ours_data, theirs_data,
                               ours_score, theirs_score))

    def test_summary_level(self):
        for ours_data, theirs_data in load_summary_pairs():
            ours_score = rouge_w_summary_level(*ours_data)
            theirs_score = _rouge_w_summary_level(*theirs_data)

            for ours, theirs in zip(ours_score, theirs_score):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                         ours_data = %r
                         theirs_data = %r
    
                         ours_score = %r
                         theirs_score = %r
                         """ % (ours_data, theirs_data, ours_score, theirs_score))

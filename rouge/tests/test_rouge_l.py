# Copyright 2019 Cong Feng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Testing ROUGE-L."""
import unittest

from rouge.tests.wrapper import rouge_l_sentence_level as _rouge_l_sentence_level
from rouge.tests.wrapper import rouge_l_summary_level as _rouge_l_summary_level

from rouge.tests import summary, reference
from rouge.tests import load_sentence_pairs
from rouge.tests import load_summary_pairs

from rouge.rouge import rouge_l_sentence_level
from rouge.rouge import rouge_l_summary_level


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
            score = rouge_l_sentence_level(*ours_data)
            score_ = _rouge_l_sentence_level(*theirs_data)

            for ours, theirs in zip(score, score_):
                self.assertAlmostEqual(ours, theirs, delta=1e-5, msg="""
                        ours_data = %r
                        theirs_data = %r
                        """ % (ours_data, theirs_data))

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

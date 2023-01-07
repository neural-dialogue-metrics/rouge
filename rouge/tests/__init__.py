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

"""
Data loading utilities for testing.
"""
import os
import json
import re

__all__ = ["load_sentence_pairs", "load_summary_pairs", "summary", "reference"]

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ROUGE-test.json')

# Keys in the json data.
SUMMARIES = 'summaries'
REFERENCES = 'references'


def load_data():
    """
    Load the test data in json format.
    :return: a dict of json data.
    """
    with open(DATA_PATH) as f:
        raw_data = json.load(f)
    return raw_data


def load_sentence_pairs():
    for raw_data in load_data().values():
        for pair in zip(raw_data[SUMMARIES], raw_data[REFERENCES]):
            theirs = [clean(sentence) for sentence in pair]
            ours = [sentence.split() for sentence in theirs]
            yield ours, theirs


def load_summary_pairs():
    for raw_data in load_data().values():
        summary_sentences = [clean(sentence) for sentence in raw_data[SUMMARIES]]
        reference_sentences = [clean(sentence) for sentence in raw_data[REFERENCES]]
        theirs = (summary_sentences, reference_sentences)
        ours = ([sentence.split() for sentence in summary_sentences],
                [sentence.split() for sentence in reference_sentences])
        yield ours, theirs


# One pair of data points.
summary = 'gunman kill the police'.split()
reference = 'the police killed the gunman'.split()


def clean(sentence):
    """
    Clean up a sentence. Remove non alpha number letters. Lower cased.

    >>> clean('all done.')
    'all done'
    >>> clean('ALL DONE.')
    'all done'
    >>> clean('here we go:')
    'here we go'

    :param sentence:
    :return:
    """
    return re.sub('[^A-Za-z0-9]', ' ', sentence).lower().strip()

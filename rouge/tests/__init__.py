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

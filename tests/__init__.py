import os
import json
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ROUGE-test.json')

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
            yield [clean(sentence) for sentence in pair]


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

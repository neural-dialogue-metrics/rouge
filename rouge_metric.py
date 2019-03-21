"""Driver script to compute ROUGE score."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

from rouge import rouge_n_sentence_level
from rouge import rouge_l_sentence_level
from rouge import rouge_w_sentence_level


def _break_into_words(line):
    """
    Turn a already-tokenized line into a list of words.
    :param line: string, already tokenized. All tokens are separated by space.
    :return: List[string], broken into words.
    """
    return line.strip().split(' ')


def _read_corpus(file):
    with open(file) as f:
        return [_break_into_words(line) for line in f.readlines()]


def _compute_mean_score(metric_fn, summary, reference):
    """
    Compute a metric on every pair of (summary, reference).
    Return their mean.

    >>> _compute_mean_score(lambda x,y: x+y, [1,2,3], [4,5,6])
    7.0

    :param metric_fn: A function that takes two sentence as input and returns a float.
    :param summary: a list of sentence.
    :param reference: a list of sentence.
    :return: the mean of applying metric_fn to every pair of summary and reference.
    """
    assert len(summary) == len(reference)
    assert len(summary) > 0

    sentence_pairs = zip(summary, reference)
    total = sum(metric_fn(s, r) for s, r in sentence_pairs)
    return total / len(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute average ROUGE-1 ROUGE-2 and ROUGE-L on sentence pairs")
    parser.add_argument("summary", help="a file of summaries, one sentence per line.")
    parser.add_argument("reference", help="a file of references, one sentence per line")
    parser.add_argument("-a", "--alpha", help="weight factor for the recall in the F1-measure", type=float)
    parser.add_argument("-w", "--weight", help="weight factor for the ROUGE-W", type=float)
    args = parser.parse_args()

    summary = _read_corpus(args.summary)
    reference = _read_corpus(args.reference)

    metric_fns = {
        'ROUGE-1': lambda s, r: rouge_n_sentence_level(s, r, 1, args.alpha)[-1],
        'ROUGE-2': lambda s, r: rouge_n_sentence_level(s, r, 2, args.alpha)[-1],
        'ROUGE-L': lambda s, r: rouge_l_sentence_level(s, r, args.alpha)[-1],
        'ROUGE-W': lambda s, r: rouge_w_sentence_level(s, r, weight=args.weight, alpha=args.alpha)[-1],
    }

    for name, fn in metric_fns.items():
        mean = _compute_mean_score(fn, summary, reference)
        print('%s: %f' % (name, mean))

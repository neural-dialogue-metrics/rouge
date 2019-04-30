#!/usr/bin/env python3
"""Driver script to compute ROUGE score."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from agenda.metric_helper import write_score
from pathlib import Path
import logging

from rouge import *

logger = logging.getLogger(__name__)


class MetricWrapper:

    def __init__(self, name, sentence_score, summary_score, alpha, **kwargs):
        self.alpha = alpha
        self.name = name
        self._sentence_score = sentence_score
        self._summary_score = summary_score
        self._kwargs = kwargs

    def sentence_score(self, sum, ref):
        return self._sentence_score(sum, ref, alpha=self.alpha, **self._kwargs).f1_measure

    def summary_score(self, sum, ref):
        return self._summary_score(sum, ref, alpha=self.alpha, **self._kwargs).f1_measure

    @property
    def params(self):
        # don't send empty dict.
        return self._kwargs or None

    def output_file(self, dir: Path):
        return dir.joinpath(self.name).with_suffix('.json')

    def eval(self, summary, reference, output_dir):
        logger.info('computing %s', self.name)
        scores = [self.sentence_score(s, r) for s, r in zip(summary, reference)]
        system = self.summary_score(summary, reference)
        write_score(
            name=self.name,
            params=self.params,
            system=system,
            output=self.output_file(output_dir),
            scores=scores
        )


def _read_corpus(file):
    def _break_into_words(line):
        """
        Turn a already-tokenized line into a list of words.
        :param line: string, already tokenized. All tokens are separated by space.
        :return: List[string], broken into words.
        """
        return line.strip().split(' ')

    with open(file) as f:
        return [_break_into_words(line) for line in f.readlines()]


class Runner:

    def __init__(self, summary_file, reference_file, output_dir):
        logger.info('summary_file: %s', summary_file)
        logger.info('reference_file: %s', reference_file)
        logger.info('output_dir: %s', output_dir)

        self.summary = _read_corpus(summary_file)
        self.reference = _read_corpus(reference_file)
        self.output_dir = Path(output_dir)

    def eval_metric(self, args):
        for wrapper in self.get_metrics(args):
            wrapper.eval(
                summary=self.summary,
                reference=self.reference,
                output_dir=self.output_dir,
            )

    def get_metrics(self, args):
        if args.rouge_n:
            for n in args.rouge_n:
                yield MetricWrapper(
                    name='rouge_n_%d' % n,
                    sentence_score=rouge_n_sentence_level,
                    summary_score=rouge_n_summary_level,
                    alpha=args.alpha,
                    n=n,
                )

        if args.rouge_l:
            yield MetricWrapper(
                name='rouge_l',
                sentence_score=rouge_l_sentence_level,
                summary_score=rouge_l_summary_level,
                alpha=args.alpha,
            )

        if args.rouge_w:
            yield MetricWrapper(
                name='rouge_w',
                sentence_score=rouge_w_sentence_level,
                summary_score=rouge_w_summary_level,
                alpha=args.alpha,
                weight=args.weight,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="a file of summaries, one sentence per line.")
    parser.add_argument("reference", help="a file of references, one sentence per line")
    parser.add_argument('--output_dir', help='output dir')
    parser.add_argument("-a", "--alpha", help="weight factor for the recall in the F1-measure", type=float)
    parser.add_argument("-w", "--weight", help="weight factor for the ROUGE-W", type=float)

    # Options for various rouges:
    parser.add_argument('-N', dest='rouge_n', type=int, nargs='*', help='compute ROUGE-N for all specified n')
    parser.add_argument('-W', dest='rouge_w', action='store_true', help='compute ROUGE-W')
    parser.add_argument('-L', dest='rouge_l', action='store_true', help='compute ROUGE-L')
    args = parser.parse_args()

    runner = Runner(args.summary, args.reference, args.output_dir)
    runner.eval_metric(args)

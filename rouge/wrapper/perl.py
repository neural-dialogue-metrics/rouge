#  MIT License
#
#  Copyright (c) 2019 Cong Feng
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import os
import collections
import tempfile

ROUGE_HOME = os.path.join(os.path.dirname(__file__), 'ROUGE-1.5.5')
assert os.path.isdir(ROUGE_HOME), 'ROUGE_HOME is broken!'

ROUGE_DATA_HOME = os.path.join(ROUGE_HOME, 'data')
assert os.path.isdir(ROUGE_DATA_HOME), 'ROUGE_DATA_HOME is broken!'

ROUGE_EXECUTABLE = os.path.join(ROUGE_HOME, 'ROUGE-1.5.5.pl')
assert os.path.isfile(ROUGE_EXECUTABLE), 'ROUGE_EXECUTABLE is broken!'

_ROUGE_README = os.path.join(ROUGE_HOME, 'README.txt')


def print_readme():
    """
    Print README.txt of the perl package.
    :return:
    """
    with open(_ROUGE_README) as f:
        print(f.read())


class RougeParams:
    """
    RougeParams is a container of all the command line options that the perl script accepts.
    It gives them readable names, sensible default values and online document.
    It translates our readable names to the single-character option understood by the perl script.
    """
    # valid values for config_format.
    CONFIG_FORMATS = ('SEE', 'SPL', 'ISI', 'SIMPLE')

    # valid values for scoring_formula.
    SCORING_FORMULAS = ('A', 'B')

    # valid values for counting_unit.
    COUNT_SENTENCE = 0
    COUNT_TOKEN = 1
    COUNT_TOKEN_WITH_RAW_COUNTS = 2
    COUNTING_UNITS = (
        COUNT_SENTENCE,
        COUNT_TOKEN,
        COUNT_TOKEN_WITH_RAW_COUNTS,
    )

    # valid values for basic_element.
    BE_H = 0  # head only scoring (does not applied to Minipar-based BEs).
    BE_HM = 1  # head and modifier pair scoring.
    BE_HMR = 2  # head, modifier and relation triple scoring.
    BE_HM1 = 3  # H and HM scoring (same as HM for Minipar-based BEs).
    BE_HMR1 = 4  # HM and HMR scoring (same as HMR for Minipar-based BEs).
    BE_HMR2 = 5  # H, HM and HMR scoring (same as HMR for Minipar-based BEs).
    BASIC_ELEMENTS = (
        BE_H, BE_HM, BE_HMR,
        BE_HM1, BE_HMR1, BE_HMR2,
    )

    # valid values for confidence_interval [0, 100].
    CONFIDENCE_INTERVALS = range(101)

    def __init__(self,
                 config_file: str,
                 system_id=None,
                 skip_distance: int = None,
                 skip_with_unigram: bool = None,
                 basic_element: int = None,
                 all_systems: bool = None,
                 confidence_interval: int = None,
                 print_when_eval: bool = None,
                 env: str = None,
                 scoring_formula: str = None,
                 print_help: bool = None,
                 max_bytes: int = None,
                 max_words: int = None,
                 stemming: bool = None,
                 max_ngram: int = None,
                 alpha: float = None,
                 remove_stopwords: bool = None,
                 counting_unit: int = None,
                 resampling_points: int = None,
                 wlcs_weight: float = None,
                 verbose: bool = None,
                 no_rouge_l: bool = None,
                 config_format: str = None):
        """

        :param config_file: The configure file describing the peer and model summaries.
        :param system_id: Decide which system to evaluate. Mutually excluded with all_all_systems.
        :param skip_distance: When given, enable ROUGE-S and provide the skip distance parameter.
        :param skip_with_unigram: When given, enable ROUGE-SU.
        :param basic_element: When given, enable ROUGE-BE and provide the BE parameter.
        :param all_systems: Evaluate all systems.
        :param confidence_interval: Specify confidence interval to compute.
        :param print_when_eval: Print per evaluation average score for each system.
        :param env: Specify ROUGE_EVAL_HOME directory where the ROUGE data files can be found.
        :param scoring_formula: Select scoring formula: 'A' => model average; 'B' => best model.
        :param print_help: Print usage information.
        :param max_bytes: Only use the first n bytes in the system/peer summary for the evaluation.
        :param max_words: Only use the first n words in the system/peer summary for the evaluation.
        :param stemming:  Stem both model and system summaries using Porter stemmer.
        :param max_ngram: Compute ROUGE-N up to max-ngram length will be computed.
        :param alpha: Relative importance of recall and precision ROUGE scores.
        :param remove_stopwords: Remove stopwords in model and system summaries.
        :param counting_unit: Compute average ROUGE by averaging over the whole test corpus instead of sentences.
        See `self.COUNTING_UNITS` for possible values.
        :param resampling_points: If not None, enable bootstrap resampling and specify the number of sampling point.
        :param wlcs_weight: If not None, enable ROUGE-W and specify the weight for it.
        ROUGE-W gives consecutive matches of length L in an LCS a weight of 'L ^ weight' instead of just 'L' as in LCS.
        :param verbose: Print debugging information for diagnostic purpose.
        :param no_rouge_l: Do not calculate ROUGE-L.
        :param config_format: ROUGE-eval-config-file is a list of peer-model pair per line in the specified format.
        See `self.CONFIG_FORMATS` for possible values.
        """
        self.print_help = print_help
        self.config_file = config_file

        if system_id is not None and all_systems:
            raise ValueError('either system_id or all_systems, not both')
        self.system_id = system_id
        self.all_systems = all_systems

        self.skip_distance = skip_distance
        self.skip_with_unigram = skip_with_unigram

        if basic_element is not None:
            if basic_element not in self.BASIC_ELEMENTS:
                raise ValueError('invalid basic_element')
        self.basic_element = basic_element

        if not (confidence_interval is None or confidence_interval in self.CONFIDENCE_INTERVALS):
            raise ValueError('invalid confidence_interval')

        self.confidence_interval = confidence_interval
        self.print_when_eval = print_when_eval
        self.env = env or ROUGE_DATA_HOME

        if scoring_formula is not None:
            if scoring_formula not in self.SCORING_FORMULAS:
                raise ValueError('invalid scoring formula')
        self.scoring_formula = scoring_formula

        self.max_bytes = max_bytes
        self.max_words = max_words
        self.stemming = stemming

        if max_ngram is None:
            max_ngram = -1  # disable ROUGE-N

        self.max_ngram = max_ngram
        self.alpha = alpha
        self.remove_stopwords = remove_stopwords

        if counting_unit is not None:
            if counting_unit not in self.COUNTING_UNITS:
                raise ValueError('invalid counting_unit')
        self.counting_unit = counting_unit

        self.resampling_points = resampling_points
        self.wlcs_weight = wlcs_weight
        self.verbose = verbose
        self.no_rouge_l = no_rouge_l
        if config_format is not None:
            if config_format not in self.CONFIG_FORMATS:
                raise ValueError('invalid config_format')
        self.config_format = config_format

    def make_options(self):
        """

        :return:
        """
        options = [ROUGE_EXECUTABLE]
        options.extend(['-e', self.env])
        options.extend(['-n', self.max_ngram])
        if self.all_systems:
            options.append('-a')
        if self.skip_distance is not None:
            options.extend(['-2', self.skip_distance])
        if self.basic_element is not None:
            options.extend(['-3', self.basic_element])
        if self.skip_with_unigram:
            options.append('-u')
        if self.remove_stopwords:
            options.append('-s')
        if self.stemming:
            options.append('-m')
        if self.print_when_eval is not None:
            options.append('-d')
        if self.alpha is not None:
            options.extend(['-p', self.alpha])
        if self.max_bytes is not None:
            options.extend(['-b', self.max_bytes])
        if self.max_words is not None:
            options.extend(['-l', self.max_words])
        if self.counting_unit is not None:
            options.extend(['-t', self.counting_unit])
        if self.confidence_interval is not None:
            options.extend(['-c', self.confidence_interval])
        if self.verbose:
            options.append('-v')
        if self.no_rouge_l:
            options.append('-x')
        if self.resampling_points is not None:
            options.extend(['-r', self.resampling_points])
        if self.scoring_formula is not None:
            options.extend(['-f', self.scoring_formula])

        options.append(self.config_file)

        if self.system_id is not None:
            assert not self.all_systems
            options.append(self.system_id)

        return list(map(str, options))

    def make_cmdline(self):
        return ' '.join(self.make_options())



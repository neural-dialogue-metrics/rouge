import collections
import fractions


def num_ngrams(words, n):
    """
    Return the number of nth gram of words.

    >>> num_ngrams([1, 2, 3], 3)
    1
    >>> num_ngrams([1, 2, 3], 2)
    2

    :param words: a list of tokens.
    :param n: int.
    :return:
    """
    return max(len(words) - n + 1, 0)


def get_ngram(words, n):
    """
    Return a generator on all nth grams of words.

    >>> list(get_ngram([1, 2, 3], 2))
    [(1, 2), (2, 3)]
    >>> list(get_ngram([1, 2, 3], 1))
    [(1,), (2,), (3,)]

    :param words: a list of tokens.
    :param n: int.
    :return: a generator
    """
    for i in range(num_ngrams(words, n)):
        n_gram = words[i:i + n]
        yield tuple(n_gram)


def count_ngrams(words, n):
    """
    Collect nth gram of words into a Counter.

    >>> count_ngrams([1, 1, 2, 2], 2)
    Counter({(1, 1): 1, (1, 2): 1, (2, 2): 1})
    >>> count_ngrams([1, 2, 3], 2)
    Counter({(1, 2): 1, (2, 3): 1})

    :param words:
    :param n:
    :return:
    """
    return collections.Counter(get_ngram(words, n))


def _divide_or_zero(numerator, denominator):
    """
    Divide numerator by denominator. If the latter is 0, return 0.

    >>> from fractions import Fraction
    >>> _divide_or_zero(1, 2)
    Fraction(1, 2)
    >>> _divide_or_zero(1, 0)
    Fraction(0, 1)

    :param numerator: int
    :param denominator: int
    :return: Fraction object.
    """
    if denominator == 0:
        return fractions.Fraction()
    return fractions.Fraction(numerator, denominator)


def f1_measure(numerator, r_denominator, p_denominator, alpha):
    """
    Compute a weighted F-measure.

    >>> f1_measure(1, 2, 3, 0.5)
    (0.5, 0.3333333333333333, 0.4)

    >>> f1_measure(1, 0, 1, 0.5)
    (0.0, 1.0, 0.0)

    :param numerator: 
    :param r_denominator: 
    :param p_denominator: 
    :param alpha: the weighting factor.
    :return: 3-tuple of recall, precision and f1.
    """
    recall = _divide_or_zero(numerator, r_denominator)
    precision = _divide_or_zero(numerator, p_denominator)
    f1 = (precision * recall) / ((1 - alpha) * precision + alpha * recall)
    precision = float(precision)
    recall = float(recall)
    return recall, precision, f1


def _clipped_sum(summary_ngrams, reference_ngrams):
    """
    For each instance of ngram that appear in both summary_ngrams and reference_ngrams,
    first clip their count by taking minimum value. Then sum up the clipped counts.

    >>> from collections import Counter
    >>> summary_ngrams = Counter('the police killed the gunman'.split())
    >>> summary_ngrams
    Counter({'the': 2, 'police': 1, 'killed': 1, 'gunman': 1})
    >>> reference_ngrams = Counter('gunman the police killed'.split())
    >>> reference_ngrams
    Counter({'gunman': 1, 'the': 1, 'police': 1, 'killed': 1})
    >>> summary_ngrams & reference_ngrams
    Counter({'the': 1, 'police': 1, 'killed': 1, 'gunman': 1})
    >>> _clipped_sum(summary_ngrams, reference_ngrams)
    4

    :param summary_ngrams: a Counter.
    :param reference_ngrams: a Counter
    :return: int

    """
    overlap = summary_ngrams & reference_ngrams
    return sum(overlap.values())


def rouge_n(summary, references, n=2, alpha=0.5):
    """
    Calculate ROUGE-N on already preprocessed sentences.

    >>> summary = 'gunman kill the police'.split()
    >>> reference = 'the police killed the gunman'.split()
    >>> rouge_n(summary, [reference])
    (0.25, 0.3333333333333333, 0.28571428571428575)

    :param summary: a list of tokens.
    :param references: a nested list of tokens.
    :param n: n for ngram.
    :param alpha: weight on the recall.
    :return: a 3-tuple, recall, precision and f1 measure.
    """
    summary_ngrams = count_ngrams(summary, n)
    total_matches = sum(
        _clipped_sum(summary_ngrams, count_ngrams(ref, n)) for ref in references
    )

    recall_denominator = sum(num_ngrams(ref, n) for ref in references)
    precision_denominator = len(references) * num_ngrams(summary, n)
    return f1_measure(total_matches, recall_denominator, precision_denominator, alpha)

import collections
import itertools

__all__ = ["rouge_n_sentence_level", "rouge_l_sentence_level", "rouge_l_summary_level", "rouge_n_summary_level"]


def num_ngrams(words, n):
    """
    Return the number of nth gram of words.

    >>> num_ngrams([1, 2, 3], 3)
    1
    >>> num_ngrams([1, 2, 3], 2)
    2

    :param words: a list of tokens.
    :param n: int.
    :return: number of n-gram.
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

    >>> _divide_or_zero(1, 2)
    Fraction(1, 2)
    >>> _divide_or_zero(1, 0)
    Fraction(0, 1)

    :param numerator: int
    :param denominator: int
    :return: Fraction object.
    """
    if denominator == 0:
        return 0
    return numerator / denominator


def _f1_measure(numerator, r_denominator, p_denominator, alpha):
    """
    Compute a weighted F-measure.

    >>> _f1_measure(1, 2, 3, 0.5)
    (0.5, 0.3333333333333333, 0.4)

    >>> _f1_measure(1, 0, 1, 0.5)
    (0.0, 1.0, 0.0)

    :param numerator: 
    :param r_denominator: 
    :param p_denominator: 
    :param alpha: the weighting factor.
    :return: 3-tuple of recall, precision and f1.
    """
    if alpha is None:
        alpha = 0.5
    recall = _divide_or_zero(numerator, r_denominator)
    precision = _divide_or_zero(numerator, p_denominator)
    f1 = _divide_or_zero(precision * recall, (1 - alpha) * precision + alpha * recall)
    return recall, precision, f1


def _clipped_ngram_count(summary_ngrams, reference_ngrams):
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
    >>> _clipped_ngram_count(summary_ngrams, reference_ngrams)
    4

    :param summary_ngrams: a Counter.
    :param reference_ngrams: a Counter
    :return: int

    """
    overlap = summary_ngrams & reference_ngrams
    return sum(overlap.values())


def rouge_n_sentence_level(summary_sentence, reference_sentence, n, alpha=None):
    """
    Calculate ROUGE-N on already preprocessed sentences.

    :param summary_sentence: a list of tokens.
    :param reference_sentence: a nested list of tokens.
    :param n: n for ngram.
    :param alpha: weight on the recall.
    :return: a 3-tuple, recall, precision and f1 measure.
    """
    summary_ngrams = count_ngrams(summary_sentence, n)
    reference_ngrams = count_ngrams(reference_sentence, n)
    total_matches = _clipped_ngram_count(summary_ngrams, reference_ngrams)

    recall_denominator = num_ngrams(reference_sentence, n)
    precision_denominator = num_ngrams(summary_sentence, n)
    return _f1_measure(total_matches, recall_denominator, precision_denominator, alpha)


def _flatten_sentences(sentences):
    """
    Flatten a list of sentences into a concatenated list of tokens.
    Adapted from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists.
    :param sentences: a list of sentences.
    :return: a list of tokens.

    >>> s1 = 'the gunman kill police'.split()
    >>> s2 = 'police killed the gunman'.split()
    >>> _flatten_sentences([s1, s2])
    ['the', 'gunman', 'kill', 'police', 'police', 'killed', 'the', 'gunman']

    """
    return list(itertools.chain.from_iterable(sentences))


def rouge_n_summary_level(summary_sentences, reference_sentences, n, alpha=None):
    """
    Calculate summary level ROUGE-N.
    The sentences are first flatten and then feed to rouge_n_sentence_level.
    :param summary_sentences: a list of sentences.
    :param reference_sentences: a list of sentences.
    :param n: n for ngram.
    :param alpha:
    :return:

    >>> rouge_n_summary_level()
    """
    summary_sentences = _flatten_sentences(summary_sentences)
    reference_sentences = _flatten_sentences(reference_sentences)

    return rouge_n_sentence_level(summary_sentences, reference_sentences, n, alpha)


def _compute_lcs_table(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    :param x: collection of words
    :param y: collection of words
    :return Table of dictionary of coord
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _lcs_length(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    >>> _lcs_length('ABCDE', 'CD')
    2
    >>> _lcs_length('the police killed the gunman'.split(), 'gunman police killed'.split())
    2

    :param x: sequence of words
    :param y: sequence of words
    :return: Length of LCS between x and y
    """
    table = _compute_lcs_table(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def rouge_l_sentence_level(summary_sentence, reference_sentence, alpha=None):
    """
    Calculate sentence level ROUGE-L.

    :param summary_sentence: a list of token.
    :param reference_sentence: a *single* reference, a list of tokens.
    :param alpha:
    :return:
    """
    lcs_length = _lcs_length(summary_sentence, reference_sentence)
    r_denominator = len(reference_sentence)
    p_denominator = len(summary_sentence)
    return _f1_measure(lcs_length, r_denominator, p_denominator, alpha)


def _lcs_sequence(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

    >>> _lcs_sequence('abc', 'bcd')
    [('b', 1, 0), ('c', 2, 1)]

    :param x: sequence of words
    :param y: sequence of words

    :return: a list of 3-tuple: the element, its index in x, its index in y.
    """
    m, n = len(x), len(y)
    table = _compute_lcs_table(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i - 1, j - 1)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    return _recon(m, n)


def _make_lcs_union(summary_sentences, reference_sentence):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example if
        r_i = w1 w2 w3 w4 w5
        c1 = w1 w2 w6 w7 w8
        c2 = w1 w3 w8 w9 w5
    then:
        LCS(r_i, c1) = "w1 w2"
        LCS(r_i, c2) = "w1 w3 w5"
    union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5" and
    LCS_u(r_i, C) = 4.

    >>> r_i = 'w1 w2 w3 w4 w5'.split()
    >>> c1 = 'w1 w2 w6 w7 w8'.split()
    >>> c2 = 'w1 w3 w8 w8 w5'.split()

    >>> _make_lcs_union([c1, c2], r_i)


    :param summary_sentences: The sentences that have been picked by the summarizer
    :param reference_sentence: One of the sentences in the reference summaries.

    :return: LCS_u(r_i, C)
    """

    lcs_union = set()
    for sentence in summary_sentences:
        lcs = _lcs_sequence(sentence, reference_sentence)
        lcs_set = set(ref_idx for _, _, ref_idx in lcs)
        lcs_union |= lcs_set
    return lcs_union


def _flatten_and_count_ngrams(sentences, n):
    """
    First flatten a list of sentences, then count ngrams on it.

    >>> s1 = 'the cat sat on the mat'.split()
    >>> s2 = 'the cat on the mat'.split()
    >>> _flatten_and_count_ngrams([s1, s2], 1)
    Counter({('the',): 4, ('cat',): 2, ('on',): 2, ('mat',): 2, ('sat',): 1})

    :param sentences: a list of sentences.
    :param n: N for ngrams.
    :return: Counter.
    """
    return count_ngrams(_flatten_sentences(sentences), n)


def rouge_l_summary_level(summary_sentences, reference_sentences, alpha=None):
    """
    Calculate the summary level ROUGE-L.
    :param summary_sentences: a list of sentence, each sentence is a list of tokens.
    :param reference_sentences: Same shape as summary.
    :param alpha:
    :return:
    """
    summary_unigrams = _flatten_and_count_ngrams(summary_sentences, 1)
    reference_unigrams = _flatten_and_count_ngrams(reference_sentences, 1)

    total_lcs_words = 0
    for reference in reference_sentences:
        lcs_union = _make_lcs_union(summary_sentences, reference)
        for word in lcs_union:
            unigram = (reference[word],)
            if (unigram in summary_unigrams and unigram in reference_unigrams
                    and summary_unigrams[unigram] > 0 and reference_unigrams[unigram] > 0):
                summary_unigrams[unigram] -= 1
                reference_unigrams[unigram] -= 1
                total_lcs_words += 1

    r_denominator = sum(len(sentence) for sentence in reference_sentences)
    p_denominator = sum(len(sentence) for sentence in summary_sentences)
    return _f1_measure(total_lcs_words, r_denominator, p_denominator, alpha)

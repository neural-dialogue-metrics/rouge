"""
Wrapper of Pythonrouge to provide similar interface of rouge.
"""
from pythonrouge import Pythonrouge

_METRIC_KEYS = (
    'R', 'P', 'F'
)


def _parse_output(prefix, score):
    """
    Parse the score dict returned by calc_score() into a 3-tuple
    of recall, precision, f1.
    :param prefix: Prefix of the measurement name.
    :param score: a dict.
    :return: a 3-tuple.
    """
    return tuple(score[prefix + kind] for kind in _METRIC_KEYS)


def make_rouge(**kwargs):
    """
    Create a Pythonrouge according to our settings.
    :param kwargs:
    :return:
    """
    kwargs.setdefault('summary_file_exist', False)
    kwargs.setdefault('resampling', False)
    kwargs.setdefault('stopwords', False)
    kwargs.setdefault('ROUGE_SU4', False)
    kwargs.setdefault('stemming', False)
    kwargs.setdefault('favor', True)  # use alpha
    kwargs.setdefault('word_level', True)
    kwargs.setdefault('length_limit', False)
    if 'p' not in kwargs or kwargs['p'] is None:
        kwargs['p'] = 0.5
    return Pythonrouge(**kwargs)


def make_rouge_n(summary, reference, n_gram):
    return make_rouge(summary=summary, reference=reference, n_gram=n_gram)


def make_rouge_l(summary, reference):
    return make_rouge(
        summary=summary,
        reference=reference,
        ROUGE_L=True,
        n_gram=-1,  # Suppress computation of ROUGE-N.
    )


def rouge_n_sentence_level(summary_sentence, reference_sentence, n, alpha=None):
    """
    Calculate sentence level ROUGE-N using pythonrouge.
    :param summary_sentence: a string.
    :param reference_sentence: a string.
    :param n:
    :param alpha:
    :return:
    """
    rouge = make_rouge_n(
        summary=[[summary_sentence]],
        reference=[[[reference_sentence]]],
        n_gram=n,
    )
    score = rouge.calc_score()
    prefix = 'ROUGE-%d-' % n
    return _parse_output(prefix, score)


def rouge_n_summary_level(summary_sentences, reference_sentences, n, alpha=None):
    """
    Calculate summary level ROUGE-N using wrapper.
    :param summary_sentences:
    :param reference_sentences:
    :param n:
    :param alpha:
    :return:
    """
    rouge = make_rouge_n(
        summary=[summary_sentences],
        reference=[[reference_sentences]],
        n_gram=n,
    )
    score = rouge.calc_score()
    prefix = 'ROUGE-%d-' % n
    return _parse_output(prefix, score)


def rouge_l_sentence_level(summary_sentence, reference_sentence, alpha=None):
    """
    Calculate sentence level ROUGE-L using wrapper.
    :param summary_sentence:
    :param reference_sentence:
    :param alpha:
    :return:
    """
    rouge = make_rouge_l(
        summary=[[summary_sentence]],
        reference=[[[reference_sentence]]],
    )
    prefix = 'ROUGE-L-'
    score = rouge.calc_score()
    return _parse_output(prefix, score)


def rouge_l_summary_level(summary_sentences, reference_sentences, alpha=None):
    """
    Calculate summary level ROUGE-L using wrapper.
    :param summary_sentences:
    :param reference_sentences:
    :param alpha:
    :return:
    """
    rouge = make_rouge_l(
        summary=[summary_sentences],
        reference=[[reference_sentences]],
    )
    prefix = 'ROUGE-L-'
    score = rouge.calc_score()
    return _parse_output(prefix, score)


def _get_command(rouge):
    cmd = rouge.set_command()
    return ' '.join(cmd)

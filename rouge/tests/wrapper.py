"""
Wrapper of Pythonrouge to provide similar interface of rouge.
"""
import logging
from pythonrouge import Pythonrouge

logging.basicConfig(level=logging.INFO)

_METRIC_KEYS = (
    'R', 'P', 'F'
)


# Clean this up when we have our own wrapper instead of wrapping a wrapper.

def _parse_output(prefix, score):
    """
    Parse the score dict returned by calc_score() into a 3-tuple
    of recall, precision, f1.
    :param prefix: Prefix of the measurement name.
    :param score: a dict.
    :return: a 3-tuple.
    """
    return tuple(score[prefix + kind] for kind in _METRIC_KEYS)


def _make_rouge(**kwargs):
    """
    Create a Pythonrouge according to our settings.
    :param kwargs:
    :return:
    """
    # Use Python data as input.
    kwargs.setdefault('summary_file_exist', False)

    # Disable all preprocessing and postprocessing.
    # don't remove stopwords.
    kwargs.setdefault('stopwords', False)
    # don't do stemming.
    kwargs.setdefault('stemming', False)
    # don't do bootstrap resampling
    kwargs.setdefault('resampling', False)
    # don't compute confidence interval.
    kwargs.setdefault('cf', False)

    # Disable all metrics by default.
    kwargs.setdefault('n_gram', -1)  # disable ROUGE-N
    kwargs.setdefault('ROUGE_L', False)
    kwargs.setdefault('ROUGE_SU4', False)
    kwargs.setdefault('ROUGE_W', False)

    # Use default alpha.
    kwargs.setdefault('favor', False)
    # Evaluate based on words instead of bytes.
    kwargs.setdefault('word_level', True)
    # No bytes limit.
    kwargs.setdefault('length_limit', False)

    return Pythonrouge(**kwargs)


def _make_rouge_n(summary, reference, n_gram):
    """
    Create a Pythonrouge for ROUGE-N.
    :param summary:
    :param reference:
    :param n_gram:
    :return:
    """
    return _make_rouge(
        summary=summary,
        reference=reference,
        n_gram=n_gram
    )


def _make_rouge_l(summary, reference):
    """
    Create a Pythonrouge for ROUGE-L.
    :param summary:
    :param reference:
    :return:
    """
    return _make_rouge(
        summary=summary,
        reference=reference,
        ROUGE_L=True,
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
    rouge = _make_rouge_n(
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
    rouge = _make_rouge_n(
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
    rouge = _make_rouge_l(
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
    rouge = _make_rouge_l(
        summary=[summary_sentences],
        reference=[[reference_sentences]],
    )
    prefix = 'ROUGE-L-'
    score = rouge.calc_score()
    return _parse_output(prefix, score)


def _get_command(rouge):
    """
    Return the command to invoke the perl script from a Pythonrouge.
    :param rouge:
    :return:
    """
    cmd = rouge.set_command()
    return ' '.join(cmd)


def _make_rouge_w(summary, reference):
    # Use default weight.
    return _make_rouge(
        summary=summary,
        reference=reference,
        ROUGE_W=True,
    )


def rouge_w_sentence_level(summary_sentence, reference_sentence):
    rouge = _make_rouge_w(
        summary=[[summary_sentence]],
        reference=[[[reference_sentence]]],
    )
    prefix = 'ROUGE-W-1.2-'
    score = rouge.calc_score()
    # logging.info(_get_command(rouge))
    return _parse_output(prefix, score)


def rouge_w_summary_level(summary_sentences, reference_sentences):
    rouge = _make_rouge_w(
        summary=[summary_sentences],
        reference=[[reference_sentences]],
    )
    prefix = 'ROUGE-W-1.2-'
    score = rouge.calc_score()
    # logging.info(_get_command(rouge))
    return _parse_output(prefix, score)


def _make_rouge_s(summary, reference):
    return _make_rouge(

    )

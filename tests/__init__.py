from pythonrouge import Pythonrouge

_METRIC_KEYS = (
    'R', 'P', 'F'
)

# One pair of data points.
summary = 'gunman kill the police'.split()
reference = 'the police killed the gunman'.split()



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
    return Pythonrouge(**kwargs)


def rouge_n(summary, references, n, alpha):
    """
    Calculate ROUGE-N using pythonrouge.
    Args have the same meaning as our rouge.rouge_n_sentence_level.
    :param summary:
    :param references:
    :param n:
    :param alpha:
    :return:
    """
    rouge = make_rouge(summary=[[summary]], reference=[[references]], n_gram=n, p=alpha)
    score = rouge.calc_score()
    prefix = 'ROUGE-%d-' % n
    return _parse_output(prefix, score)


def rouge_l_sentence_level(summary_sentence, reference_sentence, alpha):
    rouge = make_rouge(
        summary=[[summary_sentence]],
        reference=[[[reference_sentence]]],
        p=alpha,
        ROUGE_L=True,
    )
    prefix = 'ROUGE-L-'
    score = rouge.calc_score()
    return _parse_output(prefix, score)

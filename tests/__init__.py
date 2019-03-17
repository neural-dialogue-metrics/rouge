from pythonrouge import Pythonrouge


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
    Args have the same meaning as our rouge.rouge_n.
    :param summary:
    :param references:
    :param n:
    :param alpha:
    :return:
    """
    rouge = make_rouge(summary=[[summary]], reference=[[references]], n_gram=n, p=alpha)
    score = rouge.calc_score()
    prefix = 'ROUGE-%d-' % n
    return [score[prefix + kind] for kind in ['R', 'P', 'F']]

from pythonrouge import Pythonrouge


def make_rouge(**kwargs):
    kwargs.setdefault('summary_file_exist', False)
    kwargs.setdefault('resampling', False)
    kwargs.setdefault('stopwords', False)
    kwargs.setdefault('ROUGE_SU4', False)
    kwargs.setdefault('stemming', False)
    return Pythonrouge(**kwargs)

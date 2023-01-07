# ROUGE

This is a pure Python implementation of the ROUGE metrics family in the automatic summarization fields following the paper *ROUGE: A Package for Automatic Evaluation of Summaries Chin-Yew Lin et al.*. It is an attempt to implement these metrics correctly and elegantly in total Python. It provides the following features:

- ROUGE-N, ROUGE-L, and ROUGE-W are currently supported.
- Flexible input. For each metric supported, sentence-level and summary-level variants are provided, which means you can use them in a machine translation context with sentence pairs.
- Correctness. All the claimed implemented metrics are tested against a non-trivial amount of data, using the plain old Perl script as a baseline.
- Self-contained. The total implementation is *one single script* in *one single package*. No dependency except a Python-3 is required. _No _Perl_ script is_ involved.*
- Fast. At least faster than the Perl script wrappers. Preprocessing is the total freedom of the client. We establish API on the concept of a *sentence*, which is a list of tokens, and _sentences_, which is a list of *sentence*s. Preprocessing like *stopword removal*, *stemming* and *tokenization* is left to the client.
- Well documented. Every function has a `doctest`.
- Procedural style API. You don't need to instantiate an object. Just call the function that does the right job.

For more information about why we reimplemented this metric, please read our [rationales](docs/rationale.md).

## Usages

An example use of calculating `ROUGE-2` on the sentence level is provided:

```python
from rouge import rouge_n_sentence_level

summary_sentence = 'the capital of China is Beijing'.split()
reference_sentence = 'Beijing is the capital of China'.split()

# Calculate ROUGE-2.
recall, precision, rouge = rouge_n_sentence_level(summary_sentence, reference_sentence, 2)
print('ROUGE-2-R', recall)
print('ROUGE-2-P', precision)
print('ROUGE-2-F', rouge)

# If you just want the F-measure you can do this:
*_, rouge = rouge_n_sentence_level(summary_sentence, reference_sentence, 2)  # Requires a Python-3 to use *_.
print('ROUGE-2-R', recall)

```
For more usage examples, please refer to `example.py`.

## Install

Currently not uploaded to PyPi...
```bash
git clone https://github.com/neural-dialogue-metrics/rouge.git
python ./setup.py install
```

## Dependencies

The code is *only* tested on `python==3.6.2` but it should work with a higher version of Python. If you want to run the tests locally, you need to install [Pythonrouge](https://github.com/tagucci/pythonrouge.git), which is a wrapper on the original Perl script. To install it, please run the following commands:

```bash
# not using pip
git clone https://github.com/tagucci/pythonrouge.git
python setup.py install

# using pip
pip install git+https://github.com/tagucci/pythonrouge.git
```
Then go to the project root directory, and run:
```bash
# Run doctests.
python -m doctest ./rouge/*.py -v

# Run unittests.
python -m unittest -v
```
Since the python wrapper is generally slow, the tests take more than a few minutes to finish.

## Acknowledgment

The test data is taken literally from [sumeval](https://github.com/chakki-works/sumeval.git).
The code borrows ideas from both `sumeval` and the `rouge.py` script of [tensorflow/nmt](https://github.com/tensorflow/nmt.git).
The API style closely follows that of `nltk.translate.bleu_score`.

## References

[1] ROUGE: A Package for Automatic Evaluation of Summaries.

[2] ROUGE 2.0: Updated and Improved Measures for Evaluation of Summarization Tasks.

[3] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence and Skip-Bigram Statistics.

# ROUGE

This is a pure Python implementation of the ROUGE metrics family in the automatic summarization fields
following the paper *ROUGE: A Package for Automatic Evaluation of Summaries Chin-Yew Lin et al.*.
It is an attempt to implement these metrics correctly and elegantly in total Python. It provides the following features:
- ROUGE-N, ROUGE-L, and ROUGE-W are currently supported.
- Flexible input. For each metric supported, sentence-level and summary-level variants are provided, which means you
can use them in a machine translation context with sentence pairs.
- Correctness. All the claimed implemented metrics are tested against a non-trivial amount of data, using the plain old Perl script as a baseline.
- Self-contained. The total implementation is *one single script* in *one single package*.
No dependency except a Python-3 is required. _No _Perl_ script is_ involved.*
- Fast. At least faster than the Perl script wrappers.
Preprocessing is the total freedom of the client. We establish API on the concept of a *sentence*, which is a list of tokens, and _sentences_, which is a list of *sentence*s. Preprocessing like *stopword removal*, *stemming* and *tokenization* is
left to the client.
- Well documented. Every function is `doctest` powered.
- Procedural style API. You don't need to instantiate an object. Just call the function that does the right job.

# Usage

An example use of calculating `ROUGE-2` on a sentence level is provided:
```python
from rouge.rouge import rouge_n_sentence_level

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

# Install

Currently not uploaded to PyPi...
```bash
git clone https://github.com/neural-dialogue-metrics/rouge.git
python ./setup.py install
```

# Dependencies

The code is *only* tested on `python==3.6.2` but it should work with a higher version of Python.
If you want to run the tests locally, you need to install [Pythonrouge](https://github.com/tagucci/pythonrouge.git), which is a wrapper on the original Perl script. To install it:
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

# Rationals

In this section, we talk about the rationales for reinventing the wheels or reimplementing the ROUGE metrics in a few words.

## The Complexity

ROUGE is *not* a trivial metric. First of all, it has a lot of variants, `ROUGE-N`, `ROUGE-L`, `ROUGE-W`, `ROUGE-S`, `ROUGE-SU`, and `ROUGE-BE`.
That's why the author called it *a package*. Implementing each of them correctly is not a trivial job.
Second, ROUGE has two completely different signatures. By _signature_, I mean the shape of the input data. It has both *sentence level* and
*summary level* defined, sometimes in varying forms. Third, ROUGE can handle multiple references and multiple summaries altogether.
It handles multiple references by fixing the summary to a single one and calculating a list of values given a list of references. It then
uses a *jackknifing procedure* to reduce those values to a scalar. It handles multiple summaries by using an *A-or-B* scheme to reduce the list
of values produced by the list of summaries to a scalar. For *A-or-B*, `A` means taking the average and `B`` means taking the best one.
If you feel calm after learning these, ROUGE can handle a great deal of preprocessing on the input data and each of them can affect the final score.
Few projects on Github implement all these functionalities correctly. Some of them don't even realize the problems with signatures.

## The Plain Old Perl Script

While the plain old Perl script `ROUGE-1.5.5.pl`` implements all these things correctly, its interface is quite discouraging.
It has a long array of single-character options with some options affecting others!
It takes the input from a fixed directory format and requires a configuration file in XML, making it less usable in the context of
rapid prototyping and development. Worse still, it *does not* provide an application programmer interface even in Perl. You have no way
to use it programmatically except by launching a process, which is very expensive.

## Our Usage Scenario

Although ROUGE originated from automatic summarization, we want to use it in a sentence-oriented style.
That's why the sentence-level API is emphasized in the project. We want to evaluate the average ROUGE score on a large
number of `response-ground_truth` pairs quickly. The wrapper way is not taken for its inefficiency.

## Simplification of the Problem

The first simplification we made is to throw away any preprocessing and stick to a general representation of the _sentence_.
A sentence is just a list of strings or tokens. Since then we don't have to implement any tokenizer. We don't
*need* to because there are a lot of libraries that can do that nicely, like `nltk` and `spacy`.

The second decision is that we don't care about *multiple references and multiple summaries*.
We *only* care about *a single reference and a single summary*. Unlike BLEU, multiple references mean all the same
to every variant of ROUGE (almost always Jackknifing). And how to combine scores from different instances of summaries is not our interest.
They can be implemented in the project, however, only as extensions to the core metrics.

After making the two decisions, our code can be implemented in a clear and reusable way.
Hopefully, it will also be *extensible* because right now we haven't implemented _all_ of the ROUGE metrics.
And you know, a ROUGE can never have too many metrics!

# Acknowledgment

The test data is taken literally from [sumeval](https://github.com/chakki-works/sumeval.git).
The code borrows ideas from both `sumeval` and the `rouge.py` script of [tensorflow/nmt](https://github.com/tensorflow/nmt.git).
The API style closely follows that of `nltk.translate.bleu_score`.

# References

[1] ROUGE: A Package for Automatic Evaluation of Summaries.

[2] ROUGE 2.0: Updated and Improved Measures for Evaluation of Summarization Tasks.

[3] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence and Skip-Bigram Statistics.

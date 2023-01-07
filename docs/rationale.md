
# Rationales

In this section, we talk about the rationales for reinventing the wheels or reimplementing the ROUGE metrics in a few words.

## The Complexity

ROUGE is *not* a trivial metric. First of all, it has a lot of variants, `ROUGE-N`, `ROUGE-L`, `ROUGE-W`, `ROUGE-S`, `ROUGE-SU`, and `ROUGE-BE`. That's why the author called it *a package*. Implementing each of them correctly is not a trivial job. Second, ROUGE has two completely different signatures. By _signature_, I mean the shape of the input data. It has both *sentence level* and *summary level* defined, sometimes in varying forms. Third, ROUGE can handle multiple references and multiple summaries altogether. It handles multiple references by fixing the summary to a single one and calculating a list of values given a list of references. It then uses a *jackknifing procedure* to reduce those values to a scalar. It handles multiple summaries by using an *A-or-B* scheme to reduce the list of values produced by the list of summaries to a scalar. For *A-or-B*, `A` means taking the average and `B`` means taking the best one. If you feel calm after learning these, ROUGE can handle a great deal of preprocessing on the input data and each of them can affect the final score. Few projects on Github implement all these functionalities correctly. Some of them don't even realize the problems with signatures.

## The Plain Old Perl Script

While the plain old Perl script `ROUGE-1.5.5.pl`` implements all these things correctly, its interface is quite discouraging. It has a long array of single-character options with some options affecting others! It takes the input from a fixed directory format and requires a configuration file in XML, making it less usable in the context of rapid prototyping and development. Worse still, it *does not* provide an application programmer interface even in Perl. You have no way to use it programmatically except by launching a process, which is very expensive.

## Our Usage Scenario

Although ROUGE originated from automatic summarization, we want to use it in a sentence-oriented style. That's why the sentence-level API is emphasized in the project. We want to evaluate the average ROUGE score on a large number of `response-ground_truth` pairs quickly. The wrapper way is not taken for its inefficiency.

## Simplifying the Problems

The first simplification we made is to throw away any preprocessing and stick to a general representation of the _sentence_. A sentence is just a list of strings or tokens. Since then we don't have to implement any tokenizer. We don't *need* to because there are a lot of libraries that can do that nicely, like `nltk` and `spacy`.

The second decision is that we don't care about *multiple references and multiple summaries*. We *only* care about *a single reference and a single summary*. Unlike BLEU, multiple references mean all the same to every variant of ROUGE (almost always Jackknifing). And how to combine scores from different instances of summaries is not our interest. They can be implemented in the project, however, only as extensions to the core metrics.

After making the two decisions, our code can be implemented in a clear and reusable way. Hopefully, it will also be *extensible* because right now we haven't implemented _all_ of the ROUGE metrics.

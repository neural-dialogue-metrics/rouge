# Copyright 2019 Cong Feng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Examples."""
from rouge.rouge import rouge_n_sentence_level
from rouge.rouge import rouge_l_sentence_level
from rouge.rouge import rouge_n_summary_level
from rouge.rouge import rouge_l_summary_level
from rouge.rouge import rouge_w_sentence_level
from rouge.rouge import rouge_w_summary_level

if __name__ == '__main__':
    # The use of sentence level rouges.
    reference_sentence = 'the police killed the gunman'.split()
    summary_sentence = 'the gunman police killed'.split()

    print('Sentence level:')
    _, _, rouge_1 = rouge_n_sentence_level(summary_sentence, reference_sentence, 1)
    print('ROUGE-1: %f' % rouge_1)

    _, _, rouge_2 = rouge_n_sentence_level(summary_sentence, reference_sentence, 2)
    print('ROUGE-2: %f' % rouge_2)

    _, _, rouge_l = rouge_l_sentence_level(summary_sentence, reference_sentence)
    print('ROUGE-L: %f' % rouge_l)

    _, _, rouge_w = rouge_w_sentence_level(summary_sentence, reference_sentence)
    print('ROUGE-W: %f' % rouge_w)

    # The use of summary level rouges.
    # Each summary is a list of sentences.
    reference_sentences = [
        'The gunman was shot dead by the police before more people got hurt'.split(),
        'This tragedy causes lives of five , the gunman included'.split(),
        'The motivation of the gunman remains unclear'.split(),
    ]

    summary_sentences = [
        'Police killed the gunman . no more people got hurt'.split(),
        'Five people got killed including the gunman'.split(),
        'It is unclear why the gunman killed people'.split(),
    ]

    print('Summary level:')
    _, _, rouge_1 = rouge_n_summary_level(summary_sentences, reference_sentences, 1)
    print('ROUGE-1: %f' % rouge_1)

    _, _, rouge_2 = rouge_n_summary_level(summary_sentences, reference_sentences, 2)
    print('ROUGE-2: %f' % rouge_2)

    _, _, rouge_l = rouge_l_summary_level(summary_sentences, reference_sentences)
    print('ROUGE-L: %f' % rouge_l)

    _, _, rouge_w = rouge_w_summary_level(summary_sentences, reference_sentences)
    print('ROUGE-W: %f' % rouge_w)

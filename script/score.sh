#!/usr/bin/env bash

SUM=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses.txt_First.txt
REF=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_responses.txt
DIR=/home/cgsdfc/Result/Test

CONFIG="-N 4 -W -L"

python rouge_score.py $CONFIG $SUM $REF --output_dir $DIR

#!/usr/bin/env bash
python translate.py \
-beam_size 1 \
-model ./output/checkpoints/model_step_21000.pt \
-src ../data/news_corpus/cnndm/test.txt.tgt.bpe \
-min_length 200 \
-max_length 300 \
-random_sampling_topk 100 -random_sampling_temp 0.9 \
-batch_size 40




python classification.py \
-model ./output/gossip_clf_dropout/checkpoints_step_500.pt \
-src ./output_19500_0.9



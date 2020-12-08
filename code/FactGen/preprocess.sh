#!/usr/bin/env bash
export TYPE=$1

echo $TYPE
python preprocess.py  -train_src ../data/$TYPE/train.txt.tgt_.bpe \
 -train_tgt ../data/$TYPE/train.txt.tgt.bpe \
 -valid_src ../data/$TYPE/val.txt.tgt_.bpe \
 -valid_tgt ../data/$TYPE/val.txt.src.bpe \
 -save_data ../data/$TYPE/encoder_tgt \
 -src_seq_length_trunc 100 \
 -tgt_seq_length_trunc 300 \
 -src_vocab gpt2/vocab.txt \
 -tgt_vocab gpt2/vocab.txt \
 -dynamic_dict \
 -fixed_vocab


 ######### Fact Query

 python preprocess.py  -train_src ../data/cnndm_expand/train.txt.src.bpe \
 -train_tgt ../data/cnndm/train.txt.src.bpe \
 -valid_src ../data/cnndm_expand/val.txt.src.bpe \
 -valid_tgt ../data/cnndm/val.txt.src.bpe \
 -save_data ../data/cnndm_expand/encoder_tgt \
 -src_seq_length_trunc 200 \
 -tgt_seq_length_trunc 300 \
 -src_vocab gpt2/vocab.txt \
 -tgt_vocab gpt2/vocab.txt \
 -dynamic_dict \
 -fixed_vocab


  python preprocess.py  -train_src ../data/gossip/gpt2_train.txt.src.bpe \
  -train_label ../data/gossip/gpt2_train.txt.label \
 -train_tgt ../data/gossip/gpt2_train.txt.src.bpe \
 -valid_src ../data/gossip/gpt2_val.txt.src.bpe \
 -valid_tgt ../data/gossip/gpt2_val.txt.src.bpe \
 -valid_label ../data/gossip/gpt2_val.txt.label \
 -save_data ../data/gossip/encoder_label \
 -src_seq_length_trunc 200 \
 -tgt_seq_length_trunc 300 \
 -src_vocab gpt2/vocab.txt \
 -tgt_vocab gpt2/vocab.txt \
 -dynamic_dict \
 -fixed_vocab \
 -train_clf
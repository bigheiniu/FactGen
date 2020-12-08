export TEXT=../../data/cnndm
export OUTPUT=./output/cnndm_copy
export TRAIN_STEP=20000

onmt_preprocess -train_src $TEXT/train.txt.src \
                -train_tgt $TEXT/train.txt.tgt \
                -valid_src $TEXT/val.txt.src \
                -valid_tgt $TEXT/val.txt.tgt \
                -save_data $TEXT/CNNDM \
                -src_seq_length 10000 \
                -tgt_seq_length 10000 \
                -src_seq_length_trunc 100 \
                -tgt_seq_length_trunc 400 \
                -dynamic_dict \
                -share_vocab \
                -shard_size 100000



onmt_train -data $TEXT/CNNDM \
           -save_model $OUTPUT/checkpoint \
           -layers 4 \
           -rnn_size 512 \
           -word_vec_size 512 \
           -max_grad_norm 0 \
           -optim adam \
           -encoder_type transformer \
           -decoder_type transformer \
           -position_encoding \
           -dropout 0.2 \
           -param_init 0 \
           -warmup_steps 1 \
           -learning_rate 2 \
           -decay_method noam \
           -label_smoothing 0.1 \
           -adam_beta2 0.998 \
           -batch_size 4096 \
           -batch_type tokens \
           -normalization tokens \
           -max_generator_batches 2 \
           -train_steps $TRAIN_STEP \
           -accum_count 4 \
           -share_embeddings \
           -copy_attn \
           -param_init_glorot \
           -world_size 1 \
           -gpu_ranks 0 \
           --keep_checkpoint 1


onmt_translate -gpu 0 \
               -batch_size 20 \
               -beam_size 1 \
               -random_sampling_topk 100 \
               -model $OUTPUT/checkpoint_step_$TRAIN_STEP.pt \
               -src $TEXT/test.txt.src \
               -output cp_cnndm.out \
               -min_length 200 \
               -max_length 300 \
               -verbose \
               -stepwise_penalty \
               -coverage_penalty summary \
               -beta 5 \
               -length_penalty wu \
               -alpha 0.9 \
               -verbose \
               -ignore_when_blocking "." "</t>" "<t>"
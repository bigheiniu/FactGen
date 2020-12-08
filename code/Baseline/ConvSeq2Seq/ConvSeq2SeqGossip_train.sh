export TEXT=../../data/news_corpus/gossip
export OUTPUT=./output/gossip_convseq2seq
export OUTPUT_TEXT=gossip_convseq2seq.out

# Preprocess
fairseq-preprocess --source-lang txt.src --target-lang txt.tgt \
    --trainpref $TEXT/train --validpref $TEXT/val --testpref $TEXT/test \
    --destdir $TEXT/bin --padding-factor 1 --thresholdtgt 10 --thresholdsrc 10

# Model Training
fairseq-train $TEXT/bin -a fconv_self_att_wp --lr 0.25 --clip-norm 0.1 \
--max-tokens 1500 --lr-scheduler reduce_lr_on_plateau --decoder-attention True \
--encoder-attention False --criterion label_smoothed_cross_entropy --weight-decay .0000001 \
--label-smoothing 0 --source-lang txt.src --target-lang txt.tgt --gated-attention True \
--self-attention True --project-input True  --skip-invalid-size-inputs-valid-test \
--save-dir $OUTPUT  --keep-best-checkpoints 1 --max-epoch 10 --pretrained False


fairseq-generate $TEXT/bin --path $OUTPUT/checkpoint_best.pt --batch-size 32 \
--beam 1 --sampling --sampling-topk 10 --temperature 0.8 --nbest 1 --source-lang txt.src --target-lang txt.tgt --min-len 200 > $OUTPUT_TEXT

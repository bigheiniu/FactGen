export TRAIN_FILE=./data/news_corpus/gossip/train_fact.tsv
export TEST_FILE=./data/news_corpus/gossip/val_fact.tsv

python run_lm.py \
--output_dir=output_gpt2_gossip \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=$TRAIN_FILE \
--do_eval \
--eval_data_file=$TEST_FILE \
--evaluate_during_training \
--per_gpu_train_batch_size 2 \
--per_gpu_eval_batch_size 3 \
--gradient_accumulation_steps 10 \
--num_train_epochs 1 \
--logging_steps 100 \
--save_steps 200 \
--save_total_limit 3 \
--max_length 512 \
--overwrite_cache





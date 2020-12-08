#!/usr/bin/env bash
python run_gen.py --model_type gpt2 \
--model_name_or_path ./output_gpt2_gossip \
--length 300 \
--prompt_file ./data/gossip/test_fact.tsv \
--output_file ./generated/gossip_gen.tsv
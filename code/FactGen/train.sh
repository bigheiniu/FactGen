#!/usr/bin/env bash
export TYPE=$1
python train.py \
-config config/cnndm/transformer_cnndm_psa_$TYPE.yml \
-run_name psa \
-gpt2_params_path gpt2/models/124M/ \
-gpt2_init_embanddec
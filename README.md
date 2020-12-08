# Factual-Enhanced Synthetic News Generation

## Download GPT2 weights

`cd gpt2 && python download_model.py 124M`

## Train FactGen

### Preprocess
We provide the raw file under *../data* directory. 
0. Move to the work directory
`cd gpt2`
1. You should first encode the test into byte. 
`python encode_text.py --directory /path/to/data/directory` 
2. Zip the training data together
`python preprocess.py -train_src /path/to/data/directorytrain.txt.src.bpe -train_tgt /path/to/data/directory/train.txt.tgt.bpe -valid_src /path/to/data/directory/val.txt.src.bpe -valid_tgt data/cnndm/val.txt.tgt.bpe -save_data data/cnndm/CNNDM_BPE_COPY -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -src_vocab gpt2/vocab.txt -tgt_vocab gpt2/vocab.txt -dynamic_dict -fixed_vocab`

### Pre-Train the PSA Language model
This stage only trains with claim and news content. 
0. Move to new work Directory after preprocess 
`cd ../`
1. Pre-train the Language Model, you can specify the training hyper-parameters in *./config/\*.config* file. 
`python train.py -config config/transformer_cnndm_psa.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Pre-Train Fact Reconstructor
0. Move to the work directory 
`cd onmt/modules/`
1. Pre-train the Fact Reconstructor
`python clsAttention.py --file_path /path/to/raw/src/file`

### Train Fact Reconstructor with Language Model 
0. Move to the work directory
`cd ../../`
1. Train LM and FR together. Please specify **train_from** and **clf_path** in the configure file. It should one of the checkpoint in Pre-trained language model and fact reconstructor path respectively. 
`python train.py -config config/transformer_cnndm_psa_Fact.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`


### Train All Components Together
1. Please specify **train_from** and **data** in the configuration file. **train_from** should be new checkpoint from last step and **data** file should contain the queried fact information.
`python train.py -config config/transformer_cnndm_psa_Fact_Query.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Generation
Generation is performed via top-k/random sampling.

        python translate.py \
        -beam_size 1 \
        -model ./output/checkpoints/model_step_21000.pt \
        -src ../data/news_corpus/cnndm/test.txt.tgt.bpe \
        -min_length 200 \
        -max_length 300 \
        -random_sampling_topk 100 -random_sampling_temp 0.9 \
        -batch_size 40


## Evaluate Model

### Automatic Quality Evaluation
Please check *AutomaticEvaluation* directory for three automatic text quality evaluation scripts. 

### Detection
Please check *Detection* directory for Neural Generation Detection 

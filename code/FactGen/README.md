# Factual-Enhanced Synthetic News Generation

This was made to run with the following dependencies:

1. Install python 3.6.1.
2. Install pip 21.0.1 via `pip install --upgrade pip`.
3. Install python dependencies via `pip install -r code/requirements.txt`.

## Download GPT2 weights

`cd gpt2 && python download_model.py 124M`

## Train FactGen

### Download Datasets

The datasets used are provided [here](https://github.com/abisee/cnn-dailymail).

1. Download the processed data to `code/data/news_corpus/cnndm`.
2. Convert the format from `.bin` to `.tgt.txt` and `.src.txt` using [this](https://gist.github.com/jorgeramirez/15286b588dc2669ced95bbf6a6803420) script.

### Preprocess

We provide the raw file under *../data* directory.

1. Move to the work directory:
`cd gpt2`
2. You should first encode the test into byte.
`python encode_text.py --directory ../../data/news_corpus/cnndm`
3. Move to the FactGen directory:
`cd ..`
4. Zip the training data together:
`python preprocess.py -train_src ../data/news_corpus/cnndm/train.txt.src.bpe -train_tgt ../data/news_corpus/cnndm/train.txt.tgt.bpe -valid_src ../data/news_corpus/cnndm/val.txt.src.bpe -valid_tgt ../data/news_corpus/cnndm/val.txt.tgt.bpe -save_data ../data/news_corpus/cnndm/CNNDM_BPE_COPY -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -src_vocab gpt2/vocab.txt -tgt_vocab gpt2/vocab.txt -dynamic_dict -fixed_vocab`

### Pre-Train the PSA Language model

This stage only trains with claim and news content.

1. Pre-train the Language Model, you can specify the training hyper-parameters in *./config/\*.config* file:
`python train.py -config config/transformer_cnndm_psa.yml -run_name psa -gpt2_params_path gpt2/models/124M/ -gpt2_init_embanddec`

### Pre-Train Fact Reconstructor

0. Move to the work directory `cd onmt/modules/`
1. Pre-train the Fact Reconstructor
`python clsAttention.py --file_path /path/to/raw/src/file`

### Train Fact Reconstructor with Language Model

0. Move to the work directory `cd ../../`
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

'''
Human Written and Machine Generated Text detection;
The Classifier is trained with 2k GPT-2 output, 2k WIKI human written text and 100 FactGen generated samples
'''

from simpletransformers.classification import ClassificationModel
import pandas as pd
import argparse
def train(human_file, gen_file, our_gen_file, output_dir):
    data = []
    data += [(i.strip(), 1) for i in open(human_file,'r').readlines()]
    data += [(i.strip(), 0) for i in open(gen_file,'r').readlines()]
    data += [(i.strip(), 0) for i in open(our_gen_file,'r').readlines()]

    all_df = pd.DataFrame(data)

    train_args = {
    'overwrite_output_dir':True,
    'num_train_epochs':  10,
    'process_count': 10,
    'train_batch_size': 10,
    'eval_batch_size': 20,
    'max_seq_length': 300,
    'reprocess_input_data':True,
    'learning_rate':1e-5,
    "evaluate_during_training": True,
    "use_early_stopping":True,
    'early_stopping_patience':3,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "no_cache":True,
    'output_dir':output_dir
    }

    model = ClassificationModel('roberta', "roberta-base", args=train_args) # You can set class weights by using the optional weight argument

    # Train the model

    model.train_model(all_df)
    print("finish the training")


def eval(model_path, our_gen_file, human_file):
    gen = open(our_gen_file, 'r').readlines()
    gen = [i.strip() for i in gen]
    human = open(human_file, 'r').readlines()
    human = [i.strip() for i in human]

    assert len(human) - len(gen) == 0, "please balance the eval file"

    test_df = pd.DataFrame(gen+human)
    test_input = test_df.sample(frac=1, random_state=123)

    train_args={
        'learning_rate':3e-5,
        'num_train_epochs': 5,
        'reprocess_input_data': True,
        'overwrite_output_dir': False,
        'process_count': 10,
        'train_batch_size': 4,
        'eval_batch_size': 400,
        'max_seq_length': 300,
        "fp16":False
    }

    model = ClassificationModel('roberta', model_path, num_labels=4, use_cuda=True, cuda_device=0, args=train_args)

    result, model_outputs, wrong_predictions = model.eval_model(test_input)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['train','eval'], required=True)
    parser.add_argument("--human_file", type=str)
    parser.add_argument("--gen_file", type=str)
    parser.add_argument("--our_gen_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()
    if args.type == "train":
        train(args.human_file, args.gen_file, args.our_gen_file, args.output_dir)
    else:
        eval(args.model_path, args.our_gen_file, args.our_gen_file)

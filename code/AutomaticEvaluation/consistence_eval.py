'''
cite: https://medium.com/@vslovik/fake-news-detection-empowered-with-bert-and-friends-20397f7e1675
'''


import os
import csv
import pandas as pd
from tqdm import tqdm
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from simpletransformers.classification import ClassificationModel
import argparse
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def fnc(path_headlines, path_bodies):

    map = {'agree': 0, 'disagree':1, 'discuss':2, 'unrelated':3}

    with open(path_bodies, encoding='utf_8') as fb:  # Body ID,articleBody
        body_dict = {}
        lines_b = csv.reader(fb)
        for i, line in enumerate(tqdm(list(lines_b), ncols=80, leave=False)):
            if i > 0:
                body_id = int(line[0].strip())
                body_dict[body_id] = line[1]

    with open(path_headlines, encoding='utf_8') as fh: # Headline,Body ID,Stance
        lines_h = csv.reader(fh)
        h = []
        b = []
        l = []
        for i, line in enumerate(tqdm(list(lines_h), ncols=80, leave=False)):
            if i > 0:
                body_id = int(line[1].strip())
                labels = line[2].strip()
                if labels in map and body_id in body_dict:
                    h.append(line[0])
                    l.append(map[line[2]])
                    b.append(body_dict[body_id])
    return h, b, l


def train_stance_clf(data_dir, output_dir, **kwargs):
    headlines, bodies, labels = fnc(
        os.path.join(data_dir, 'combined_stances_train.csv'),
        os.path.join(data_dir, 'combined_bodies_train.csv')
    )

    list_of_tuples = list(zip(headlines, bodies, labels))
    df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'label'])
    train_df, val_df = train_test_split(df, random_state=123)
    train_args={
        'learning_rate':3e-5,
        'num_train_epochs': 5,
        'reprocess_input_data': True,
        'overwrite_output_dir': False,
        'process_count': 10,
        'train_batch_size': 4,
        'eval_batch_size': 20,
        'max_seq_length': 300,
        "fp16":False,
        'output_dir':output_dir
    }

    model = ClassificationModel('roberta', "roberta-base", num_labels=4, use_cuda=True, cuda_device=0, args=train_args)

    model.train_model(train_df)


def eval_stance_clf(model_path, src_path, gen_path, **kwargs):
    src = open(src_path,'r').readlines()
    gen = open(gen_path, 'r').readlines()
    gen = [i.strip() for i in gen]
    src = [i.strip() for i in src]


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

    input = [[i, j] for i, j in zip(src, gen)]
    predictions, raw_outputs = model.predict(input)
    th = Counter(predictions)
    th = sorted(th.items(), key=lambda x: x[0])
    print(th)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=['train','eval'])
    group = parser.add_argument_group("train")
    group.add_argument("--data_dir", type=str)
    group.add_argument("--output_dir", type=str, help="the output path for the stance detection model")

    group = parser.add_argument_group("eval")
    group.add_argument("--model_path", type=str, help="the path to the stance detection model")
    group.add_argument("--gen_path", type=str, help="the generated samples to fine-tune the classification mdoel")
    group.add_argument("--src_path", type=str)


    args = parser.parse_args()
    if args.task == "train":
        train_stance_clf(args.data_dir, args.output_dir)
    else:
        eval_stance_clf(args.model_path, args.gen_path, args.src_path)





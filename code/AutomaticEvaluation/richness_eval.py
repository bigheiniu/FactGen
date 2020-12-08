import spacy
import numpy as np
import pandas as pd
from spacy.lang.en import English
from multiprocessing import Process
import os
import argparse

def multiprocess_function(num_process, function_ref,args):
    jobs = []

    for idx in range(num_process):

        process = Process(target=function_ref, args=(idx,) + args)
        process.daemon = True
        jobs.append(process)
        process.start()

    for i in range(num_process):
        jobs[i].join()

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def evaluate_entity(idx, text_chunk):
    nlp = spacy.load("en_core_web_sm")
    entity_avg = []
    text_chunk_idx = text_chunk[idx]
    for line in text_chunk_idx:
        doc = nlp(line)
        entity_avg.append(len(set(i.text.lower() for i in doc.ents)))
    print(np.sum(entity_avg), "\t", len(entity_avg))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path",type=str)
    args = parser.parse_args()
    data = open(args.gen_path,'r').readlines()
    data = [" ".join(i.strip().split()) for i in data]
    # MultiProcess
    num_process = os.cpu_count() - 2
    chunk_data = chunkify(data, num_process)
    multiprocess_function(num_process, evaluate_entity, (chunk_data, ))

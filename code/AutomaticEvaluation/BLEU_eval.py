from nltk.translate.bleu_score import corpus_bleu
import rouge
import re
import pandas as pd
import argparse
def get_bleu(candidate, ref):
    candidate = [i.split() for i in candidate]
    ref = [[i.split()] for i in ref]
    score = corpus_bleu(ref, candidate)
    print("BLEU score is {}".format(score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_file",
        type=str,
        help="reference file",
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        help="generated news content file",
    )
    args = parser.parse_args()
    ref = open(args.ref_file).readlines()
    gen = open(args.gen_file).readlines()
    ref = [i.strip() for i in ref]
    gen = [i.strip() for i in gen]

    gen = [re.sub(r"<INDEX.*?|>","",i) for i in gen]
    gen = [" ".join(i.split()) for i in gen]
    get_bleu(gen, ref)


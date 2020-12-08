import sys
from transformers import GPT2Tokenizer
import regex as re
from os.path import join as osjoin
import argparse

def encode_file(directory):
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    with_tldr = False
    replace_newline = False
    tok_trunc = 1000000


    for type in ["train",'val']:
        for t in ["src",'tgt']:
        # for t in ['src']:
            filename = osjoin(directory, type+".txt."+t)
            print(filename)
            write_name = filename+'.bpe'
            if with_tldr and 'src' in filename:
                write_name += '.tldr'

            with open(filename, 'r') as f:
                with open(write_name, 'w') as fw:
                    for line in f:
                        txt = line.strip()
                        if with_tldr and 'src' in filename:
                            txt += '\nTL;DR:'

                        if replace_newline:
                            txt = txt.replace('<newline>', '\n')

                        bpe_tokens = []
                        for token in re.findall(pat, txt): # line.strip() to make sure newline is not encoded
                            token = ''.join(enc.byte_encoder[b] for b in token.encode('utf-8'))
                            bpe_tokens.extend(enc.bpe(token).split(' '))
                        fw.write(' '.join(bpe_tokens[:tok_trunc]) + '\n')

def decode_file(file_name):
    with open(file_name, 'r') as f1:
        data = f1.readlines()
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    for line in data:
        th = enc.convert_tokens_to_string(line.split())
        print(th)
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, help="the data directory")
    args = parser.parse_args()
    encode_file(args.directory)





            
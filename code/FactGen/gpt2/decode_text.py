import argparse
import pickle
from transformers import GPT2Tokenizer


enc = GPT2Tokenizer.from_pretrained('gpt2')


def decode_text(args):
    
    
    if args.dst is None:
        if args.src[-4:] == '.bpe':
            args.dst = args.src[:-4]
        elif args.src[-8:] == '.encoded':
            args.dst = args.src[:-8]
        else:
            raise ValueError('dst needed or src that ends in .bpe or .encoded')
    
    i = 0
    if "pickle" in args.src:
        with open(args.src, 'rb') as f:
            data = pickle.load(f)
    else:
        data = open(args.src, 'r').readlines()
    # print(args.dst)
    # exit()
    with open(args.dst, 'w') as fw:
        for line in data:
            i += 1
            # line = line[0]
            text = line.strip()
            text = text.replace("\x00", "")
            text = ''.join(text.split(' '))

            decoded = bytearray([enc.byte_decoder[c] for c in text]).decode('utf-8', errors=enc.errors)
            decoded = decoded.replace('\n', '') # We need one example per line
            decoded = decoded.replace('\r', '')
            decoded += '\n'
            print("The length is {}".format(len(decoded.split())))
            fw.write(decoded)
    print(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str)
    parser.add_argument('--dst', '-dst', type=str, default=None)
    
    args = parser.parse_args()
    decode_text(args)

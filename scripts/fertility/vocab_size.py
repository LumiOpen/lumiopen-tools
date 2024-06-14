#!/usr/bin/env python3

import sys

from argparse import ArgumentParser

from transformers import AutoTokenizer


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer.vocab_size)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

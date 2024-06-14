#!/usr/bin/env python3

import sys
import os
import regex

from statistics import mean
from argparse import ArgumentParser

from transformers import AutoTokenizer


WORD_RE = regex.compile(r'[[:alnum:]]+|[^[:space:]]')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--no-split', action='store_true',
                    help='process text without splitting lines')
    ap.add_argument('tokenizer')
    ap.add_argument('text', nargs='+')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    fertilities = []
    total_token_count, total_word_count = 0, 0
    for fn in args.text:
        with open(fn) as f:
            text = f.read()

        if args.no_split:
            lines = [text]
        else:
            lines = text.splitlines()

        token_count, word_count = 0, 0
        for line in lines:
            token_count += len(tokenizer(line).input_ids)
            word_count += len(WORD_RE.findall(line))

        print(f'{os.path.basename(fn)} {token_count}/{word_count} '
              f'({token_count/word_count:.2f})')

        fertilities.append(token_count/word_count)
        total_token_count += token_count
        total_word_count += word_count

    print(f'TOTAL {total_token_count}/{total_word_count} '
          f'({total_token_count/total_word_count:.2f})')
    print(f'AVERAGE {mean(fertilities):.2f}')


if __name__ == '__main__':
    sys.exit(main(sys.argv))

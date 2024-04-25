#!/usr/bin/env python3

# Fix poorly translated texts

import sys

import json

import torch

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = 'LumiOpen/Poro-34B'


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('file', nargs='?')
    return ap


def translate(stream, tokenizer, model, args):
    with open("./fix_translations.json") as file:
        data = json.load(file)

    template = data[0]
    test_inputs = data[1:]

    print(template)
    print('\n'.join(str(x) for x in test_inputs))

    for line in test_inputs:
        line = line.rstrip('\n')
        prompt = template.format(line)

        encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = model.generate(
            **encoded,
            max_length=1024,
            do_sample=True,
            repetition_penalty=2.0,
            max_new_tokens=512
        )
        decoded = tokenizer.decode(output[0])

        assert decoded.startswith(prompt)
        pred = decoded[len(prompt):]
        pred = pred.rstrip('\n')
        if pred.endswith(tokenizer.eos_token):
            pred = pred[:-len(tokenizer.eos_token)]
        pred = pred.rstrip('\n')

        print(pred)


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    print('model loaded.', file=sys.stderr)

    if args.file is None:
        translate(sys.stdin, tokenizer, model, args)
    else:
        with open(args.file) as f:
            translate(f, tokenizer, model, args)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

#!/usr/bin/env python3

# Translate between English and Finnish using the Poro model
# (https://huggingface.co/LumiOpen/Poro-34B) with the template format
# used to include translation data in the model pretraining.

import sys

import torch

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL = 'LumiOpen/Poro-34B'

TEMPLATES = {
    ('eng', 'fin'): '<|user|>Käännä suomeksi: {} <|assistant|>',
    ('fin', 'eng'): '<|user|>Käännä englanniksi: {} <|assistant|>',
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--fin-to-eng', action='store_true')
    ap.add_argument('file', nargs='?')
    return ap


def translate(stream, tokenizer, model, args):
    if args.fin_to_eng:
        template = TEMPLATES[('fin', 'eng')]
    else:
        template = TEMPLATES[('eng', 'fin')]

    for line in stream:
        line = line.rstrip('\n')
        prompt = template.format(line)

        encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = model.generate(**encoded, max_length=256)
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

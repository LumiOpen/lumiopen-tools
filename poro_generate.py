#!/usr/bin/env python3

# Adaptation of poro_translate.py for chatting with the model.

import sys

import torch

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = 'LumiOpen/Poro-34B'

TEMPLATES = {
    "default": "<|käyttäjä|>Hei, kuka sinä olet?\n<|avustaja|>Olen avustajaluonteinen keskusteleva kielimalli. Voit kysyä minulta mitä tahansa ja yritän vastata sinulle parhaani mukaan.\n<|käyttäjä|>{}\n<|avustaja|>"
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('file', nargs='?')
    return ap


def generate(stream, tokenizer, model, args):
    template = TEMPLATES["default"]

    for line in stream:
        line = line.rstrip('\n')
        prompt = template.format(line)

        encoded = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = model.generate(
            **encoded,
            max_length=128,
            min_length=1,
            top_k=10,
            repetition_penalty=2.0,
            do_sample=True
        )
        decoded = tokenizer.decode(output[0])

        assert decoded.startswith(prompt)
        pred = decoded[len(prompt):]
        pred = pred.rstrip('\n')
        if pred.endswith(tokenizer.eos_token):
            pred = pred[:-len(tokenizer.eos_token)]
        pred = pred.rstrip('\n')
        print("\n------RESPONSE BLOCK BEGIN------\n")
        print(pred)
        print("\n------ RESPONSE BLOCK END ------\n")


def main(argv):
    args = argparser().parse_args(argv[1:])

    print("Please wait until the model has loaded...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    print('The model was successfully loaded.', file=sys.stderr)

    if args.file is None:
        generate(sys.stdin, tokenizer, model, args)
    else:
        with open(args.file) as f:
            generate(f, tokenizer, model, args)


if __name__ == '__main__':
    sys.exit(main(sys.argv))

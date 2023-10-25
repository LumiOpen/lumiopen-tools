#!/usr/bin/env python3

import sys

import torch

from argparse import ArgumentParser
from logging import warning

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from utils import timed, load_model, DTYPE_MAP, DMAP_CHOICES


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--min_new_tokens', default=10, type=int)
    ap.add_argument('--max_new_tokens', default=100, type=int)
    ap.add_argument('--temperature', default=1.0, type=float)
    ap.add_argument('--num_return_sequences', default=1, type=int)
    ap.add_argument('--memory-usage', action='store_true')
    ap.add_argument('--show-devices', action='store_true')    
    ap.add_argument('--dtype', choices=DTYPE_MAP.keys(), default='fp32')
    ap.add_argument('--device-map', choices=DMAP_CHOICES, default='auto')
    ap.add_argument('--trust-remote-code', default=None, action='store_true')
    ap.add_argument('model')
    ap.add_argument('file', nargs='?')
    return ap


def report_memory_usage(message, out=sys.stderr):
    print(f'max memory allocation {message}:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)


@timed
def generate(prompts, tokenizer, model, args):
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=args.temperature,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        repetition_penalty=1.2,
    )

    for prompt in prompts:
        prompt = prompt.rstrip('\n')

        generated = pipe(prompt)
        for g in generated:
            text = g['generated_text']
            text = text.replace(prompt, f'**{prompt}**', 1)
            print(text)
            print('-'*78)


def check_devices(model, args):
    if args.show_devices:
        print(f'devices:', file=sys.stderr)
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if args.show_devices:
                print(f'  {name}.{param_name}:{param.device}', file=sys.stderr)
            elif param.device.type != 'cuda':
                warning(f'{name}.{param_name} on device {param.device}')


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)

    if args.memory_usage:
        report_memory_usage('after model load')

    check_devices(model, args)
    
    if not args.file:
        generate(sys.stdin, tokenizer, model, args)
    else:
        with open(args.file) as f:
            generate(f, tokenizer, model, args)

    if args.memory_usage:
        report_memory_usage('after generation')


if __name__ == '__main__':
    sys.exit(main(sys.argv))

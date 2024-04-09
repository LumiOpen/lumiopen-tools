#!/usr/bin/env python3

# Very basic script to calculate loss for text in JSONL (default key
# "text") using given model. Truncates text longer than model max len.

import sys
import json

import torch

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--key', default='text')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('model')
    ap.add_argument('jsonl')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype='auto',
    )

    losses = []
    with open(args.jsonl) as f:
        for ln, l in enumerate(f, start=1):
            data = json.loads(l)
            text = data[args.key]

            # Note: truncation=True
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=True,
                return_tensors='pt',
                max_length=args.max_length,
            ).to(model.device)
            
            if encoded.input_ids.shape[1] < 2:
                warning(f'skip line {ln} in {fn}: too short: "{line}"')
                continue

            with torch.no_grad():
                output = model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    labels=encoded.input_ids
                )

            loss = float(output.loss)
            if args.verbose:
                truncated = text if len(text) < 40 else text[:40] + ' [...]'
                truncated = json.dumps(truncated, ensure_ascii=False)
                print(f'{ln}: loss {loss:.2f}: {truncated}')
            losses.append(loss)

    mean = sum(losses)/len(losses)
    print(f'mean loss {mean:.2f} ({len(losses)} values)')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
